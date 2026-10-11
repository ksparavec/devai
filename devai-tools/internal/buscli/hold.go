package buscli

import (
	"context"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/nats-io/nats.go"

	"github.com/sparavec/gpu-arbiter/pkg/bus"
)

// Hold defaults (C3).
const (
	DefaultLeaseInterval  = 10 * time.Second
	DefaultConnectTimeout = 30 * time.Second
	stateRetry            = time.Second
)

// HoldOptions is one `devai-bus hold <devai.container.service> --job J
// --parent P [--lease-only]` (C3). The process-level parts (ignoring TERM
// and INT, PR_SET_PDEATHSIG, the --parent check at start) are cmd/devai-bus's.
type HoldOptions struct {
	Bus       bus.Config
	Subject   string
	Job       int64
	LeaseOnly bool

	LeaseInterval  time.Duration
	ConnectTimeout time.Duration
	// ParentAlive is polled every ParentCheck; the hold exits once it
	// returns false (its script died and it was re-parented). nil: no check.
	ParentAlive func() bool
	ParentCheck time.Duration
	// HUP ends a --lease-only hold.
	HUP    <-chan os.Signal
	Stderr io.Writer
}

type serviceKind int

const (
	kindEngine serviceKind = iota
	kindActions
	kindTrainer
)

func kindOf(container, service string) serviceKind {
	switch {
	case container == bus.ContainerOperator && service == bus.ServiceActions:
		return kindActions
	case container == bus.ContainerLayaTrainer && service == bus.ServiceTrainer:
		return kindTrainer
	}
	return kindEngine
}

// holder decides from states; seen remembers that the job's run has been
// listed since the hold started.
type holder struct {
	kind serviceKind
	job  int64
	seen bool
}

// decide returns the exit code once the state ends the hold.
func (h *holder) decide(s bus.State, fromReply bool) (int, bool) {
	switch h.kind {
	case kindEngine:
		if s.JobIs(h.job) {
			switch s.Status {
			case bus.StatusLoading, bus.StatusServing, bus.StatusWaking:
				h.seen = true
				return 0, false
			case bus.StatusFailed:
				return ExitLost, true
			}
		}
		// unloaded, gone to sleep, another job's model, stopped
		return ExitOK, true
	default:
		var run *bus.Run
		if fromReply {
			run = s.Run
		} else if r, ok := s.FindRun(h.job); ok {
			run = &r
		}
		if run == nil {
			if h.seen {
				return ExitLost, true // the service no longer knows a run the hold had seen
			}
			return ExitOK, true // the job has had no run since the hold started
		}
		switch run.Status {
		case bus.RunRunning, bus.RunStopping:
			h.seen = true
			return 0, false
		case bus.RunExited:
			if run.ExitCode != nil && *run.ExitCode == 0 {
				return ExitOK, true
			}
			return ExitFailed, true
		case bus.RunEnded:
			if run.Outcome == "failed" {
				return ExitFailed, true
			}
			return ExitOK, true
		case bus.RunLost:
			return ExitLost, true
		}
		return ExitLost, true
	}
}

// Hold renews the job's lease every LeaseInterval and returns once the
// service no longer runs the job's model or run: 0, 1 or 3 by the run's end
// (C3), 2 when no state reply came within ConnectTimeout. It survives a bus
// outage: nats.go reconnects without limit, the subscription is restored,
// and after every reconnect it sends a lease at once and asks the state
// again, because core NATS drops events published while it was away.
func Hold(ctx context.Context, o HoldOptions) int {
	if o.Stderr == nil {
		o.Stderr = io.Discard
	}
	if o.LeaseInterval <= 0 {
		o.LeaseInterval = DefaultLeaseInterval
	}
	if o.ConnectTimeout <= 0 {
		o.ConnectTimeout = DefaultConnectTimeout
	}
	if o.ParentCheck <= 0 {
		o.ParentCheck = time.Second
	}
	logf := func(format string, a ...any) { fmt.Fprintf(o.Stderr, "devai-bus hold: "+format+"\n", a...) }
	container, service, err := bus.ParseServiceSubject(o.Subject)
	if err != nil {
		logf("%v", err)
		return ExitFailed
	}
	if o.Job <= 0 {
		logf("--job must be the job's Slurm id")
		return ExitFailed
	}
	h := &holder{kind: kindOf(container, service), job: o.Job}

	reconnected := make(chan struct{}, 1)
	opts, err := o.Bus.Options("devai-bus hold")
	if err != nil {
		logf("%v", err)
		return ExitFailed
	}
	opts = append(opts,
		nats.RetryOnFailedConnect(true),
		nats.ReconnectHandler(func(*nats.Conn) {
			select {
			case reconnected <- struct{}{}:
			default:
			}
		}),
		nats.ConnectHandler(func(*nats.Conn) {
			select {
			case reconnected <- struct{}{}:
			default:
			}
		}),
	)
	nc, err := nats.Connect(o.Bus.URL, opts...)
	if err != nil {
		logf("connect: %v", err)
		return ExitUnreachable
	}
	defer nc.Close()

	events := make(chan bus.State, 64)
	if !o.LeaseOnly {
		if _, err := nc.Subscribe(bus.EventSubject(container, service), func(m *nats.Msg) {
			if s, err := bus.DecodeState(m.Data); err == nil {
				events <- s
			}
		}); err != nil {
			logf("subscribe: %v", err)
			return ExitUnreachable
		}
	}

	leaseSubject := bus.CommandSubject(container, service, bus.VerbLease)
	lease, _ := bus.Marshal(bus.Request{V: bus.Version, Job: o.Job})
	sendLease := func() {
		if err := nc.Publish(leaseSubject, lease); err != nil {
			logf("lease: %v", err)
		}
	}
	stateSubject := bus.CommandSubject(container, service, bus.VerbState)
	stateReq, _ := bus.Marshal(bus.Request{V: bus.Version, Job: o.Job})
	type stateResult struct {
		s   bus.State
		err error
	}
	stateReplies := make(chan stateResult, 4)
	askState := func(timeout time.Duration) {
		go func() {
			c, cancel := context.WithTimeout(ctx, timeout)
			defer cancel()
			msg, err := nc.RequestWithContext(c, stateSubject, stateReq)
			if err != nil {
				stateReplies <- stateResult{err: err}
				return
			}
			r, err := bus.DecodeReply(msg.Data)
			switch {
			case err != nil:
				stateReplies <- stateResult{err: err}
			case !r.OK || r.State == nil:
				stateReplies <- stateResult{err: fmt.Errorf("state refused: %v", r.Error)}
			default:
				stateReplies <- stateResult{s: *r.State}
			}
		}()
	}

	leaseTick := time.NewTicker(o.LeaseInterval)
	defer leaseTick.Stop()
	parentTick := time.NewTicker(o.ParentCheck)
	defer parentTick.Stop()

	gotFirstState := o.LeaseOnly
	// needState: a state is owed since the last (re)connect; asked again on
	// each lease tick until answered (devai-control may come back after
	// the hold does).
	needState := false
	startDeadline := time.NewTimer(o.ConnectTimeout)
	defer startDeadline.Stop()
	retryState := time.NewTimer(stateRetry)
	defer retryState.Stop()
	inFlight := false
	if !o.LeaseOnly {
		askState(stateRetry * 3)
		inFlight = true
	}
	sendLease()

	for {
		select {
		case <-ctx.Done():
			return ExitFailed
		case <-o.HUP:
			if o.LeaseOnly {
				return ExitOK
			}
		case <-parentTick.C:
			if o.ParentAlive != nil && !o.ParentAlive() {
				logf("its job script is gone; exiting")
				return ExitFailed
			}
		case <-leaseTick.C:
			sendLease()
			if needState && !inFlight {
				askState(o.LeaseInterval)
				inFlight = true
			}
		case <-reconnected:
			sendLease()
			if !o.LeaseOnly && gotFirstState {
				needState = true
				if !inFlight {
					askState(o.LeaseInterval)
					inFlight = true
				}
			}
		case <-startDeadline.C:
			if !gotFirstState {
				logf("no state reply from %s within %s: the bus could not be reached", stateSubject, o.ConnectTimeout)
				return ExitUnreachable
			}
		case <-retryState.C:
			if !gotFirstState && !inFlight {
				askState(stateRetry * 3)
				inFlight = true
			}
			if !gotFirstState {
				retryState.Reset(stateRetry)
			}
		case res := <-stateReplies:
			inFlight = false
			if res.err != nil {
				if gotFirstState {
					logf("state: %v", res.err)
				}
				continue
			}
			gotFirstState = true
			needState = false
			if code, done := h.decide(res.s, true); done {
				return code
			}
		case s := <-events:
			if s.Container != container || s.Service != service {
				continue
			}
			if code, done := h.decide(s, false); done {
				return code
			}
		}
	}
}
