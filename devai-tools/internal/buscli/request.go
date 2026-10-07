package buscli

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/nats-io/nats.go"

	"github.com/sparavec/gpu-arbiter/pkg/bus"
)

// RequestOptions is one `devai-bus request <subject> <json> --job N
// [--timeout D]` (C3).
type RequestOptions struct {
	Bus     bus.Config
	Subject string
	Body    string
	Job     int64
	Timeout time.Duration
	Stdout  io.Writer
	Stderr  io.Writer
	// RetryWait is the pause before a `load` is sent again after "no
	// responders" (devai-control restarting); 1 s by default.
	RetryWait time.Duration
}

// BuildRequest adds v and job to body, a JSON object that must carry
// neither (so the two never disagree), and checks the result strictly.
func BuildRequest(body string, job int64) ([]byte, error) {
	if job < 0 {
		return nil, errors.New("--job must be a Slurm job id, 0 for none")
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal([]byte(body), &m); err != nil || m == nil {
		return nil, fmt.Errorf("the body must be a JSON object: %q", body)
	}
	for _, k := range []string{"v", "job"} {
		if _, ok := m[k]; ok {
			return nil, fmt.Errorf("the body carries %q; devai-bus adds v and job (--job) itself", k)
		}
	}
	m["v"] = json.RawMessage(`1`)
	m["job"] = json.RawMessage(fmt.Sprint(job))
	data, err := json.Marshal(m)
	if err != nil {
		return nil, err
	}
	if _, err := bus.DecodeRequest(data); err != nil {
		return nil, err
	}
	if len(data) > bus.MaxMessageBytes {
		return nil, fmt.Errorf("request of %d bytes exceeds the bus limit", len(data))
	}
	return data, nil
}

func printReply(w io.Writer, r bus.Reply) {
	b, _ := json.Marshal(r)
	fmt.Fprintln(w, string(b))
}

// Request sends one command and prints the reply. For `load` (a wake
// included) it subscribes to the service's events first: a `serving` event
// for its job counts as an ok reply and a `failed` one as load_failed, so a
// reply lost with a restarted devai-control or a bus restart does not leave
// the job waiting out its timeout (C1, "Adoption after a restart"; C3).
func Request(ctx context.Context, o RequestOptions) int {
	if o.Stdout == nil {
		o.Stdout = io.Discard
	}
	if o.Stderr == nil {
		o.Stderr = io.Discard
	}
	if o.RetryWait <= 0 {
		o.RetryWait = time.Second
	}
	fail := func(code int, format string, a ...any) int {
		fmt.Fprintf(o.Stderr, "devai-bus request: "+format+"\n", a...)
		return code
	}
	container, service, verb, err := bus.ParseCommandSubject(o.Subject)
	if err != nil {
		return fail(ExitFailed, "%v", err)
	}
	data, err := BuildRequest(o.Body, o.Job)
	if err != nil {
		return fail(ExitFailed, "%v", err)
	}
	ctx, cancel := context.WithTimeout(ctx, o.Timeout)
	defer cancel()

	isLoad := verb == bus.VerbLoad
	reconnected := make(chan struct{}, 1)
	nc, err := bus.Connect(o.Bus, "devai-bus request", nats.ReconnectHandler(func(*nats.Conn) {
		select {
		case reconnected <- struct{}{}:
		default:
		}
	}))
	if err != nil {
		return fail(ExitUnreachable, "%v", err)
	}
	defer nc.Close()

	events := make(chan bus.State, 16)
	if isLoad {
		_, err := nc.Subscribe(bus.EventSubject(container, service), func(m *nats.Msg) {
			if s, err := bus.DecodeState(m.Data); err == nil && s.JobIs(o.Job) {
				select {
				case events <- s:
				default:
				}
			}
		})
		if err != nil {
			return fail(ExitUnreachable, "subscribe: %v", err)
		}
		if err := nc.FlushWithContext(ctx); err != nil {
			return fail(ExitUnreachable, "%v", err)
		}
	}

	type result struct {
		reply bus.Reply
		err   error
	}
	send := func() <-chan result {
		ch := make(chan result, 1)
		go func() {
			msg, err := nc.RequestWithContext(ctx, o.Subject, data)
			if err != nil {
				ch <- result{err: err}
				return
			}
			r, err := bus.DecodeReply(msg.Data)
			ch <- result{reply: r, err: err}
		}()
		return ch
	}
	// After a reconnect the reply and the event may both be gone: ask the
	// state, as the hold does.
	askState := func() <-chan result {
		ch := make(chan result, 1)
		go func() {
			req, _ := json.Marshal(bus.Request{V: bus.Version, Job: o.Job})
			msg, err := nc.RequestWithContext(ctx, bus.CommandSubject(container, service, bus.VerbState), req)
			if err != nil {
				ch <- result{err: err}
				return
			}
			r, err := bus.DecodeReply(msg.Data)
			ch <- result{reply: r, err: err}
		}()
		return ch
	}

	replies := send()
	var stateReplies <-chan result
	for {
		select {
		case <-ctx.Done():
			return fail(ExitFailed, "no reply from %s within %s", o.Subject, o.Timeout)
		case s := <-events:
			if r, done := loadOutcome(s); done {
				fmt.Fprintf(o.Stderr, "devai-bus request: no reply yet; took job %d's %s event as the reply\n", o.Job, s.Status)
				printReply(o.Stdout, r)
				return exitFor(r)
			}
		case <-reconnected:
			if isLoad {
				stateReplies = askState()
			}
		case res := <-stateReplies:
			stateReplies = nil
			if res.err == nil && res.reply.OK && res.reply.State != nil && res.reply.State.JobIs(o.Job) {
				if r, done := loadOutcome(*res.reply.State); done {
					fmt.Fprintf(o.Stderr, "devai-bus request: no reply after the reconnect; took job %d's %s state as the reply\n", o.Job, res.reply.State.Status)
					printReply(o.Stdout, r)
					return exitFor(r)
				}
			}
		case res := <-replies:
			replies = nil
			switch {
			case res.err == nil:
				printReply(o.Stdout, res.reply)
				return exitFor(res.reply)
			case errors.Is(res.err, nats.ErrNoResponders) && isLoad:
				// devai-control is restarting: nobody received it, so it is
				// safe to send again.
				t := time.NewTimer(o.RetryWait)
				select {
				case <-t.C:
					replies = send()
				case <-ctx.Done():
					t.Stop()
				case s := <-events:
					t.Stop()
					if r, done := loadOutcome(s); done {
						fmt.Fprintf(o.Stderr, "devai-bus request: took job %d's %s event as the reply\n", o.Job, s.Status)
						printReply(o.Stdout, r)
						return exitFor(r)
					}
					replies = send()
				}
			case errors.Is(res.err, nats.ErrNoResponders):
				return fail(ExitFailed, "no service answers %s", o.Subject)
			case errors.Is(res.err, context.DeadlineExceeded) || errors.Is(res.err, nats.ErrTimeout):
				return fail(ExitFailed, "no reply from %s within %s", o.Subject, o.Timeout)
			case isLoad:
				// The connection dropped with the request in flight; the
				// event or the state after the reconnect decides.
				fmt.Fprintf(o.Stderr, "devai-bus request: %v; waiting for the service's state\n", res.err)
			default:
				return fail(ExitFailed, "%v", res.err)
			}
		}
	}
}

// loadOutcome turns a state for the job into the reply a load would have
// got: serving -> ok, failed -> load_failed.
func loadOutcome(s bus.State) (bus.Reply, bool) {
	switch s.Status {
	case bus.StatusServing:
		return bus.OKReply(s), true
	case bus.StatusFailed:
		msg, tail := "the load failed", ""
		if s.Error != nil {
			msg, tail = s.Error.Message, s.Error.LogTail
		}
		return bus.ErrReply(bus.CodeLoadFailed, msg, tail), true
	}
	return bus.Reply{}, false
}

func exitFor(r bus.Reply) int {
	if r.OK {
		return ExitOK
	}
	return ExitFailed
}

// PublishOptions is `devai-bus publish <subject> <json>`: an event, sent as
// given (the guard's devai.events.slurm.guard).
type PublishOptions struct {
	Bus     bus.Config
	Subject string
	Body    string
	Stderr  io.Writer
}

// Publish sends one message and flushes it.
func Publish(ctx context.Context, o PublishOptions) int {
	if o.Stderr == nil {
		o.Stderr = io.Discard
	}
	if !json.Valid([]byte(o.Body)) {
		fmt.Fprintf(o.Stderr, "devai-bus publish: the body is not JSON\n")
		return ExitFailed
	}
	asyncErr := make(chan error, 4)
	nc, err := bus.Connect(o.Bus, "devai-bus publish", nats.ErrorHandler(func(_ *nats.Conn, _ *nats.Subscription, err error) {
		select {
		case asyncErr <- err:
		default:
		}
	}))
	if err != nil {
		fmt.Fprintf(o.Stderr, "devai-bus publish: %v\n", err)
		return ExitUnreachable
	}
	defer nc.Close()
	if err := nc.Publish(o.Subject, []byte(o.Body)); err != nil {
		fmt.Fprintf(o.Stderr, "devai-bus publish: %v\n", err)
		return ExitFailed
	}
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	if err := nc.FlushWithContext(ctx); err != nil {
		fmt.Fprintf(o.Stderr, "devai-bus publish: %v\n", err)
		return ExitUnreachable
	}
	// nats.go records a permissions violation as the connection's last
	// error before it handles the flush's PONG, which follows the -ERR on
	// the wire; the async handler may run later.
	if err := nc.LastError(); err != nil {
		fmt.Fprintf(o.Stderr, "devai-bus publish: %v\n", err)
		return ExitFailed
	}
	select {
	case err := <-asyncErr:
		fmt.Fprintf(o.Stderr, "devai-bus publish: %v\n", err)
		return ExitFailed
	default:
	}
	return ExitOK
}
