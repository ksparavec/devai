package control

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/sparavec/gpu-arbiter/pkg/bus"
)

// program is the `program` kind: one supervisord program, which the
// operator's supervisor verbs start, stop, restart and reload (C1).
type program struct {
	c    *Controller
	name string
	cfg  ServiceConfig

	mu     sync.Mutex
	status string
	since  time.Time
	err    *bus.Error
}

func newProgram(c *Controller, name string, sc ServiceConfig) *program {
	return &program{c: c, name: name, cfg: sc, status: bus.StatusStopped, since: c.now()}
}

// statusOf maps supervisord's process states onto C1's statuses.
func statusOf(state string) string {
	switch state {
	case "RUNNING":
		return bus.StatusServing
	case "STARTING", "BACKOFF":
		return bus.StatusLoading
	case "STOPPED", "STOPPING", "EXITED":
		return bus.StatusStopped
	}
	return bus.StatusFailed // FATAL, UNKNOWN
}

func (p *program) refresh(ctx context.Context) {
	cctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	info, err := p.c.sup.GetProcessInfo(cctx, p.cfg.Program)
	cancel()
	p.mu.Lock()
	old := p.status
	if err != nil {
		p.status = bus.StatusFailed
		p.err = &bus.Error{Code: bus.CodeBadRequest, Message: err.Error()}
	} else {
		p.status = statusOf(info.StateName)
		p.err = nil
		if p.status == bus.StatusFailed && info.SpawnErr != "" {
			p.err = &bus.Error{Code: bus.CodeLoadFailed, Message: info.SpawnErr}
		}
	}
	if p.status != old {
		p.since = p.c.now()
	}
	changed := p.status != old
	p.mu.Unlock()
	if changed {
		p.c.publish(p.name)
	}
}

func (p *program) event() bus.State {
	p.mu.Lock()
	defer p.mu.Unlock()
	return bus.State{Container: p.c.cfg.Container, Service: p.name, Status: p.status,
		Since: bus.Timestamp(p.since), Error: p.err}
}

func (p *program) handle(ctx context.Context, verb string, req bus.Request) (bus.Reply, bool) {
	var err error
	switch verb {
	case bus.VerbState:
	case bus.VerbStart:
		err = p.c.sup.StartProcess(ctx, p.cfg.Program, true)
		if IsFault(err, FaultAlreadyStarted) {
			err = nil
		}
	case bus.VerbStop:
		err = p.c.sup.StopProcess(ctx, p.cfg.Program, true)
		if IsFault(err, FaultNotRunning) {
			err = nil
		}
	case bus.VerbRestart:
		if err = p.c.sup.StopProcess(ctx, p.cfg.Program, true); IsFault(err, FaultNotRunning) {
			err = nil
		}
		if err == nil {
			err = p.c.sup.StartProcess(ctx, p.cfg.Program, true)
		}
	case bus.VerbReload:
		// Re-read the program's configuration; restart it if it changed.
		var touched []string
		touched, err = p.c.sup.Update(ctx)
		if err == nil {
			for _, g := range touched {
				if g == p.cfg.Program {
					err = p.c.sup.StartProcess(ctx, p.cfg.Program, true)
					if IsFault(err, FaultAlreadyStarted) {
						err = nil
					}
				}
			}
		}
	case bus.VerbLease:
		return bus.Reply{}, false
	default:
		return unknownVerb("program", verb), true
	}
	p.refresh(ctx)
	if err != nil {
		return bus.ErrReply(bus.CodeLoadFailed, fmt.Sprintf("%s %s: %v", verb, p.cfg.Program, err), ""), true
	}
	return bus.OKReply(p.event()), true
}

func (p *program) tick(ctx context.Context, _ time.Time) { p.refresh(ctx) }

func (p *program) persist() json.RawMessage { return json.RawMessage(`{}`) }

func (p *program) adopt(ctx context.Context, _ json.RawMessage) {}
