package bus_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/nats-io/nats-server/v2/server"
	"github.com/nats-io/nats.go"

	"github.com/sparavec/gpu-arbiter/pkg/bus"
	"github.com/sparavec/gpu-arbiter/pkg/bus/bustest"
)

// connect connects as user and records the server's permission errors.
func connect(t *testing.T, s *bustest.Server, user string) (*nats.Conn, *permErrors) {
	t.Helper()
	pe := &permErrors{}
	nc, err := bus.Connect(s.Config(user), "test-"+user, nats.ErrorHandler(func(_ *nats.Conn, _ *nats.Subscription, err error) {
		pe.add(err)
	}))
	if err != nil {
		t.Fatalf("connect as %s: %v", user, err)
	}
	t.Cleanup(nc.Close)
	return nc, pe
}

type permErrors struct {
	mu   sync.Mutex
	errs []string
}

func (p *permErrors) add(err error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.errs = append(p.errs, err.Error())
}

func (p *permErrors) wait(t *testing.T, substr string) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		p.mu.Lock()
		for _, e := range p.errs {
			if strings.Contains(strings.ToLower(e), strings.ToLower(substr)) {
				p.mu.Unlock()
				return
			}
		}
		p.mu.Unlock()
		time.Sleep(20 * time.Millisecond)
	}
	t.Errorf("no permission error containing %q (got %v)", substr, p.errs)
}

// receives reports whether sub gets a message within d.
func receives(sub *nats.Subscription, d time.Duration) bool {
	_, err := sub.NextMsg(d)
	return err == nil
}

func TestRenderedTemplateHasEveryC1UserAndFifteenMinuteResponses(t *testing.T) {
	s := bustest.Start(t)
	want := []string{"buslog", "control-engines", "control-laya", "control-operator", "jobs", "operator", "router"}
	if strings.Join(s.Users, ",") != strings.Join(want, ",") {
		t.Fatalf("users = %v, want %v", s.Users, want)
	}
	opts, err := server.ProcessConfigFile(s.ConfPath)
	if err != nil {
		t.Fatal(err)
	}
	for _, u := range opts.Users {
		ctl := strings.HasPrefix(u.Username, "control-")
		if ctl {
			if u.Permissions == nil || u.Permissions.Response == nil {
				t.Errorf("%s: no allow_responses", u.Username)
				continue
			}
			if u.Permissions.Response.MaxMsgs != 1 || u.Permissions.Response.Expires != 15*time.Minute {
				t.Errorf("%s: allow_responses = %+v, want max 1, expires 15m", u.Username, u.Permissions.Response)
			}
		} else if u.Permissions != nil && u.Permissions.Response != nil {
			t.Errorf("%s has allow_responses", u.Username)
		}
	}
	if opts.MaxPayload != bus.MaxMessageBytes {
		t.Errorf("max_payload = %d, want %d", opts.MaxPayload, bus.MaxMessageBytes)
	}
}

func TestJobsRequestAndControlReplyWithinPermissions(t *testing.T) {
	s := bustest.Start(t)
	ctl, ctlErr := connect(t, s, "control-engines")
	jobs, _ := connect(t, s, "jobs")

	_, err := ctl.Subscribe("devai.engines.>", func(m *nats.Msg) {
		_ = m.Respond([]byte(`{"v":1,"ok":true,"state":{"status":"serving"}}`))
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := ctl.Flush(); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	r, err := bus.Do(ctx, jobs, "devai.engines.fake.load", bus.Request{V: 1, Job: 9})
	if err != nil {
		t.Fatalf("jobs -> load -> control reply: %v", err)
	}
	if !r.OK {
		t.Errorf("reply = %+v", r)
	}
	ctlErr.mu.Lock()
	defer ctlErr.mu.Unlock()
	if len(ctlErr.errs) != 0 {
		t.Errorf("control user hit permission errors replying: %v", ctlErr.errs)
	}
}

// Phase 5 step 5: a jobs client cannot publish devai.engines.ollama.restart.
func TestJobsCannotSendSupervisorVerbs(t *testing.T) {
	s := bustest.Start(t)
	ctl, _ := connect(t, s, "control-engines")
	jobs, jobsErr := connect(t, s, "jobs")
	sub, err := ctl.SubscribeSync("devai.engines.>")
	if err != nil {
		t.Fatal(err)
	}
	_ = ctl.Flush()
	for _, verb := range []string{"restart", "start", "stop", "reload"} {
		if err := jobs.Publish("devai.engines.ollama."+verb, []byte(`{"v":1,"job":0}`)); err != nil {
			t.Fatal(err)
		}
	}
	_ = jobs.Flush()
	if receives(sub, 500*time.Millisecond) {
		t.Error("devai-control received a supervisor verb a jobs client sent")
	}
	jobsErr.wait(t, "permissions violation for publish")
}

func TestRouterMayOnlyAskStateAndUnload(t *testing.T) {
	s := bustest.Start(t)
	router, routerErr := connect(t, s, "router")
	ctl, _ := connect(t, s, "control-engines")
	cmds, _ := ctl.SubscribeSync("devai.engines.>")
	_ = ctl.Flush()

	_ = router.Publish("devai.engines.vllm.state", []byte(`{}`))
	_ = router.Publish("devai.engines.vllm.unload", []byte(`{}`))
	_ = router.Flush()
	for i := 0; i < 2; i++ {
		if !receives(cmds, 2*time.Second) {
			t.Fatalf("message %d of the router's allowed verbs did not arrive", i)
		}
	}
	_ = router.Publish("devai.engines.vllm.load", []byte(`{}`))
	_ = router.Flush()
	if receives(cmds, 500*time.Millisecond) {
		t.Error("the router could send load")
	}
	routerErr.wait(t, "permissions violation for publish")
}

func TestUsersCannotReadEachOthersInbox(t *testing.T) {
	s := bustest.Start(t)
	jobs, jobsErr := connect(t, s, "jobs")
	if _, err := jobs.SubscribeSync("_INBOX.router.>"); err != nil {
		t.Fatal(err)
	}
	_ = jobs.Flush()
	jobsErr.wait(t, "permissions violation for subscription")
}

func TestBuslogPublishesNothing(t *testing.T) {
	s := bustest.Start(t)
	log, logErr := connect(t, s, "buslog")
	op, _ := connect(t, s, "operator")
	sub, _ := op.SubscribeSync("devai.events.>")
	_ = op.Flush()
	_ = log.Publish("devai.events.engines.fake", []byte(`{}`))
	_ = log.Flush()
	if receives(sub, 500*time.Millisecond) {
		t.Error("buslog published an event")
	}
	logErr.wait(t, "permissions violation for publish")
}

func TestControlUserPublishesOnlyItsOwnEvents(t *testing.T) {
	s := bustest.Start(t)
	ctl, ctlErr := connect(t, s, "control-engines")
	op, _ := connect(t, s, "operator")
	sub, _ := op.SubscribeSync("devai.events.>")
	_ = op.Flush()
	_ = ctl.Publish("devai.events.engines.fake", []byte(`{}`))
	_ = ctl.Publish("devai.events.operator.actions", []byte(`{}`))
	_ = ctl.Flush()
	m, err := sub.NextMsg(2 * time.Second)
	if err != nil || m.Subject != "devai.events.engines.fake" {
		t.Fatalf("own event: %v %v", m, err)
	}
	if receives(sub, 500*time.Millisecond) {
		t.Error("control-engines published an operator event")
	}
	ctlErr.wait(t, "permissions violation for publish")
}

func TestNoRespondersIsDistinct(t *testing.T) {
	s := bustest.Start(t)
	jobs, _ := connect(t, s, "jobs")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_, err := bus.Do(ctx, jobs, "devai.engines.vllm.state", bus.Request{V: 1})
	if !errors.Is(err, nats.ErrNoResponders) {
		t.Errorf("err = %v, want no responders", err)
	}
}
