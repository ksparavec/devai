package control

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/nats-io/nats.go"

	"github.com/sparavec/devai-tools/internal/buscli"
	"github.com/sparavec/gpu-arbiter/pkg/bus"
	"github.com/sparavec/gpu-arbiter/pkg/bus/bustest"
)

const fakeSubject = "devai.engines.fake"

func TestFakeEngineEveryVerbAndOwnership(t *testing.T) {
	srv := bustest.Start(t)
	r := startRig(t, srv, engineConfig(t, srv, 0), nil, nil)

	code, rep := r.requestAsJob(fakeSubject+".load", loadBody(30), 7, 10*time.Second)
	if code != 0 || !rep.OK || rep.State.Status != bus.StatusServing || !rep.State.JobIs(7) {
		t.Fatalf("load: exit %d, reply %+v", code, rep)
	}
	if r.mem.get() != 2+2048 {
		t.Errorf("card = %d MiB, want the 2 GiB held", r.mem.get())
	}
	// The same launch again for the same job: a no-op.
	if code, rep := r.requestAsJob(fakeSubject+".load", loadBody(30), 7, 5*time.Second); code != 0 || !rep.OK {
		t.Errorf("repeated load: exit %d %+v", code, rep)
	}
	// Another job: busy; its unload and sleep: not_owner.
	if code, rep := r.requestAsJob(fakeSubject+".load", loadBody(30), 8, 5*time.Second); code != 1 || rep.Error.Code != bus.CodeBusy {
		t.Errorf("load by another job: exit %d %+v", code, rep)
	}
	for _, verb := range []string{"unload", "sleep"} {
		if code, rep := r.requestAsJob(fakeSubject+"."+verb, `{}`, 8, 5*time.Second); code != 1 || rep.Error.Code != bus.CodeNotOwner {
			t.Errorf("%s by another job: exit %d %+v", verb, code, rep)
		}
	}
	if code, rep := r.requestAsJob(fakeSubject+".sleep", `{}`, 7, 5*time.Second); code != 1 || rep.Error.Code != bus.CodeUnsupported {
		t.Errorf("sleep on the fake engine: exit %d %+v", code, rep)
	}
	for _, verb := range []string{"restart", "reload"} {
		if rep := r.ask("operator", "fake", verb, bus.Request{}, 5*time.Second); rep.OK || rep.Error.Code != bus.CodeBusy {
			t.Errorf("%s on a serving engine: %+v", verb, rep)
		}
	}
	// lease: published, no reply.
	nc := r.connect("jobs")
	if err := bus.Publish(nc, fakeSubject+".lease", bus.Request{V: 1, Job: 7}); err != nil {
		t.Fatal(err)
	}
	if rep := r.ask("operator", "fake", "nonsense", bus.Request{}, 5*time.Second); rep.OK || rep.Error.Code != bus.CodeBadRequest {
		t.Errorf("unknown verb: %+v", rep)
	}
	// Bad requests: another backend, a timeout above 840.
	if code, rep := r.requestAsJob(fakeSubject+".load", strings.Replace(loadBody(30), `"backend":"fake"`, `"backend":"vllm"`, 1), 9, 5*time.Second); code != 1 || rep.Error.Code != bus.CodeBadRequest {
		t.Errorf("load with backend vllm: exit %d %+v", code, rep)
	}
	if code, rep := r.requestAsJob(fakeSubject+".load", loadBody(900), 9, 5*time.Second); code != 1 || rep.Error.Code != bus.CodeBadRequest {
		t.Errorf("load with timeout_s 900: exit %d %+v", code, rep)
	}

	code, rep = r.requestAsJob(fakeSubject+".unload", `{}`, 7, 10*time.Second)
	if code != 0 || rep.State.Status != bus.StatusIdle || rep.State.Job != nil {
		t.Fatalf("unload: exit %d %+v", code, rep)
	}
	if r.mem.get() != 2 {
		t.Errorf("card = %d MiB after the unload", r.mem.get())
	}
	if code, rep := r.requestAsJob(fakeSubject+".unload", `{}`, 7, 5*time.Second); code != 0 || !rep.OK {
		t.Errorf("unload with nothing loaded: exit %d %+v", code, rep)
	}
	if rep := r.ask("operator", "fake", "start", bus.Request{}, 5*time.Second); !rep.OK || rep.State.Status != bus.StatusIdle {
		t.Errorf("start on an idle engine: %+v", rep)
	}
	if rep := r.ask("operator", "fake", "restart", bus.Request{}, 5*time.Second); !rep.OK {
		t.Errorf("restart on an idle engine: %+v", rep)
	}
}

func TestLeaseExpiryUnloads(t *testing.T) {
	srv := bustest.Start(t)
	r := startRig(t, srv, engineConfig(t, srv, 0), nil, nil)
	if code, _ := r.requestAsJob(fakeSubject+".load", loadBody(30), 7, 10*time.Second); code != 0 {
		t.Fatalf("load: exit %d", code)
	}
	served := time.Now()
	r.waitStatus("fake", bus.StatusIdle, testLease+3*time.Second)
	if d := time.Since(served); d < testLease {
		t.Errorf("unloaded %s after serving, before the %s lease ran out", d, testLease)
	}
	if r.mem.get() != 2 {
		t.Errorf("card = %d MiB", r.mem.get())
	}
}

// Phase 5 step 5, "the lease clock": a load that takes three test leases
// reaches serving and is not unloaded while its job then leases; with the
// job script killed during that load, the model reaches serving and is
// unloaded one test lease later.
func TestLeaseClockCountsFromTheAnswer(t *testing.T) {
	srv := bustest.Start(t)
	delay := 3 * testLease
	r := startRig(t, srv, engineConfig(t, srv, delay.Seconds()), nil, nil)

	t.Run("job leases after its load", func(t *testing.T) {
		code, rep := r.requestAsJob(fakeSubject+".load", loadBody(60), 7, 30*time.Second)
		if code != 0 || rep.State.Status != bus.StatusServing {
			t.Fatalf("load: exit %d %+v", code, rep)
		}
		h := r.startHold(fakeSubject, 7, nil)
		if !h.running(2 * testLease) {
			t.Fatalf("the hold ended: %s", h.err.String())
		}
		if st := r.state("fake"); st.Status != bus.StatusServing || !st.JobIs(7) {
			t.Fatalf("after two leases of holding: %+v", st)
		}
		r.requestAsJob(fakeSubject+".unload", `{}`, 7, 10*time.Second)
		if c := h.wait(t, 3*time.Second); c != 0 {
			t.Errorf("hold exit %d after the unload, want 0", c)
		}
	})

	t.Run("job killed during its load", func(t *testing.T) {
		events := r.watch("operator", "devai.events.engines.fake")
		ctx, cancel := context.WithCancel(context.Background())
		go buscli.Request(ctx, buscli.RequestOptions{Bus: srv.Config("jobs"), Subject: fakeSubject + ".load",
			Body: loadBody(60), Job: 8, Timeout: 60 * time.Second, Stdout: discard(), Stderr: discard()})
		time.Sleep(time.Second)
		cancel() // the job script is SIGKILLed; no epilog
		st := r.waitStatus("fake", bus.StatusServing, delay+3*time.Second)
		if !st.JobIs(8) {
			t.Fatalf("serving %+v", st)
		}
		servedAt := time.Now()
		r.waitStatus("fake", bus.StatusIdle, testLease+3*time.Second)
		d := time.Since(servedAt)
		if d < testLease-500*time.Millisecond {
			t.Errorf("unloaded %s after serving, before one lease", d)
		}
		for _, s := range events.states(8) {
			if s == bus.StatusFailed {
				t.Errorf("job 8's model went failed: %v", events.states(8))
			}
		}
	})
}

// Phase 5 step 5, adoption: devai-control killed during a load; the
// restarted one publishes serving for the same job, the job's request
// returns ok from that event, and the hold starts.
func TestAdoptionOfALoadAfterARestart(t *testing.T) {
	srv := bustest.Start(t)
	cfg := engineConfig(t, srv, 4)
	r := startRig(t, srv, cfg, nil, nil)

	result := make(chan [2]any, 1)
	go func() {
		code, rep := r.requestAsJob(fakeSubject+".load", loadBody(60), 21, 30*time.Second)
		result <- [2]any{code, rep}
	}()
	time.Sleep(time.Second)
	if st := r.state("fake"); st.Status != bus.StatusLoading {
		t.Fatalf("before the kill: %+v", st)
	}
	r.kill()
	r2 := startRig(t, srv, cfg, r.sup, r.mem)
	var got [2]any
	select {
	case got = <-result:
	case <-time.After(15 * time.Second):
		t.Fatalf("the load request did not end\n%s", r2.logs.String())
	}
	if code := got[0].(int); code != 0 {
		t.Fatalf("load request exit %d, want 0 from the serving event: %+v", code, got[1])
	}
	rep := got[1].(bus.Reply)
	if !rep.OK || !rep.State.JobIs(21) {
		t.Errorf("reply %+v", rep)
	}
	h := r2.startHold(fakeSubject, 21, nil)
	if !h.running(2 * testLease) {
		t.Fatalf("the hold ended: %s", h.err.String())
	}
	if st := r2.state("fake"); st.Status != bus.StatusServing || !st.JobIs(21) {
		t.Errorf("after adoption: %+v", st)
	}
}

func TestAdoptionRecordsAGoneProgramFailed(t *testing.T) {
	srv := bustest.Start(t)
	cfg := engineConfig(t, srv, 0)
	r := startRig(t, srv, cfg, nil, nil)
	if code, _ := r.requestAsJob(fakeSubject+".load", loadBody(30), 5, 10*time.Second); code != 0 {
		t.Fatal("load")
	}
	r.kill()
	r.sup.crash("fake")
	r2 := startRig(t, srv, cfg, r.sup, r.mem)
	if st := r2.state("fake"); st.Status != bus.StatusFailed || !st.JobIs(5) {
		t.Errorf("after restart with the program gone: %+v", st)
	}
}

// Phase 5 step 5: unload during a load stops it, answers the waiting load
// load_failed (unloaded), leaves idle and never failed; a second unload
// during the first returns with it; one after it returns ok at once.
func TestUnloadDuringALoad(t *testing.T) {
	srv := bustest.Start(t)
	r := startRig(t, srv, engineConfig(t, srv, 5), nil, nil)
	r.sup.stopDelay = 1500 * time.Millisecond
	events := r.watch("operator", "devai.events.engines.fake")

	loadDone := make(chan [2]any, 1)
	go func() {
		code, rep := r.requestAsJob(fakeSubject+".load", loadBody(60), 31, 30*time.Second)
		loadDone <- [2]any{code, rep}
	}()
	r.waitStatus("fake", bus.StatusLoading, 3*time.Second)

	var wg sync.WaitGroup
	var first, second bus.Reply
	var firstEnd, secondEnd time.Time
	wg.Add(2)
	go func() {
		defer wg.Done()
		_, first = r.requestAsJob(fakeSubject+".unload", `{}`, 31, 20*time.Second)
		firstEnd = time.Now()
	}()
	time.Sleep(300 * time.Millisecond)
	go func() {
		defer wg.Done()
		_, second = r.requestAsJob(fakeSubject+".unload", `{}`, 31, 20*time.Second)
		secondEnd = time.Now()
	}()
	wg.Wait()
	if !first.OK || !second.OK {
		t.Fatalf("unloads: %+v / %+v", first, second)
	}
	if d := secondEnd.Sub(firstEnd); d < -300*time.Millisecond || d > 300*time.Millisecond {
		t.Errorf("the second unload returned %s apart from the first; it should join it", d)
	}
	got := <-loadDone
	if code := got[0].(int); code != 1 {
		t.Errorf("load exit %d, want 1", code)
	}
	if rep := got[1].(bus.Reply); rep.OK || rep.Error.Code != bus.CodeLoadFailed || rep.Error.Message != "unloaded" {
		t.Errorf("load reply %+v, want load_failed unloaded", rep)
	}
	if st := r.state("fake"); st.Status != bus.StatusIdle {
		t.Errorf("after: %+v", st)
	}
	start := time.Now()
	if code, rep := r.requestAsJob(fakeSubject+".unload", `{}`, 31, 5*time.Second); code != 0 || !rep.OK {
		t.Errorf("unload after: exit %d %+v", code, rep)
	}
	if time.Since(start) > time.Second {
		t.Errorf("the unload after took %s", time.Since(start))
	}
	time.Sleep(6 * time.Second) // past the load's delay: nothing comes back to life
	if st := r.state("fake"); st.Status != bus.StatusIdle {
		t.Errorf("the stopped load came back: %+v", st)
	}
	for _, s := range events.states(31) {
		if s == bus.StatusFailed {
			t.Errorf("job 31 went failed: %v", events.states(31))
		}
	}
}

// Phase 5 step 5, supervisor verbs: stop sent as job 0 without force unloads
// the job's model and its hold exits 0; restart on a serving engine is busy;
// start on an idle one replies with the state.
func TestSupervisorVerbsOnTheFakeEngine(t *testing.T) {
	srv := bustest.Start(t)
	r := startRig(t, srv, engineConfig(t, srv, 0), nil, nil)
	if code, _ := r.requestAsJob(fakeSubject+".load", loadBody(30), 41, 10*time.Second); code != 0 {
		t.Fatal("load")
	}
	h := r.startHold(fakeSubject, 41, nil)
	time.Sleep(500 * time.Millisecond)
	if rep := r.ask("operator", "fake", "restart", bus.Request{}, 5*time.Second); rep.OK || rep.Error.Code != bus.CodeBusy {
		t.Errorf("restart while serving: %+v", rep)
	}
	rep := r.ask("operator", "fake", "stop", bus.Request{Job: 0}, 10*time.Second)
	if !rep.OK || rep.State.Status != bus.StatusIdle || rep.State.Job != nil {
		t.Fatalf("stop: %+v", rep)
	}
	if c := h.wait(t, 3*time.Second); c != 0 {
		t.Errorf("hold exit %d, want 0", c)
	}
	// The epilog's unload is then a no-op.
	if code, rep := r.requestAsJob(fakeSubject+".unload", `{}`, 41, 5*time.Second); code != 0 || !rep.OK {
		t.Errorf("epilog unload: exit %d %+v", code, rep)
	}
	if rep := r.ask("operator", "fake", "start", bus.Request{}, 5*time.Second); !rep.OK || rep.State.Status != bus.StatusIdle {
		t.Errorf("start on idle: %+v", rep)
	}
}

func TestHoldExitsThreeWhenTheEngineFails(t *testing.T) {
	srv := bustest.Start(t)
	r := startRig(t, srv, engineConfig(t, srv, 0), nil, nil)
	if code, _ := r.requestAsJob(fakeSubject+".load", loadBody(30), 51, 10*time.Second); code != 0 {
		t.Fatal("load")
	}
	h := r.startHold(fakeSubject, 51, nil)
	time.Sleep(300 * time.Millisecond)
	r.sup.crash("fake")
	if c := h.wait(t, 5*time.Second); c != buscli.ExitLost {
		t.Errorf("hold exit %d, want 3", c)
	}
	if st := r.state("fake"); st.Status != bus.StatusFailed || st.Error == nil {
		t.Errorf("state %+v", st)
	}
}

// The permissions, with devai-control running: a jobs client cannot send a
// supervisor verb.
func TestJobsCannotRestartAnEngine(t *testing.T) {
	srv := bustest.Start(t)
	r := startRig(t, srv, engineConfig(t, srv, 0), nil, nil)
	if code, _ := r.requestAsJob(fakeSubject+".load", loadBody(30), 61, 10*time.Second); code != 0 {
		t.Fatal("load")
	}
	r.startHold(fakeSubject, 61, nil)
	code, _ := r.requestAsJob("devai.engines.ollama.restart", `{}`, 61, 2*time.Second)
	if code == 0 {
		t.Error("a jobs client's restart was answered")
	}
	code, _ = r.requestAsJob(fakeSubject+".stop", `{}`, 61, 2*time.Second)
	if code == 0 {
		t.Error("a jobs client's stop was answered")
	}
	if st := r.state("fake"); st.Status != bus.StatusServing {
		t.Errorf("state %+v", st)
	}
}

// Phase 5 step 5, the bus restarted on the same port: (i) a 1 s outage --
// the engine hold and an action hold renew within 2 s of the restart and the
// model stays loaded; (iii) an outage longer than the lease -- the engine
// hold exits 0 after the reconnect, because the model was unloaded.
func TestBusOutages(t *testing.T) {
	srv := bustest.Start(t)

	t.Run("one second", func(t *testing.T) {
		// A 3 s lease here, not the 2 s of the other tests: the holds
		// reconnect with nats.go's ReconnectWait of 1 s (C3), so after a 1 s
		// outage the first attempt usually finds the server still down and
		// the next one, 2 s after the disconnect, succeeds; a 2 s lease
		// would lapse in that gap whatever devai-control does (measured
		// here, 2026-10-07). Renewal within 2 s of the restart is checked as
		// the plan states it.
		ecfg := engineConfig(t, srv, 0)
		ecfg.LeaseS = 3
		e1 := startRig(t, srv, ecfg, nil, nil)
		ocfg := operatorConfig(t, srv)
		ocfg.LeaseS = 3
		op := startRig(t, srv, ocfg, nil, nil)
		t.Cleanup(op.stopAll)
		// One devai-control per container on the bus: these two end with
		// the subtest.
		defer e1.kill()
		defer op.kill()
		if code, _ := e1.requestAsJob(fakeSubject+".load", loadBody(30), 71, 10*time.Second); code != 0 {
			t.Fatal("load")
		}
		eh := e1.startHold(fakeSubject, 71, nil)
		if code, rep := op.requestAsJob(actionsSubject+".run", `{"kind":"action","action":"sleeper","params":{"sleep_s":30}}`, 72, 10*time.Second); code != 0 {
			t.Fatalf("run: %+v", rep)
		}
		ah := op.startHold(actionsSubject, 72, nil)
		leases := e1.watch("control-engines", fakeSubject+".lease")
		aleases := op.watch("control-operator", actionsSubject+".lease")
		time.Sleep(time.Second)

		srv.Shutdown()
		time.Sleep(time.Second)
		srv.Restart()
		restarted := time.Now()
		renewedAfter := func(w *watcher, job int64, n int) bool {
			for _, m := range w.all()[n:] {
				if r, err := bus.DecodeRequest(m.Data); err == nil && r.Job == job {
					return true
				}
			}
			return false
		}
		before, abefore := len(leases.all()), len(aleases.all())
		for time.Since(restarted) < 2*time.Second && !(renewedAfter(leases, 71, before) && renewedAfter(aleases, 72, abefore)) {
			time.Sleep(50 * time.Millisecond)
		}
		if !renewedAfter(leases, 71, before) {
			t.Errorf("the engine hold did not renew within 2 s of the restart")
		}
		if !renewedAfter(aleases, 72, abefore) {
			t.Errorf("the action hold did not renew within 2 s of the restart")
		}
		if !eh.running(500*time.Millisecond) || !ah.running(10*time.Millisecond) {
			t.Fatalf("a hold ended: %q / %q\nengines: %s\noperator: %s\nstate: %+v", eh.err.String(), ah.err.String(),
				e1.logs.String(), op.logs.String(), e1.state("fake"))
		}
		if st := e1.state("fake"); st.Status != bus.StatusServing || !st.JobIs(71) {
			t.Errorf("model after a 1 s outage: %+v", st)
		}
		if run := waitRun(t, op, 72, bus.RunRunning, time.Second); run.Status != bus.RunRunning {
			t.Errorf("action after a 1 s outage: %+v", run)
		}
		e1.requestAsJob(fakeSubject+".unload", `{}`, 71, 10*time.Second)
		eh.wait(t, 3*time.Second)
		op.requestAsJob(actionsSubject+".stop", `{"signal":"TERM","grace_s":1}`, 72, 5*time.Second)
		ah.wait(t, 5*time.Second)
	})

	eng := startRig(t, srv, engineConfig(t, srv, 0), nil, nil)

	t.Run("longer than the lease", func(t *testing.T) {
		if code, _ := eng.requestAsJob(fakeSubject+".load", loadBody(30), 73, 10*time.Second); code != 0 {
			t.Fatal("load")
		}
		eh := eng.startHold(fakeSubject, 73, nil)
		time.Sleep(500 * time.Millisecond)
		srv.Shutdown()
		time.Sleep(testLease + 1500*time.Millisecond)
		if eng.mem.get() != 2 {
			t.Errorf("the lease did not unload the model during the outage: card %d MiB", eng.mem.get())
		}
		srv.Restart()
		if c := eh.wait(t, 5*time.Second); c != 0 {
			t.Errorf("engine hold exit %d after the reconnect, want 0", c)
		}
	})

	t.Run("no bus before the first state", func(t *testing.T) {
		srv.Shutdown()
		defer srv.Restart()
		h := eng.startHold(fakeSubject, 74, func(o *buscli.HoldOptions) { o.ConnectTimeout = 2 * time.Second })
		if c := h.wait(t, 6*time.Second); c != buscli.ExitUnreachable {
			t.Errorf("hold exit %d, want 2", c)
		}
	})
}

var _ = nats.ErrNoResponders

// Review finding: an unload that lands while the load is still reading the
// card and starting its program must not leave that program running with
// the state idle.
func TestUnloadBeforeTheProgramStarts(t *testing.T) {
	srv := bustest.Start(t)
	mem := &fakeMemory{used: 2}
	r := startRig(t, srv, engineConfig(t, srv, 0.5), nil, mem)
	mem.mu.Lock()
	mem.delay = 600 * time.Millisecond
	mem.mu.Unlock()
	loadDone := make(chan int, 1)
	go func() {
		code, _ := r.requestAsJob(fakeSubject+".load", loadBody(30), 81, 20*time.Second)
		loadDone <- code
	}()
	time.Sleep(150 * time.Millisecond) // the load is reading the card
	code, rep := r.requestAsJob(fakeSubject+".unload", `{}`, 81, 20*time.Second)
	if code != 0 || !rep.OK {
		t.Fatalf("unload: exit %d %+v", code, rep)
	}
	if c := <-loadDone; c != 1 {
		t.Errorf("load exit %d, want 1 (unloaded)", c)
	}
	time.Sleep(1500 * time.Millisecond) // past the program's ready delay
	if st := r.sup.state("fake"); st == "RUNNING" {
		t.Errorf("the engine program is RUNNING after the unload (orphan)")
	}
	if st := r.state("fake"); st.Status != bus.StatusIdle {
		t.Errorf("state %+v", st)
	}
	if mem.get() != 2 {
		t.Errorf("card = %d MiB, want the baseline", mem.get())
	}
}

// C1: a job that does not own the model gets not_owner, also while the
// owner's unload is under way.
func TestNonOwnerUnloadDuringAnUnloadIsRefused(t *testing.T) {
	srv := bustest.Start(t)
	r := startRig(t, srv, engineConfig(t, srv, 0), nil, nil)
	r.sup.stopDelay = time.Second
	if code, _ := r.requestAsJob(fakeSubject+".load", loadBody(30), 91, 10*time.Second); code != 0 {
		t.Fatal("load")
	}
	go r.requestAsJob(fakeSubject+".unload", `{}`, 91, 10*time.Second)
	r.waitStatus("fake", bus.StatusUnloading, 2*time.Second)
	code, rep := r.requestAsJob(fakeSubject+".unload", `{}`, 92, 10*time.Second)
	if code != 1 || rep.Error.Code != bus.CodeNotOwner {
		t.Errorf("another job's unload during the owner's: exit %d %+v", code, rep)
	}
}
