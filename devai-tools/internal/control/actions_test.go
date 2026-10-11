package control

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/sparavec/devai-tools/internal/buscli"
	"github.com/sparavec/gpu-arbiter/pkg/bus"
	"github.com/sparavec/gpu-arbiter/pkg/bus/bustest"
)

const actionsSubject = "devai.operator.actions"

const testRegistry = `version: 1
actions:
  - id: sleeper
    group: test
    title: Sleep, then exit
    kind: job
    script: bin/sleeper.sh
    params:
      - {name: sleep_s, type: int, required: true, to: {env: SLEEP_S}}
      - {name: stop_delay, type: int, to: {env: STOP_DELAY}}
      - {name: exit_code, type: int, to: {env: EXIT_CODE}}
      - {name: label, type: string, pattern: '^[a-z]{1,8}$', to: {flag: --label}}
      - {name: verbose, type: bool, to: {flag: --verbose}}
      - {name: mode, type: enum, values: [fast, slow], to: {env: MODE}}
    job_kind: action
  - id: stubborn
    group: test
    title: Ignore TERM
    kind: job
    script: bin/stubborn.sh
    job_kind: action
  - id: bencher
    group: bench
    title: A bench task
    kind: job
    script: bin/bencher.sh
    params:
      - {name: write_result, type: bool, to: {env: WRITE_RESULT}}
    job_kind: bench
  - id: shown-only
    group: host
    title: A host action
    kind: host
    script: bin/sleeper.sh
`

var testScripts = map[string]string{
	"sleeper.sh": `#!/bin/bash
trap 'echo "got TERM"; sleep "${STOP_DELAY:-0}"; exit 143' TERM
trap 'echo "got INT"; exit 130' INT
echo "started job=$DEVAI_JOB_ID args=$* mode=$MODE results=$DEVAI_RESULTS_DIR repo=$DEVAI_REPO"
sleep "${SLEEP_S:-30}" & wait $!
echo done
exit "${EXIT_CODE:-0}"
`,
	"stubborn.sh": `#!/bin/bash
trap '' TERM INT
sleep 60 &
echo $! > "$DEVAI_RESULTS_DIR/child.pid"
wait
`,
	"bencher.sh": `#!/bin/bash
if [ "$WRITE_RESULT" = true ]; then
  echo '{"v":1,"job":'"$DEVAI_JOB_ID"',"kind":"bench","status":"ok","summary":"its own"}' > "$DEVAI_RESULTS_DIR/result.json"
fi
exit 0
`,
}

func operatorConfig(t *testing.T, srv *bustest.Server) *Config {
	dir := t.TempDir()
	repo := filepath.Join(dir, "repo")
	if err := os.MkdirAll(filepath.Join(repo, "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	for name, body := range testScripts {
		if err := os.WriteFile(filepath.Join(repo, "bin", name), []byte(body), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	reg := filepath.Join(repo, "actions.yaml")
	if err := os.WriteFile(reg, []byte(testRegistry), 0o644); err != nil {
		t.Fatal(err)
	}
	return &Config{
		Container: "operator", Bus: srv.Config("control-operator"),
		StateFile: filepath.Join(dir, "control.json"),
		LeaseS:    testLease.Seconds(), HeartbeatS: testHeartbeat.Seconds(), PollS: 0.2, RunKeepS: 600,
		GPU: "none", ResultsDir: filepath.Join(dir, "results"), MaxContext: 262144,
		Services: map[string]ServiceConfig{
			"actions": {Kind: KindActions, Registry: reg, Repo: repo, OpsEnv: []string{}},
		},
	}
}

// stopAll kills every run's process group (test cleanup).
func (r *rig) stopAll() {
	a, ok := r.ctrl.services["actions"].(*actions)
	if !ok {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, x := range a.runs {
		killGroup(x.PID)
	}
}

func (r *rig) run(job int64, kind, action, params string) (int, bus.Reply) {
	r.t.Helper()
	body := `{"kind":"` + kind + `","action":"` + action + `","params":` + params + `}`
	return r.requestAsJob(actionsSubject+".run", body, job, 15*time.Second)
}

func (r *rig) results(job int64, name string) string {
	b, _ := os.ReadFile(filepath.Join(r.cfg.ResultsDir, strconv.FormatInt(job, 10), name))
	return string(b)
}

func waitRun(t *testing.T, r *rig, job int64, status string, within time.Duration) bus.Run {
	t.Helper()
	deadline := time.Now().Add(within)
	for {
		rep := r.ask("operator", "actions", bus.VerbState, bus.Request{Job: job}, 5*time.Second)
		if rep.State.Run != nil && rep.State.Run.Status == status {
			return *rep.State.Run
		}
		if time.Now().After(deadline) {
			t.Fatalf("job %d's run: %+v, want %s within %s\n%s", job, rep.State.Run, status, within, r.logs.String())
		}
		time.Sleep(50 * time.Millisecond)
	}
}

func TestRunRepliesOnStartAndExitedIsAnEvent(t *testing.T) {
	srv := bustest.Start(t)
	r := startRig(t, srv, operatorConfig(t, srv), nil, nil)
	t.Cleanup(r.stopAll)
	events := r.watch("operator", "devai.events.operator.actions")
	start := time.Now()
	code, rep := r.run(11, "action", "sleeper", `{"sleep_s":1,"label":"abc","verbose":true,"mode":"slow"}`)
	if code != 0 || rep.State.Run == nil || rep.State.Run.Status != bus.RunRunning {
		t.Fatalf("run: exit %d %+v", code, rep)
	}
	if time.Since(start) > 900*time.Millisecond {
		t.Errorf("run replied after %s; it replies on start", time.Since(start))
	}
	h := r.startHold(actionsSubject, 11, nil)
	if c := h.wait(t, 5*time.Second); c != 0 {
		t.Errorf("hold exit %d, want 0", c)
	}
	var sawExited bool
	for _, m := range events.all() {
		s, _ := bus.DecodeState(m.Data)
		if run, ok := s.FindRun(11); ok && run.Status == bus.RunExited && run.ExitCode != nil && *run.ExitCode == 0 {
			sawExited = true
		}
	}
	if !sawExited {
		t.Error("no `exited` event for job 11")
	}
	log := r.results(11, "action.log")
	for _, want := range []string{"started job=11", "args=--label abc --verbose", "mode=slow", "done"} {
		if !strings.Contains(log, want) {
			t.Errorf("action.log lacks %q:\n%s", want, log)
		}
	}
	var res Result
	if err := json.Unmarshal([]byte(r.results(11, "result.json")), &res); err != nil {
		t.Fatalf("result.json: %v", err)
	}
	if res.Status != "ok" || res.Kind != "action" || res.Outputs["exit_code"] != float64(0) {
		t.Errorf("result.json = %+v", res)
	}
}

func TestRunRefusedWhenItBreaksTheRegistry(t *testing.T) {
	srv := bustest.Start(t)
	r := startRig(t, srv, operatorConfig(t, srv), nil, nil)
	t.Cleanup(r.stopAll)
	cases := map[string][3]string{
		"unknown parameter":  {"action", "sleeper", `{"sleep_s":1,"nope":1}`},
		"wrong type":         {"action", "sleeper", `{"sleep_s":"one"}`},
		"pattern":            {"action", "sleeper", `{"sleep_s":1,"label":"ABC; rm -rf /"}`},
		"missing required":   {"action", "sleeper", `{}`},
		"enum":               {"action", "sleeper", `{"sleep_s":1,"mode":"medium"}`},
		"another job kind":   {"bench", "sleeper", `{"sleep_s":1}`},
		"unknown action":     {"action", "nothing", `{}`},
		"not a job action":   {"action", "shown-only", `{}`},
		"params not object":  {"action", "sleeper", `[1]`},
		"bool for an int":    {"action", "sleeper", `{"sleep_s":true}`},
		"fraction for int":   {"action", "sleeper", `{"sleep_s":1.5}`},
		"string for a bool":  {"action", "sleeper", `{"sleep_s":1,"verbose":"yes"}`},
		"a kind not allowed": {"probe", "bencher", `{}`},
	}
	job := int64(100)
	for name, c := range cases {
		job++
		code, rep := r.run(job, c[0], c[1], c[2])
		if code != 1 || rep.OK || rep.Error.Code != bus.CodeBadRequest {
			t.Errorf("%s: exit %d %+v, want bad_request", name, code, rep)
		}
		if out := r.results(job, "action.log"); out != "" {
			t.Errorf("%s: the action ran: %s", name, out)
		}
	}
}

// A run whose process group is signalled and then killed after its grace.
func TestStopSignalsTheGroupThenKills(t *testing.T) {
	srv := bustest.Start(t)
	r := startRig(t, srv, operatorConfig(t, srv), nil, nil)
	t.Cleanup(r.stopAll)
	if code, _ := r.run(12, "action", "stubborn", `{}`); code != 0 {
		t.Fatal("run")
	}
	time.Sleep(300 * time.Millisecond)
	child, _ := strconv.Atoi(strings.TrimSpace(r.results(12, "child.pid")))
	if child <= 0 {
		t.Fatal("no child pid")
	}
	start := time.Now()
	rep := r.ask("jobs", "actions", bus.VerbStop, bus.Request{Job: 12, Signal: "TERM", GraceS: intp(1)}, 5*time.Second)
	if !rep.OK {
		t.Fatalf("stop: %+v", rep)
	}
	if time.Since(start) > 500*time.Millisecond {
		t.Errorf("stop replied after %s; it replies at once", time.Since(start))
	}
	run := waitRun(t, r, 12, bus.RunExited, 4*time.Second)
	if d := time.Since(start); d < 900*time.Millisecond {
		t.Errorf("killed after %s, before its 1 s grace", d)
	}
	if run.ExitCode == nil || *run.ExitCode != 128+int(syscall.SIGKILL) {
		t.Errorf("exit code %v, want 137", run.ExitCode)
	}
	time.Sleep(200 * time.Millisecond)
	if err := syscall.Kill(child, 0); err == nil {
		t.Errorf("the action's child %d outlived the group kill", child)
	}
	var res Result
	_ = json.Unmarshal([]byte(r.results(12, "result.json")), &res)
	if res.Status != "cancelled" {
		t.Errorf("result.json status %q, want cancelled", res.Status)
	}
}

func intp(n int) *int        { return &n }
func i64p(n int64) *int64    { return &n }
func boolText(b bool) string { return strconv.FormatBool(b) }

func TestRunStoppedOnceItsLeaseLapses(t *testing.T) {
	srv := bustest.Start(t)
	r := startRig(t, srv, operatorConfig(t, srv), nil, nil)
	t.Cleanup(r.stopAll)
	start := time.Now()
	if code, _ := r.run(13, "action", "sleeper", `{"sleep_s":30}`); code != 0 {
		t.Fatal("run")
	}
	run := waitRun(t, r, 13, bus.RunExited, testLease+3*time.Second)
	if d := time.Since(start); d < testLease {
		t.Errorf("stopped after %s, before the lease lapsed", d)
	}
	if run.ExitCode == nil || *run.ExitCode != 143 {
		t.Errorf("exit %v, want 143 (TERM)", run.ExitCode)
	}
}

func TestRunRefusedAfterAStopForItsJob(t *testing.T) {
	srv := bustest.Start(t)
	r := startRig(t, srv, operatorConfig(t, srv), nil, nil)
	t.Cleanup(r.stopAll)
	if code, rep := r.requestAsJob(actionsSubject+".stop", `{"signal":"TERM","grace_s":20}`, 14, 5*time.Second); code != 0 {
		t.Fatalf("stop with no run: exit %d %+v", code, rep)
	}
	code, rep := r.run(14, "action", "sleeper", `{"sleep_s":1}`)
	if code != 1 || rep.Error.Code != bus.CodeBadRequest {
		t.Errorf("run after its stop: exit %d %+v", code, rep)
	}
	// The hold of a job with no run exits 0 at once.
	h := r.startHold(actionsSubject, 14, nil)
	if c := h.wait(t, 3*time.Second); c != 0 {
		t.Errorf("hold exit %d", c)
	}
}

func TestForcedStopWithRunJob(t *testing.T) {
	srv := bustest.Start(t)
	r := startRig(t, srv, operatorConfig(t, srv), nil, nil)
	t.Cleanup(r.stopAll)
	if code, _ := r.run(15, "action", "sleeper", `{"sleep_s":30}`); code != 0 {
		t.Fatal("run")
	}
	h := r.startHold(actionsSubject, 15, nil)
	rep := r.ask("jobs", "actions", bus.VerbStop, bus.Request{Job: 16, RunJob: i64p(15), Signal: "TERM", GraceS: intp(5)}, 5*time.Second)
	if rep.OK || rep.Error.Code != bus.CodeNotOwner {
		t.Errorf("stop of another job's run without force: %+v", rep)
	}
	rep = r.ask("jobs", "actions", bus.VerbStop, bus.Request{Job: 16, RunJob: i64p(15), Force: true, Signal: "TERM", GraceS: intp(5)}, 5*time.Second)
	if !rep.OK {
		t.Fatalf("forced stop: %+v", rep)
	}
	waitRun(t, r, 15, bus.RunExited, 4*time.Second)
	if c := h.wait(t, 3*time.Second); c != 1 {
		t.Errorf("hold exit %d, want 1 (the action exited non-zero)", c)
	}
}

// Phase 5 step 5: an action hold whose devai-control is replaced by a fresh
// one with an empty state file exits 3 within one heartbeat.
func TestActionHoldExitsThreeWhenItsRunIsForgotten(t *testing.T) {
	srv := bustest.Start(t)
	cfg := operatorConfig(t, srv)
	r := startRig(t, srv, cfg, nil, nil)
	t.Cleanup(r.stopAll)
	if code, _ := r.run(17, "action", "sleeper", `{"sleep_s":30}`); code != 0 {
		t.Fatal("run")
	}
	h := r.startHold(actionsSubject, 17, nil)
	time.Sleep(500 * time.Millisecond)
	a := r.ctrl.services["actions"].(*actions)
	a.mu.Lock()
	pid := a.runs[17].PID
	a.mu.Unlock()
	t.Cleanup(func() { killGroup(pid) })
	r.kill()
	fresh := *cfg
	fresh.StateFile = filepath.Join(t.TempDir(), "control.json")
	started := time.Now()
	startRig(t, srv, &fresh, nil, nil)
	if c := h.wait(t, testHeartbeat+3*time.Second); c != buscli.ExitLost {
		t.Errorf("hold exit %d, want 3", c)
	}
	if d := time.Since(started); d > testHeartbeat+1500*time.Millisecond {
		t.Errorf("the hold took %s, more than one heartbeat", d)
	}
}

func TestRestartedControlPublishesLostForARunItCannotAdopt(t *testing.T) {
	srv := bustest.Start(t)
	cfg := operatorConfig(t, srv)
	r := startRig(t, srv, cfg, nil, nil)
	t.Cleanup(r.stopAll)
	if code, _ := r.run(18, "action", "sleeper", `{"sleep_s":30}`); code != 0 {
		t.Fatal("run")
	}
	a := r.ctrl.services["actions"].(*actions)
	a.mu.Lock()
	pid := a.runs[18].PID
	a.mu.Unlock()
	r.kill()
	killGroup(pid) // the run died while devai-control was down
	time.Sleep(300 * time.Millisecond)
	events := r.watch("operator", "devai.events.operator.actions")
	r2 := startRig(t, srv, cfg, nil, nil)
	run := waitRun(t, r2, 18, bus.RunLost, 3*time.Second)
	if run.ExitCode != nil {
		t.Errorf("a lost run has exit code %v", *run.ExitCode)
	}
	found := false
	for _, m := range events.all() {
		s, _ := bus.DecodeState(m.Data)
		if run, ok := s.FindRun(18); ok && run.Status == bus.RunLost {
			found = true
		}
	}
	if !found {
		t.Error("no `lost` event for job 18")
	}
	h := r2.startHold(actionsSubject, 18, nil)
	if c := h.wait(t, 3*time.Second); c != buscli.ExitLost {
		t.Errorf("hold exit %d, want 3", c)
	}
}

// Phase 5 step 5, result.json (C4).
func TestResultJSONWriterRule(t *testing.T) {
	srv := bustest.Start(t)
	r := startRig(t, srv, operatorConfig(t, srv), nil, nil)
	t.Cleanup(r.stopAll)

	if code, _ := r.run(19, "bench", "bencher", `{"write_result":true}`); code != 0 {
		t.Fatal("run")
	}
	waitRun(t, r, 19, bus.RunExited, 5*time.Second)
	if got := r.results(19, "result.json"); !strings.Contains(got, `"its own"`) {
		t.Errorf("a bench run's own result.json was replaced: %s", got)
	}

	if code, _ := r.run(20, "bench", "bencher", `{"write_result":false}`); code != 0 {
		t.Fatal("run")
	}
	waitRun(t, r, 20, bus.RunExited, 5*time.Second)
	var res Result
	if err := json.Unmarshal([]byte(r.results(20, "result.json")), &res); err != nil {
		t.Fatalf("no fallback result.json: %v", err)
	}
	if res.Status != "failed" || res.Kind != "bench" || res.Outputs["exit_code"] != float64(0) {
		t.Errorf("fallback = %+v", res)
	}
}

// Phase 5 step 5 (ii): the action exits during a bus outage; its hold exits
// with the action's code after the reconnect.
func TestActionHoldSeesAnExitThatHappenedDuringAnOutage(t *testing.T) {
	srv := bustest.Start(t)
	r := startRig(t, srv, operatorConfig(t, srv), nil, nil)
	t.Cleanup(r.stopAll)
	if code, _ := r.run(22, "action", "sleeper", `{"sleep_s":1,"exit_code":4}`); code != 0 {
		t.Fatal("run")
	}
	h := r.startHold(actionsSubject, 22, nil)
	time.Sleep(200 * time.Millisecond)
	srv.Shutdown()
	time.Sleep(2 * time.Second) // the action exits at 1 s, during the outage
	if !h.running(10 * time.Millisecond) {
		t.Fatalf("the hold ended during the outage: %s", h.err.String())
	}
	srv.Restart()
	if c := h.wait(t, 5*time.Second); c != 1 {
		t.Errorf("hold exit %d, want 1 (the action exited 4)", c)
	}
}

func TestRegistryRules(t *testing.T) {
	base := "version: 1\nactions:\n"
	entry := func(extra string) string {
		return base + "  - id: x\n    group: test\n    title: t\n    kind: job\n    script: s.sh\n" + extra
	}
	cases := map[string]string{
		"job without job_kind":        entry(""),
		"bad job_kind":                entry("    job_kind: other\n"),
		"gpu on a bench":              entry("    job_kind: bench\n    gpu: true\n    time_limit_min: 5\n"),
		"gpu without time limit":      entry("    job_kind: action\n    gpu: true\n"),
		"comment on a probe":          entry("    job_kind: probe\n    comment: {backend: vllm, model: m}\n    params:\n      - {name: m, type: string, pattern: '.*', to: {flag: --m}}\n"),
		"comment names an undeclared": entry("    job_kind: action\n    comment: {backend: vllm, model: nope}\n"),
		"comment without model":       entry("    job_kind: action\n    comment: {backend: vllm}\n"),
		"absolute script":             base + "  - id: x\n    group: test\n    title: t\n    kind: job\n    script: /bin/sh\n    job_kind: action\n",
		"unknown key":                 entry("    job_kind: action\n    shell: true\n"),
		"string without pattern":      entry("    job_kind: action\n    params:\n      - {name: a, type: string, to: {flag: --a}}\n"),
		"param both flag and env":     entry("    job_kind: action\n    params:\n      - {name: a, type: int, to: {flag: --a, env: A}}\n"),
	}
	for name, y := range cases {
		if _, err := ParseRegistry([]byte(y)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	ok := entry("    job_kind: action\n    gpu: true\n    time_limit_min: 30\n    comment: {backend: vllm, model: m, ctx: c}\n    params:\n      - {name: m, type: string, pattern: '^[a-z]+$', required: true, to: {flag: --model}}\n      - {name: c, type: int, to: {env: CTX}}\n")
	if _, err := ParseRegistry([]byte(ok)); err != nil {
		t.Errorf("a valid entry was refused: %v", err)
	}
	_ = boolText
}

// Review finding: a stop that arrives while the run is still being prepared
// (ops-env, the hold key) refuses the run and signals nothing -- above all
// not devai-control's own process group.
func TestStopWhileTheRunIsPrepared(t *testing.T) {
	srv := bustest.Start(t)
	cfg := operatorConfig(t, srv)
	sc := cfg.Services["actions"]
	sc.OpsEnv = []string{"sh", "-c", "sleep 1"}
	cfg.Services["actions"] = sc
	r := startRig(t, srv, cfg, nil, nil)
	t.Cleanup(r.stopAll)
	done := make(chan bus.Reply, 1)
	go func() {
		_, rep := r.run(23, "action", "sleeper", `{"sleep_s":30}`)
		done <- rep
	}()
	time.Sleep(300 * time.Millisecond)
	rep := r.ask("jobs", "actions", bus.VerbStop, bus.Request{Job: 23, Signal: "TERM", GraceS: intp(1)}, 5*time.Second)
	if !rep.OK {
		t.Fatalf("stop: %+v", rep)
	}
	run := <-done
	if run.OK || run.Error.Code != bus.CodeBadRequest {
		t.Errorf("the run after its stop: %+v", run)
	}
	time.Sleep(1500 * time.Millisecond) // past the stop's grace
	if st := r.state("actions"); st.Status != bus.StatusIdle {
		t.Errorf("state %+v", st)
	}
	if out := r.results(23, "action.log"); strings.Contains(out, "started") {
		t.Errorf("the action ran: %s", out)
	}
	// devai-control is still alive and answering.
	if rep := r.ask("operator", "actions", bus.VerbState, bus.Request{Job: 23}, 3*time.Second); !rep.OK {
		t.Errorf("devai-control no longer answers: %+v", rep)
	}
}
