package control

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/sparavec/devai-tools/internal/jobs"
	"github.com/sparavec/gpu-arbiter/pkg/bus"
	"github.com/sparavec/gpu-arbiter/pkg/slurmrest"
)

// Stop defaults (C1, C3).
const (
	DefaultStopGraceS = 30
	MaxStopGraceS     = 600
	leaseStopGrace    = 30 * time.Second
)

// copiedEnv is copied from devai-control's own environment into every run
// when set, after ops-env's lines, which it wins over (C8).
var copiedEnv = []string{
	"CONTAINER_HOST", "DEVAI_GPU_DEVICE", "HF_TOKEN_PATH",
	"DEVAI_BUS_URL", "DEVAI_BUS_USER", "DEVAI_BUS_PASSWORD_FILE",
	"SLURM_REST_URL", "SLURM_JWT_KEY_FILE", "SLURM_USER",
	"DEVAI_HOST_IP", "DEVAI_HOME_VOLUME", "DEVAI_HOST_WORK_DIR", "LAB_PORT", "DSH_PORT",
	"DEVAI_HOST_GITCONFIG", "DEVAI_HOST_SSH_DIR",
}

var opsEnvLineRE = regexp.MustCompile(`^([A-Z_][A-Z0-9_]*)=(.*)$`)

// run is one action run, owned by the job that started it.
type run struct {
	Job       int64  `json:"job"`
	Action    string `json:"action"`
	Kind      string `json:"kind"`
	PID       int    `json:"pid"`
	StartTick uint64 `json:"start_tick"`
	Status    string `json:"status"`
	ExitCode  *int   `json:"exit_code,omitempty"`
	Since     string `json:"since"`
	Ended     string `json:"ended,omitempty"`
	Stopped   bool   `json:"stopped"`

	startedAt time.Time
	endedAt   time.Time
	leaseAt   time.Time
	killTimer *time.Timer
	// adopted: a restarted devai-control took the run over; it is not its
	// child, so its exit status cannot be collected.
	adopted bool
}

func (r *run) bus() bus.Run {
	return bus.Run{Job: r.Job, Status: r.Status, Action: r.Action, Kind: r.Kind, ExitCode: r.ExitCode,
		Since: r.Since, Ended: r.Ended}
}

// actions is the `actions` kind: devai-operator's service that runs
// registered actions for jobs (C1, C8).
type actions struct {
	c        *Controller
	name     string
	cfg      ServiceConfig
	registry *Registry

	mu      sync.Mutex
	runs    map[int64]*run
	stopped map[int64]time.Time // a stop that came before any run of its job
}

func newActions(c *Controller, name string, sc ServiceConfig) (*actions, error) {
	reg, err := LoadRegistry(sc.Registry)
	if err != nil {
		return nil, err
	}
	return &actions{c: c, name: name, cfg: sc, registry: reg, runs: map[int64]*run{}, stopped: map[int64]time.Time{}}, nil
}

func (a *actions) runsLocked() []bus.Run {
	out := make([]bus.Run, 0, len(a.runs))
	for _, r := range a.runs {
		out = append(out, r.bus())
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Job < out[j].Job })
	return out
}

func (a *actions) serviceStatusLocked() string {
	for _, r := range a.runs {
		if r.Status == bus.RunRunning {
			return bus.RunRunning
		}
	}
	return bus.StatusIdle
}

// event is the heartbeat: every run the service still keeps (C1).
func (a *actions) event() bus.State {
	a.mu.Lock()
	defer a.mu.Unlock()
	return bus.State{Container: a.c.cfg.Container, Service: a.name, Status: a.serviceStatusLocked(),
		Since: bus.Timestamp(a.c.now()), Runs: a.runsLocked()}
}

// runStateLocked is `state` for one job: its run (null when none).
func (a *actions) runStateLocked(job int64) bus.State {
	j := job
	st := bus.State{Container: a.c.cfg.Container, Service: a.name, Status: bus.StatusIdle, Job: &j,
		Since: bus.Timestamp(a.c.now()), Runs: a.runsLocked()}
	if r, ok := a.runs[job]; ok {
		br := r.bus()
		st.Run = &br
		st.Status = r.Status
		st.Since = r.Since
	}
	return st
}

func (a *actions) runChanged(job int64) {
	a.mu.Lock()
	st := a.runStateLocked(job)
	a.mu.Unlock()
	a.c.save()
	a.c.publishState(a.name, st)
}

func (a *actions) handle(ctx context.Context, verb string, req bus.Request) (bus.Reply, bool) {
	switch verb {
	case bus.VerbState:
		a.mu.Lock()
		st := a.runStateLocked(req.Job)
		a.mu.Unlock()
		return bus.OKReply(st), true
	case bus.VerbRun:
		return a.doRun(ctx, req), true
	case bus.VerbStop:
		return a.doStop(req), true
	case bus.VerbLease:
		a.mu.Lock()
		if r, ok := a.runs[req.Job]; ok && r.Status == bus.RunRunning {
			if now := a.c.now(); now.After(r.leaseAt) {
				r.leaseAt = now
			}
		}
		a.mu.Unlock()
		return bus.Reply{}, false
	case bus.VerbStart, bus.VerbRestart, bus.VerbReload:
		return bus.ErrReply(bus.CodeBadRequest, fmt.Sprintf("%s does not apply to the actions service (C1)", verb), ""), true
	}
	return unknownVerb("actions", verb), true
}

func (a *actions) resultsDir(job int64) string {
	return filepath.Join(a.c.cfg.ResultsDir, strconv.FormatInt(job, 10))
}

// opsEnv runs `make -s -C $DEVAI_REPO ops-env` (or the configured command)
// and reads its NAME=value lines; a non-zero exit refuses the action (C8).
func (a *actions) opsEnv(ctx context.Context) (map[string]string, error) {
	argv := a.cfg.OpsEnv
	if argv == nil {
		argv = []string{"make", "-s", "-C", "{repo}", "ops-env"}
	}
	out := map[string]string{}
	if len(argv) == 0 {
		return out, nil
	}
	expanded := make([]string, len(argv))
	for i, x := range argv {
		expanded[i] = strings.ReplaceAll(x, "{repo}", a.cfg.Repo)
	}
	cctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(cctx, expanded[0], expanded[1:]...)
	cmd.Dir = a.cfg.Repo
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	b, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("ops-env: %v: %s", err, strings.TrimSpace(stderr.String()))
	}
	sc := bufio.NewScanner(bytes.NewReader(b))
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		m := opsEnvLineRE.FindStringSubmatch(line)
		if m == nil {
			return nil, fmt.Errorf("ops-env: unreadable line %q", line)
		}
		out[m[1]] = m[2]
	}
	return out, nil
}

// holdKey reads the job's hold_key from its Slurm comment (C3, C8), the
// way `devai-jobs show <jobid> --json` does, through the one REST client.
// Without SLURM_REST_URL there is no Slurm to ask, and the key stays empty.
func holdKey(ctx context.Context, job int64) (string, error) {
	if os.Getenv(slurmrest.EnvURL) == "" {
		return "", nil
	}
	c, err := slurmrest.FromEnv()
	if err != nil {
		return "", err
	}
	cctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	var comment string
	j, err := c.Job(cctx, job)
	switch {
	case err == nil:
		comment = j.Comment
	case slurmrest.IsInvalidJob(err):
		d, derr := c.DBJob(cctx, job)
		if derr != nil {
			return "", derr
		}
		comment = d.Comment.Job
	default:
		return "", err
	}
	k, _ := jobs.ParseComment(comment)["hold_key"].(string)
	return k, nil
}

// buildEnv is C8's base environment, then the parameters' variables.
func (a *actions) buildEnv(ops map[string]string, job int64, key string, params map[string]string) []string {
	env := map[string]string{}
	for _, k := range []string{"PATH", "HOME"} {
		if v, ok := os.LookupEnv(k); ok {
			env[k] = v
		}
	}
	env["DEVAI_REPO"] = a.cfg.Repo
	for k, v := range ops {
		env[k] = v
	}
	for _, k := range copiedEnv {
		if v, ok := os.LookupEnv(k); ok {
			env[k] = v
		}
	}
	env["DEVAI_JOB_ID"] = strconv.FormatInt(job, 10)
	env["DEVAI_RESULTS_DIR"] = a.resultsDir(job)
	env["DEVAI_HOLD_KEY"] = key
	for k, v := range params {
		env[k] = v
	}
	keys := make([]string, 0, len(env))
	for k := range env {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]string, 0, len(keys))
	for _, k := range keys {
		out = append(out, k+"="+env[k])
	}
	return out
}

// doRun starts a registered action for the job and replies once it has
// started (C1). Refused: an unknown action, one that is not a job action,
// a job_kind other than the sending job's kind, parameters that break the
// registry, a job that already has a run, and a job whose stop came first.
func (a *actions) doRun(ctx context.Context, req bus.Request) bus.Reply {
	bad := func(format string, args ...any) bus.Reply {
		return bus.ErrReply(bus.CodeBadRequest, fmt.Sprintf(format, args...), "")
	}
	if req.Job <= 0 {
		return bad("a run belongs to a job: job must be its Slurm id")
	}
	act, ok := a.registry.Lookup(req.Action)
	if !ok {
		return bad("no action %q in the registry", req.Action)
	}
	if act.Kind != "job" {
		return bad("action %q is a %s action, not a job action", act.ID, act.Kind)
	}
	if act.JobKind != req.Kind {
		return bad("action %q runs as a %s job, not as a %q job", act.ID, act.JobKind, req.Kind)
	}
	mapped, err := act.MapParams(req.Params)
	if err != nil {
		return bad("action %q: %v", act.ID, err)
	}
	a.mu.Lock()
	if _, ok := a.stopped[req.Job]; ok {
		a.mu.Unlock()
		return bad("a stop for job %d came before its run", req.Job)
	}
	if _, ok := a.runs[req.Job]; ok {
		a.mu.Unlock()
		return bad("job %d already has a run", req.Job)
	}
	// Reserve the job while the run is prepared.
	r := &run{Job: req.Job, Action: act.ID, Kind: act.JobKind, Status: bus.RunRunning}
	a.runs[req.Job] = r
	a.mu.Unlock()
	release := func() {
		a.mu.Lock()
		delete(a.runs, req.Job)
		a.mu.Unlock()
	}

	ops, err := a.opsEnv(ctx)
	if err != nil {
		release()
		return bad("%v", err)
	}
	key, err := holdKey(ctx, req.Job)
	if err != nil {
		release()
		return bad("cannot read job %d's hold key: %v", req.Job, err)
	}
	dir := a.resultsDir(req.Job)
	if err := ensureResultsDir(dir); err != nil {
		release()
		return bad("results directory: %v", err)
	}
	logf, err := os.OpenFile(filepath.Join(dir, "action.log"), os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o664)
	if err != nil {
		release()
		return bad("action.log: %v", err)
	}
	argv := append([]string{filepath.Join(a.cfg.Repo, act.Script)}, act.Args...)
	argv = append(argv, mapped.Args...)
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Dir = a.cfg.Repo
	cmd.Env = a.buildEnv(ops, req.Job, key, mapped.Env)
	cmd.Stdout, cmd.Stderr = logf, logf
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	// A stop that arrived while the run was prepared (the job's TERM came
	// during its `run` request) refuses it. The lock is held across Start,
	// so no stop can fall between this check and the pid being known.
	a.mu.Lock()
	if _, stopped := a.stopped[req.Job]; stopped || r.Stopped {
		delete(a.runs, req.Job)
		a.stopped[req.Job] = a.c.now()
		a.mu.Unlock()
		logf.Close()
		return bad("a stop for job %d came before its run started", req.Job)
	}
	if err := cmd.Start(); err != nil {
		delete(a.runs, req.Job)
		a.mu.Unlock()
		logf.Close()
		return bad("start %s: %v", act.Script, err)
	}
	now := a.c.now()
	r.PID = cmd.Process.Pid
	r.StartTick = procStartTick(r.PID)
	r.startedAt, r.leaseAt, r.Since = now, now, bus.Timestamp(now)
	st := a.runStateLocked(req.Job)
	a.mu.Unlock()
	a.c.log.Printf("actions: job %d: started %s (pid %d)", req.Job, act.ID, r.PID)
	a.c.save()
	a.c.publishState(a.name, st)
	go func() {
		err := cmd.Wait()
		logf.Close()
		code := exitCode(cmd.ProcessState, err)
		a.ended(req.Job, &code)
	}()
	return bus.OKReply(st)
}

// exitCode is the run's status as a shell reports it: the exit status, or
// 128 + the signal that killed it.
func exitCode(ps *os.ProcessState, err error) int {
	if ps == nil {
		return -1
	}
	if ws, ok := ps.Sys().(syscall.WaitStatus); ok && ws.Signaled() {
		return 128 + int(ws.Signal())
	}
	return ps.ExitCode()
}

// ended records a run's end: `exited` with its code, or `lost` (nil) for an
// adopted run whose status cannot be collected. It removes the job's
// containers, writes result.json (C4) and publishes the event.
func (a *actions) ended(job int64, code *int) {
	if !a.c.alive() {
		return
	}
	now := a.c.now()
	a.mu.Lock()
	r, ok := a.runs[job]
	if !ok || r.Status != bus.RunRunning {
		a.mu.Unlock()
		return
	}
	if r.killTimer != nil {
		r.killTimer.Stop()
	}
	r.endedAt, r.Ended, r.ExitCode = now, bus.Timestamp(now), code
	if code == nil {
		r.Status = bus.RunLost
	} else {
		r.Status = bus.RunExited
	}
	rc := *r
	a.mu.Unlock()
	a.removeContainers(job)
	a.writeResult(rc)
	a.c.log.Printf("actions: job %d: %s %s (exit %v)", job, rc.Action, rc.Status, derefInt(code))
	a.runChanged(job)
}

func derefInt(p *int) any {
	if p == nil {
		return "unknown"
	}
	return *p
}

// doStop signals the run's process group and replies at once; SIGKILL
// follows after the grace (C1). Another job's run needs force and run_job.
func (a *actions) doStop(req bus.Request) bus.Reply {
	sig := syscall.SIGTERM
	switch req.Signal {
	case "", "TERM":
	case "INT":
		sig = syscall.SIGINT
	default:
		return bus.ErrReply(bus.CodeBadRequest, fmt.Sprintf("signal %q: want INT or TERM", req.Signal), "")
	}
	grace := DefaultStopGraceS
	if req.GraceS != nil {
		grace = *req.GraceS
	}
	if grace < 0 || grace > MaxStopGraceS {
		return bus.ErrReply(bus.CodeBadRequest, fmt.Sprintf("grace_s %d is outside 0..%d", grace, MaxStopGraceS), "")
	}
	target := req.Job
	if req.RunJob != nil {
		if *req.RunJob != req.Job && !req.Force {
			return bus.ErrReply(bus.CodeNotOwner, fmt.Sprintf("job %d's run belongs to job %d; stopping it needs force", *req.RunJob, *req.RunJob), "")
		}
		target = *req.RunJob
	}
	if target <= 0 {
		return bus.ErrReply(bus.CodeBadRequest, "stop names no job's run", "")
	}
	a.mu.Lock()
	r, ok := a.runs[target]
	if !ok {
		// Nothing runs yet: remember the stop, so a run that arrives after it
		// is refused (the job's TERM came while its `run` was in flight).
		a.stopped[target] = a.c.now()
		st := a.runStateLocked(target)
		a.mu.Unlock()
		return bus.OKReply(st)
	}
	if r.Status != bus.RunRunning {
		st := a.runStateLocked(target)
		a.mu.Unlock()
		return bus.OKReply(st)
	}
	if r.PID <= 0 {
		// Still being prepared: no process yet. doRun sees the stop before
		// it starts one and refuses the run.
		r.Stopped = true
		a.stopped[target] = a.c.now()
		st := a.runStateLocked(target)
		a.mu.Unlock()
		return bus.OKReply(st)
	}
	a.signalLocked(r, sig, time.Duration(grace)*time.Second)
	st := a.runStateLocked(target)
	a.mu.Unlock()
	a.c.save()
	return bus.OKReply(st)
}

// signalLocked sends sig to the run's process group and arms SIGKILL.
func (a *actions) signalLocked(r *run, sig syscall.Signal, grace time.Duration) {
	r.Stopped = true
	pgid := r.PID
	if pgid <= 0 {
		return // kill(-0) would signal devai-control's own process group
	}
	if err := syscall.Kill(-pgid, sig); err != nil && !errors.Is(err, syscall.ESRCH) {
		a.c.log.Printf("actions: job %d: signal %v to group %d: %v", r.Job, sig, pgid, err)
	}
	if r.killTimer != nil {
		r.killTimer.Stop()
	}
	job := r.Job
	r.killTimer = time.AfterFunc(grace, func() {
		a.mu.Lock()
		cur, ok := a.runs[job]
		running := ok && cur.Status == bus.RunRunning
		a.mu.Unlock()
		if running {
			_ = syscall.Kill(-pgid, syscall.SIGKILL)
		}
	})
}

// tick stops a run whose lease lapsed (TERM, 30 s grace; C1), forgets
// ended runs after run_keep_s, and watches adopted runs.
func (a *actions) tick(_ context.Context, now time.Time) {
	lease := a.c.cfg.Lease()
	var gone []int64
	a.mu.Lock()
	for job, r := range a.runs {
		switch r.Status {
		case bus.RunRunning:
			if r.PID == 0 {
				continue // still being prepared
			}
			if lease > 0 && !r.Stopped && now.Sub(r.leaseAt) > lease {
				a.c.log.Printf("actions: job %d's lease lapsed (last %s); stopping its run", job, bus.Timestamp(r.leaseAt))
				a.signalLocked(r, syscall.SIGTERM, leaseStopGrace)
			}
			if r.adopted && !procAlive(r.PID, r.StartTick) {
				gone = append(gone, job)
			}
		default:
			if now.Sub(r.endedAt) > a.c.cfg.RunKeep() {
				delete(a.runs, job)
			}
		}
	}
	for job, t := range a.stopped {
		if now.Sub(t) > a.c.cfg.RunKeep() {
			delete(a.stopped, job)
		}
	}
	a.mu.Unlock()
	for _, job := range gone {
		a.ended(job, nil)
	}
}

func (a *actions) persist() json.RawMessage {
	a.mu.Lock()
	defer a.mu.Unlock()
	runs := make([]run, 0, len(a.runs))
	for _, r := range a.runs {
		if r.PID != 0 {
			runs = append(runs, *r)
		}
	}
	sort.Slice(runs, func(i, j int) bool { return runs[i].Job < runs[j].Job })
	b, _ := json.Marshal(map[string]any{"runs": runs})
	return b
}

// adopt takes over the runs the state file records: a run whose process
// still lives (same pid, same start time) is watched until it is gone and
// then published `lost`; every other running one is published `lost` at
// once. With a podman CLI, containers labelled devai.job=<id> whose job has
// no adopted run are removed (C1, C8).
func (a *actions) adopt(ctx context.Context, raw json.RawMessage) {
	var rec struct {
		Runs []run `json:"runs"`
	}
	if json.Unmarshal(raw, &rec) != nil {
		return
	}
	now := a.c.now()
	var lost []int64
	a.mu.Lock()
	for i := range rec.Runs {
		r := rec.Runs[i]
		r.leaseAt, r.startedAt = now, now
		if t, err := time.Parse(time.RFC3339Nano, r.Since); err == nil {
			r.startedAt = t
		}
		if r.Status != bus.RunRunning {
			if t, err := time.Parse(time.RFC3339Nano, r.Ended); err == nil {
				r.endedAt = t
			}
			rr := r
			a.runs[r.Job] = &rr
			continue
		}
		rr := r
		rr.adopted = true
		a.runs[r.Job] = &rr
		if !procAlive(r.PID, r.StartTick) {
			lost = append(lost, r.Job)
			continue
		}
		a.c.log.Printf("actions: adopted job %d's run of %s (pid %d)", r.Job, r.Action, r.PID)
		if rr.Stopped {
			// A stop was under way when devai-control went down; its SIGKILL
			// timer went with it. TERM again and re-arm it.
			a.signalLocked(&rr, syscall.SIGTERM, time.Duration(DefaultStopGraceS)*time.Second)
		}
	}
	a.mu.Unlock()
	for _, job := range lost {
		a.c.log.Printf("actions: job %d's run could not be adopted; lost", job)
		a.ended(job, nil)
	}
	a.removeOrphanContainers(ctx)
}

// ensureResultsDir creates results/<jobid>/ when the prolog has not (tests,
// a run outside Slurm): owned by the caller, group-writable, setgid, like
// the prolog's (C4, I26). Go's FileMode has its own setgid bit, so 0o2775
// would not set it.
func ensureResultsDir(dir string) error {
	if _, err := os.Stat(dir); err == nil {
		return nil
	}
	if err := os.MkdirAll(dir, 0o775); err != nil {
		return err
	}
	return os.Chmod(dir, 0o775|os.ModeSetgid)
}

// procStartTick is field 22 of /proc/<pid>/stat: the process's start time,
// which tells a reused pid apart.
func procStartTick(pid int) uint64 {
	b, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return 0
	}
	s := string(b)
	i := strings.LastIndexByte(s, ')')
	if i < 0 {
		return 0
	}
	f := strings.Fields(s[i+1:])
	if len(f) < 20 {
		return 0
	}
	v, _ := strconv.ParseUint(f[19], 10, 64)
	return v
}

func procAlive(pid int, tick uint64) bool {
	if pid <= 0 {
		return false
	}
	t := procStartTick(pid)
	if t == 0 || (tick != 0 && t != tick) {
		return false
	}
	b, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return false
	}
	s := string(b)
	if i := strings.LastIndexByte(s, ')'); i >= 0 && i+2 < len(s) && s[i+2] == 'Z' {
		return false
	}
	return true
}

// removeContainers removes the containers a run started, labelled
// devai.job=<id>, when a podman CLI is configured (C8).
func (a *actions) removeContainers(job int64) {
	if a.cfg.Podman == "" {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, a.cfg.Podman, "ps", "-aq", "--filter", "label=devai.job="+strconv.FormatInt(job, 10)).Output()
	if err != nil {
		a.c.log.Printf("actions: job %d: podman ps: %v", job, err)
		return
	}
	ids := strings.Fields(string(out))
	if len(ids) == 0 {
		return
	}
	args := append([]string{"rm", "-f"}, ids...)
	if b, err := exec.CommandContext(ctx, a.cfg.Podman, args...).CombinedOutput(); err != nil {
		a.c.log.Printf("actions: job %d: podman rm: %v: %s", job, err, strings.TrimSpace(string(b)))
	}
}

// removeOrphanContainers removes every container labelled devai.job=<id>
// whose job has no adopted run: devai-control's container was recreated,
// and its runs died with it.
func (a *actions) removeOrphanContainers(ctx context.Context) {
	if a.cfg.Podman == "" {
		return
	}
	cctx, cancel := context.WithTimeout(ctx, 120*time.Second)
	defer cancel()
	out, err := exec.CommandContext(cctx, a.cfg.Podman, "ps", "-a", "--filter", "label=devai.job",
		"--format", `{{.ID}} {{index .Labels "devai.job"}}`).Output()
	if err != nil {
		a.c.log.Printf("actions: podman ps: %v", err)
		return
	}
	a.mu.Lock()
	alive := map[string]bool{}
	for job, r := range a.runs {
		if r.Status == bus.RunRunning {
			alive[strconv.FormatInt(job, 10)] = true
		}
	}
	a.mu.Unlock()
	var ids []string
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		f := strings.Fields(line)
		if len(f) == 2 && !alive[f[1]] {
			ids = append(ids, f[0])
		}
	}
	if len(ids) == 0 {
		return
	}
	args := append([]string{"rm", "-f"}, ids...)
	if b, err := exec.CommandContext(cctx, a.cfg.Podman, args...).CombinedOutput(); err != nil {
		a.c.log.Printf("actions: podman rm: %v: %s", err, strings.TrimSpace(string(b)))
	}
}
