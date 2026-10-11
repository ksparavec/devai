package control

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/sparavec/gpu-arbiter/pkg/bus"
)

// Load limits (C1, C2).
const (
	DefaultLoadTimeoutS   = 600
	MaxLoadTimeoutS       = 840
	DefaultUnloadTimeoutS = 60
	MemorySlackMiB        = 256
	minCtx                = 1024
	logTailBytes          = 16 * 1024
)

// launch is C2's `launch` object. Phase 5's fake engine reads backend,
// model and ctx; the rest is checked for shape and kept, so a launch the
// router builds for a real engine passes the same decoder.
type launch struct {
	Backend         string          `json:"backend"`
	Model           string          `json:"model"`
	Ctx             int             `json:"ctx"`
	KVCacheDtype    string          `json:"kv_cache_dtype"`
	FlashAttention  *bool           `json:"flash_attention"`
	NumGPU          *int            `json:"num_gpu"`
	MemFraction     *float64        `json:"mem_fraction"`
	MaxNumSeqs      *int            `json:"max_num_seqs"`
	ToolParser      string          `json:"tool_parser"`
	ReasoningParser string          `json:"reasoning_parser"`
	MTP             json.RawMessage `json:"mtp"`
	SleepMode       *bool           `json:"sleep_mode"`
}

var (
	parserRE = regexp.MustCompile(`^[a-z0-9_]{1,64}$`)
	modelRE  = regexp.MustCompile(`^[A-Za-z0-9._:@/+-]{1,256}$`)
)

// safeModelName mirrors the router's isSafeModelName (no NUL, no ".."
// segment) and adds a plain character set, since the name reaches a
// command line.
func safeModelName(s string) bool {
	if !modelRE.MatchString(s) {
		return false
	}
	for _, seg := range strings.Split(s, "/") {
		if seg == ".." || seg == "." || seg == "" {
			return false
		}
	}
	return true
}

func decodeLaunch(raw json.RawMessage, backend string, maxCtx int) (launch, error) {
	var l launch
	if len(raw) == 0 {
		return l, errors.New("load needs a launch")
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&l); err != nil {
		return l, fmt.Errorf("bad launch: %w", err)
	}
	switch {
	case l.Backend != backend:
		return l, fmt.Errorf("launch backend %q does not match the service (%s)", l.Backend, backend)
	case !safeModelName(l.Model):
		return l, fmt.Errorf("launch model %q is not a safe model name", l.Model)
	case l.Ctx < minCtx || l.Ctx > maxCtx:
		return l, fmt.Errorf("launch ctx %d is outside %d..%d", l.Ctx, minCtx, maxCtx)
	case l.MemFraction != nil && (*l.MemFraction < 0.10 || *l.MemFraction > 0.98):
		return l, fmt.Errorf("launch mem_fraction %v is outside 0.10..0.98", *l.MemFraction)
	case l.MaxNumSeqs != nil && *l.MaxNumSeqs < 0:
		return l, errors.New("launch max_num_seqs must not be negative")
	case l.NumGPU != nil && *l.NumGPU != 0 && *l.NumGPU != 999:
		return l, errors.New("launch num_gpu must be 999 or 0")
	case l.ToolParser != "" && !parserRE.MatchString(l.ToolParser),
		l.ReasoningParser != "" && !parserRE.MatchString(l.ReasoningParser):
		return l, errors.New("launch parser names must match ^[a-z0-9_]{1,64}$")
	}
	return l, nil
}

// sameLaunch: backend, model, ctx, kv_cache_dtype and mtp equal (C2).
func sameLaunch(a, b json.RawMessage) bool {
	var x, y launch
	if json.Unmarshal(a, &x) != nil || json.Unmarshal(b, &y) != nil {
		return false
	}
	return x.Backend == y.Backend && x.Model == y.Model && x.Ctx == y.Ctx &&
		x.KVCacheDtype == y.KVCacheDtype && canonical(x.MTP) == canonical(y.MTP)
}

func canonical(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return "null"
	}
	var v any
	if json.Unmarshal(raw, &v) != nil {
		return string(raw)
	}
	b, _ := json.Marshal(v)
	return string(b)
}

type loadOp struct {
	gen   uint64
	done  chan struct{}
	reply bus.Reply
	// started is closed once the load's program start has returned; an
	// unload waits for it, so it never stops a program that is not started
	// yet and then leaves it running.
	started chan struct{}
}

type unloadOp struct {
	done  chan struct{}
	reply bus.Reply
}

// fakeEngine is Phase 5's engine: one supervisord program, the VRAM ballast,
// started by `load` and stopped by `unload`, treated as vllm (C1). It is
// "serving" once the ballast has printed `ready <MiB>`.
type fakeEngine struct {
	c    *Controller
	name string
	cfg  ServiceConfig

	mu        sync.Mutex
	status    string
	job       *int64
	launch    json.RawMessage
	since     time.Time
	err       *bus.Error
	startedAt time.Time
	timeoutS  int
	beforeMiB int // -1: unknown (no GPU)
	leaseAt   time.Time
	logPath   string
	gen       uint64
	load      *loadOp
	unload    *unloadOp
}

func newFakeEngine(c *Controller, name string, sc ServiceConfig) *fakeEngine {
	return &fakeEngine{c: c, name: name, cfg: sc, status: bus.StatusIdle, since: c.now(), beforeMiB: -1}
}

func (e *fakeEngine) program() string {
	if e.cfg.Program != "" {
		return e.cfg.Program
	}
	return e.name
}

// stateLocked is the C1 state; e.mu is held.
func (e *fakeEngine) stateLocked() bus.State {
	var job *int64
	if e.job != nil {
		j := *e.job
		job = &j
	}
	return bus.State{
		Container: e.c.cfg.Container, Service: e.name, Status: e.status, Job: job,
		Launch: e.launch, Since: bus.Timestamp(e.since), Build: "fake", Error: e.err,
	}
}

func (e *fakeEngine) event() bus.State {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.stateLocked()
}

func (e *fakeEngine) changed() {
	e.c.save()
	e.c.publish(e.name)
}

func (e *fakeEngine) handle(ctx context.Context, verb string, req bus.Request) (bus.Reply, bool) {
	switch verb {
	case bus.VerbState:
		return bus.OKReply(e.event()), true
	case bus.VerbLoad:
		return e.doLoad(ctx, req), true
	case bus.VerbUnload:
		return e.doUnload(ctx, req.Job, req.Target, req.Force, req.TimeoutS), true
	case bus.VerbSleep:
		e.mu.Lock()
		owner := e.job != nil && *e.job == req.Job
		busyOther := e.job != nil && !owner
		e.mu.Unlock()
		if busyOther {
			return bus.ErrReply(bus.CodeNotOwner, "another job's model", ""), true
		}
		return bus.ErrReply(bus.CodeUnsupported, "the fake engine has no sleep mode", ""), true
	case bus.VerbLease:
		e.mu.Lock()
		if e.job != nil && *e.job == req.Job {
			now := e.c.now()
			if now.After(e.leaseAt) {
				e.leaseAt = now
			}
		}
		e.mu.Unlock()
		return bus.Reply{}, false
	case bus.VerbStop:
		// Supervisor verb (operator only, no ownership): a forced unload of
		// the loaded model, also while it is loading; job becomes null.
		r := e.doUnload(ctx, req.Job, bus.TargetActive, true, req.TimeoutS)
		if !r.OK {
			return r, true
		}
		return bus.OKReply(e.event()), true
	case bus.VerbStart:
		return bus.OKReply(e.event()), true
	case bus.VerbRestart, bus.VerbReload:
		e.mu.Lock()
		st := e.stateLocked()
		busy := e.status == bus.StatusLoading || e.status == bus.StatusServing || e.status == bus.StatusWaking ||
			e.status == bus.StatusSleeping || e.status == bus.StatusUnloading
		e.mu.Unlock()
		if busy {
			return bus.ErrReply(bus.CodeBusy, fmt.Sprintf("a model is %s; %s is refused", st.Status, verb), ""), true
		}
		return bus.OKReply(st), true
	}
	return unknownVerb("fake engine", verb), true
}

func (e *fakeEngine) doLoad(ctx context.Context, req bus.Request) bus.Reply {
	l, err := decodeLaunch(req.Launch, KindFake, e.c.cfg.MaxContext)
	if err != nil {
		return bus.ErrReply(bus.CodeBadRequest, err.Error(), "")
	}
	timeoutS := DefaultLoadTimeoutS
	if req.TimeoutS != nil {
		timeoutS = *req.TimeoutS
	}
	if timeoutS <= 0 || timeoutS > MaxLoadTimeoutS {
		return bus.ErrReply(bus.CodeBadRequest, fmt.Sprintf("timeout_s %d is outside 1..%d", timeoutS, MaxLoadTimeoutS), "")
	}
	if req.Job <= 0 {
		return bus.ErrReply(bus.CodeBadRequest, "a load belongs to a job: job must be its Slurm id", "")
	}
	_ = l
	launchRaw := json.RawMessage(append([]byte(nil), req.Launch...))

	e.mu.Lock()
	switch {
	case e.unload != nil:
		e.mu.Unlock()
		return bus.ErrReply(bus.CodeBusy, "an unload is under way", "")
	case e.status == bus.StatusLoading && e.job != nil && *e.job == req.Job && sameLaunch(e.launch, launchRaw):
		op := e.load
		e.mu.Unlock()
		return waitLoad(ctx, op)
	case e.status == bus.StatusServing && e.job != nil && *e.job == req.Job && sameLaunch(e.launch, launchRaw):
		st := e.stateLocked()
		e.leaseAt = e.c.now()
		e.mu.Unlock()
		return bus.OKReply(st)
	case e.job != nil && (e.status == bus.StatusLoading || e.status == bus.StatusServing ||
		e.status == bus.StatusWaking || e.status == bus.StatusUnloading):
		st := e.stateLocked()
		e.mu.Unlock()
		if *st.Job == req.Job {
			return bus.ErrReply(bus.CodeBusy, "this job's model is loaded with another launch", "")
		}
		return bus.ErrReply(bus.CodeBusy, fmt.Sprintf("job %d's model is %s in this container", *st.Job, st.Status), "")
	}
	e.gen++
	gen := e.gen
	op := &loadOp{gen: gen, done: make(chan struct{}), started: make(chan struct{})}
	job := req.Job
	now := e.c.now()
	e.load, e.status, e.job, e.launch, e.since, e.err = op, bus.StatusLoading, &job, launchRaw, now, nil
	e.startedAt, e.timeoutS = now, timeoutS
	e.logPath = filepath.Join(e.cfg.LogDir, strconv.FormatInt(job, 10)+".log")
	e.beforeMiB = -1
	e.mu.Unlock()

	if e.c.mem.Present() {
		if used, err := e.c.mem.UsedMiB(ctx); err == nil {
			e.mu.Lock()
			if e.gen == gen {
				e.beforeMiB = used
			}
			e.mu.Unlock()
		}
	}
	e.changed()
	err = e.startProgram(ctx, job)
	close(op.started)
	if err != nil {
		e.finishLoad(gen, false, bus.CodeLoadFailed, "the engine program did not start: "+err.Error(), "")
	} else {
		e.mu.Lock()
		current := e.gen == gen && e.load == op
		e.mu.Unlock()
		if current { // otherwise an unload took the load over and stops the program
			go e.watch(gen, now.Add(time.Duration(timeoutS)*time.Second))
		}
	}
	return waitLoad(ctx, op)
}

func waitLoad(ctx context.Context, op *loadOp) bus.Reply {
	select {
	case <-op.done:
		return op.reply
	case <-ctx.Done():
		return bus.ErrReply(bus.CodeTimeout, "devai-control is stopping", "")
	}
}

// programFile is the supervisord program devai-control writes for a load.
func (e *fakeEngine) programFile(job int64) string {
	py := e.cfg.Python
	if py == "" {
		py = "/usr/bin/python3"
	}
	cmd := fmt.Sprintf("exec %s %s %d", py, e.cfg.Script, e.cfg.MiB)
	if e.cfg.LoadDelayS > 0 {
		cmd = fmt.Sprintf("sleep %s && %s", strconv.FormatFloat(e.cfg.LoadDelayS, 'f', -1, 64), cmd)
	}
	return fmt.Sprintf(`; written by devai-control for job %d -- do not edit
[program:%s]
command=/bin/sh -c "%s"
autostart=false
autorestart=false
startsecs=0
startretries=0
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=30
stdout_logfile=%s
stdout_logfile_maxbytes=0
redirect_stderr=true
`, job, e.program(), cmd, filepath.Join(e.cfg.LogDir, strconv.FormatInt(job, 10)+".log"))
}

var plainPathRE = regexp.MustCompile(`^[A-Za-z0-9._/+-]+$`)

func (e *fakeEngine) startProgram(ctx context.Context, job int64) error {
	if e.c.sup == nil {
		return errors.New("no supervisord configured")
	}
	for _, p := range []string{e.cfg.Script, e.cfg.LogDir, e.cfg.Python} {
		if p != "" && !plainPathRE.MatchString(p) {
			return fmt.Errorf("path %q is not plain", p)
		}
	}
	if err := os.MkdirAll(e.cfg.LogDir, 0o755); err != nil {
		return err
	}
	if err := os.MkdirAll(e.c.cfg.Supervisor.ProgramDir, 0o755); err != nil {
		return err
	}
	path := filepath.Join(e.c.cfg.Supervisor.ProgramDir, e.program()+".conf")
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, []byte(e.programFile(job)), 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		return err
	}
	if _, err := e.c.sup.Update(ctx); err != nil {
		return fmt.Errorf("supervisord update: %w", err)
	}
	if err := e.c.sup.StartProcess(ctx, e.program(), false); err != nil && !IsFault(err, FaultAlreadyStarted) {
		return err
	}
	return nil
}

func (e *fakeEngine) stopProgram(ctx context.Context) error {
	if e.c.sup == nil {
		return nil
	}
	err := e.c.sup.StopProcess(ctx, e.program(), true)
	if err != nil && (IsFault(err, FaultNotRunning) || IsFault(err, FaultBadName)) {
		return nil
	}
	return err
}

func readTail(path string, n int64) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return ""
	}
	off := st.Size() - n
	if off < 0 {
		off = 0
	}
	buf := make([]byte, st.Size()-off)
	if _, err := f.ReadAt(buf, off); err != nil && len(buf) == 0 {
		return ""
	}
	return string(buf)
}

var readyRE = regexp.MustCompile(`(?m)^ready [0-9]+\s*$`)

// watch follows a load to its end: ready (serving), the program gone
// (failed, with the log tail) or the timeout (failed, timeout). An unload
// ends it by moving the generation on.
func (e *fakeEngine) watch(gen uint64, deadline time.Time) {
	t := time.NewTicker(250 * time.Millisecond)
	defer t.Stop()
	for {
		select {
		case <-e.c.runCtx.Done():
			return
		case <-t.C:
		}
		e.mu.Lock()
		if e.gen != gen || e.load == nil {
			e.mu.Unlock()
			return
		}
		logPath := e.logPath
		e.mu.Unlock()
		if readyRE.MatchString(readTail(logPath, logTailBytes)) {
			e.finishLoad(gen, true, "", "", "")
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		info, err := e.c.sup.GetProcessInfo(ctx, e.program())
		cancel()
		if err == nil && !info.Running() {
			// One more look: the program may have printed ready and then
			// been stopped between the two reads.
			if readyRE.MatchString(readTail(logPath, logTailBytes)) {
				e.finishLoad(gen, true, "", "", "")
				return
			}
			e.finishLoad(gen, false, bus.CodeLoadFailed,
				fmt.Sprintf("the engine program is %s (exit status %d) before it served", info.StateName, info.ExitStatus),
				readTail(logPath, logTailBytes))
			return
		}
		if e.c.now().After(deadline) {
			e.timeoutLoad(gen)
			return
		}
	}
}

// timeoutLoad ends a load that did not reach serving within its timeout_s:
// it claims the load (as an unload would, so nothing else acts on this
// generation), stops the program, and records failed with code timeout.
func (e *fakeEngine) timeoutLoad(gen uint64) {
	e.mu.Lock()
	if !e.c.alive() || e.gen != gen || e.load == nil || e.unload != nil {
		e.mu.Unlock()
		return
	}
	op := e.load
	e.load = nil
	uop := &unloadOp{done: make(chan struct{})}
	e.unload = uop
	e.gen++
	e.status, e.since = bus.StatusUnloading, e.c.now()
	logPath := e.logPath
	e.mu.Unlock()
	e.changed()
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	err := e.stopProgram(ctx)
	cancel()
	msg := "the load did not reach serving within timeout_s"
	if err != nil {
		msg += "; stopping the engine program: " + err.Error()
	}
	tail := readTail(logPath, logTailBytes)
	e.mu.Lock()
	e.status, e.since = bus.StatusFailed, e.c.now()
	e.err = &bus.Error{Code: bus.CodeTimeout, Message: msg, LogTail: bus.TrimLogTail(tail)}
	e.unload = nil
	op.reply = bus.ErrReply(bus.CodeTimeout, msg, tail)
	close(op.done)
	uop.reply = bus.OKReply(e.stateLocked())
	close(uop.done)
	e.mu.Unlock()
	e.changed()
}

func (e *fakeEngine) finishLoad(gen uint64, ok bool, code, msg, tail string) {
	if !e.c.alive() {
		return
	}
	e.mu.Lock()
	if e.load == nil || e.load.gen != gen {
		e.mu.Unlock()
		return
	}
	now := e.c.now()
	e.since = now
	if ok {
		e.status = bus.StatusServing
		e.leaseAt = now // the lease clock counts from the answer (C1)
		e.err = nil
	} else {
		e.status = bus.StatusFailed
		e.err = &bus.Error{Code: code, Message: msg, LogTail: bus.TrimLogTail(tail)}
	}
	op := e.load
	e.load = nil
	if ok {
		op.reply = bus.OKReply(e.stateLocked())
	} else {
		op.reply = bus.ErrReply(code, msg, tail)
	}
	close(op.done)
	e.mu.Unlock()
	e.changed()
}

// doUnload is C1's unload (and the engine's supervisor `stop`, and the
// lease's end): it stops a load in progress too, joins an unload under way,
// and is a no-op once the job's model is gone.
func (e *fakeEngine) doUnload(ctx context.Context, job int64, target string, force bool, timeoutS *int) bus.Reply {
	if target != "" && target != bus.TargetActive && target != bus.TargetSleeper {
		return bus.ErrReply(bus.CodeBadRequest, fmt.Sprintf("target %q: want active or sleeper", target), "")
	}
	tmo := DefaultUnloadTimeoutS
	if timeoutS != nil {
		tmo = *timeoutS
	}
	if tmo <= 0 || tmo > MaxLoadTimeoutS {
		return bus.ErrReply(bus.CodeBadRequest, fmt.Sprintf("timeout_s %d is outside 1..%d", tmo, MaxLoadTimeoutS), "")
	}
	e.mu.Lock()
	if target == bus.TargetSleeper {
		st := e.stateLocked() // the fake engine never sleeps: nothing to unload
		e.mu.Unlock()
		return bus.OKReply(st)
	}
	// Ownership first: a job that does not own the model gets not_owner,
	// also while the owner's unload is under way (C1).
	if e.job != nil && *e.job != job && !force && e.status != bus.StatusIdle && e.status != bus.StatusStopped {
		owner := *e.job
		e.mu.Unlock()
		return bus.ErrReply(bus.CodeNotOwner, fmt.Sprintf("the model belongs to job %d", owner), "")
	}
	if e.unload != nil {
		op := e.unload
		e.mu.Unlock()
		select {
		case <-op.done:
			return op.reply
		case <-ctx.Done():
			return bus.ErrReply(bus.CodeTimeout, "devai-control is stopping", "")
		}
	}
	if e.job == nil || e.status == bus.StatusIdle || e.status == bus.StatusStopped {
		st := e.stateLocked()
		e.mu.Unlock()
		return bus.OKReply(st)
	}
	op := &unloadOp{done: make(chan struct{})}
	e.unload = op
	var started <-chan struct{}
	if e.load != nil {
		lo := e.load
		started = lo.started
		e.load = nil
		lo.reply = bus.ErrReply(bus.CodeLoadFailed, "unloaded", "")
		close(lo.done)
	}
	e.gen++
	wasFailed := e.status == bus.StatusFailed
	e.status, e.since = bus.StatusUnloading, e.c.now()
	before := e.beforeMiB
	e.mu.Unlock()
	if !wasFailed {
		e.changed()
	}

	// A load still starting its program: wait for the start to return, so
	// the program it starts is the one stopped here.
	if started != nil {
		select {
		case <-started:
		case <-ctx.Done():
		}
	}
	reply := bus.Reply{}
	if err := e.stopProgram(ctx); err != nil {
		reply = bus.ErrReply(bus.CodeTimeout, "stopping the engine program: "+err.Error(), "")
	} else if e.c.mem.Present() && before >= 0 {
		deadline := time.Now().Add(time.Duration(tmo) * time.Second)
		for {
			used, err := e.c.mem.UsedMiB(ctx)
			if err == nil && used <= before+MemorySlackMiB {
				break
			}
			if time.Now().After(deadline) {
				reply = bus.ErrReply(bus.CodeTimeout, fmt.Sprintf("used memory %d MiB is not back to %d MiB (+%d) within %d s", used, before, MemorySlackMiB, tmo), "")
				break
			}
			time.Sleep(250 * time.Millisecond)
		}
	}
	e.mu.Lock()
	if reply.V == 0 {
		e.status, e.job, e.launch, e.since, e.err, e.beforeMiB = bus.StatusIdle, nil, nil, e.c.now(), nil, -1
		e.unload = nil
		reply = bus.OKReply(e.stateLocked())
	} else {
		// The program did not stop, or the card did not come back: not
		// idle. The job's hold sees failed, and the epilog's idle check the
		// card.
		e.status, e.since = bus.StatusFailed, e.c.now()
		e.err = &bus.Error{Code: reply.Error.Code, Message: reply.Error.Message}
		e.unload = nil
	}
	op.reply = reply
	close(op.done)
	e.mu.Unlock()
	e.changed()
	return reply
}

// tick runs the lease clock (60 s after the later of the last lease and the
// load's answer, C1) and the liveness check of a serving model.
func (e *fakeEngine) tick(ctx context.Context, now time.Time) {
	e.mu.Lock()
	status, job, leaseAt := e.status, e.job, e.leaseAt
	e.mu.Unlock()
	if status != bus.StatusServing || job == nil {
		return
	}
	if lease := e.c.cfg.Lease(); lease > 0 && now.Sub(leaseAt) > lease {
		e.c.log.Printf("%s: job %d's lease lapsed (last %s); unloading", e.name, *job, bus.Timestamp(leaseAt))
		go e.doUnload(ctx, *job, bus.TargetActive, true, nil)
		return
	}
	cctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	info, err := e.c.sup.GetProcessInfo(cctx, e.program())
	cancel()
	if err != nil || info.Running() {
		return
	}
	e.mu.Lock()
	if e.status == bus.StatusServing && e.job != nil && *e.job == *job && e.unload == nil {
		e.status, e.since = bus.StatusFailed, e.c.now()
		e.err = &bus.Error{Code: bus.CodeLoadFailed,
			Message: fmt.Sprintf("the engine program is %s (exit status %d) while serving", info.StateName, info.ExitStatus),
			LogTail: bus.TrimLogTail(readTail(e.logPath, logTailBytes))}
		e.mu.Unlock()
		e.changed()
		return
	}
	e.mu.Unlock()
}

// engineRecord is what the state file keeps of an engine service.
type engineRecord struct {
	Program   string          `json:"program"`
	Status    string          `json:"status"`
	Job       *int64          `json:"job"`
	Launch    json.RawMessage `json:"launch"`
	StartedAt string          `json:"started_at"`
	TimeoutS  int             `json:"timeout_s"`
	BeforeMiB int             `json:"before_mib"`
	Log       string          `json:"log"`
}

func (e *fakeEngine) persist() json.RawMessage {
	e.mu.Lock()
	defer e.mu.Unlock()
	b, _ := json.Marshal(engineRecord{Program: e.program(), Status: e.status, Job: e.job, Launch: e.launch,
		StartedAt: bus.Timestamp(e.startedAt), TimeoutS: e.timeoutS, BeforeMiB: e.beforeMiB, Log: e.logPath})
	return b
}

// adopt takes over a program supervisord still runs after a devai-control
// restart (C1, "Adoption after a restart"): a load in progress is watched to
// its end with the rest of its timeout, then `serving` or `failed` is
// published for the same job; a program supervisord no longer runs is
// recorded failed.
func (e *fakeEngine) adopt(ctx context.Context, raw json.RawMessage) {
	var r engineRecord
	if json.Unmarshal(raw, &r) != nil || r.Job == nil {
		return
	}
	switch r.Status {
	case bus.StatusLoading, bus.StatusServing, bus.StatusWaking, bus.StatusUnloading:
	default:
		return
	}
	started, _ := time.Parse(time.RFC3339Nano, r.StartedAt)
	cctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	info, err := e.c.sup.GetProcessInfo(cctx, e.program())
	cancel()
	e.mu.Lock()
	job := *r.Job
	e.job, e.launch, e.startedAt, e.timeoutS, e.beforeMiB, e.logPath = &job, r.Launch, started, r.TimeoutS, r.BeforeMiB, r.Log
	e.since = e.c.now()
	if err != nil || !info.Running() {
		e.status = bus.StatusFailed
		e.err = &bus.Error{Code: bus.CodeLoadFailed, Message: "devai-control restarted and the engine program is no longer running",
			LogTail: bus.TrimLogTail(readTail(r.Log, logTailBytes))}
		e.mu.Unlock()
		e.c.log.Printf("%s: job %d's program is gone after the restart; recorded failed", e.name, job)
		return
	}
	switch r.Status {
	case bus.StatusLoading, bus.StatusWaking:
		e.gen++
		gen := e.gen
		e.status = bus.StatusLoading
		startedCh := make(chan struct{})
		close(startedCh)
		e.load = &loadOp{gen: gen, done: make(chan struct{}), started: startedCh}
		e.mu.Unlock()
		e.c.log.Printf("%s: adopted job %d's load in progress", e.name, job)
		go e.watch(gen, started.Add(time.Duration(r.TimeoutS)*time.Second))
	case bus.StatusServing:
		e.status = bus.StatusServing
		e.leaseAt = e.c.now()
		e.mu.Unlock()
		e.c.log.Printf("%s: adopted job %d's serving model", e.name, job)
	case bus.StatusUnloading:
		e.status = bus.StatusServing
		e.leaseAt = e.c.now()
		e.mu.Unlock()
		go e.doUnload(ctx, job, bus.TargetActive, true, nil)
	}
}
