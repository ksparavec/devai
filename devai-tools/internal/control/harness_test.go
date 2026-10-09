package control

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/nats-io/nats.go"

	"github.com/sparavec/devai-tools/internal/buscli"
	"github.com/sparavec/gpu-arbiter/pkg/bus"
	"github.com/sparavec/gpu-arbiter/pkg/bus/bustest"
)

// The test lease (Phase 5 step 5: "a 2 s lease"), and a reply grace scaled
// down with it (production: 2 s against a 60 s lease).
const (
	testLease      = 2 * time.Second
	testHeartbeat  = time.Second
	leaseEvery     = 400 * time.Millisecond
	testReplyGrace = 200 * time.Millisecond
)

type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

// rig is one devai-control under test, with its fake supervisord and card.
type rig struct {
	t      *testing.T
	srv    *bustest.Server
	cfg    *Config
	sup    *fakeSupervisord
	mem    *fakeMemory
	ctrl   *Controller
	cancel context.CancelFunc
	done   chan struct{}
	logs   *syncBuffer
}

// engineConfig is the engines test container's control.yaml: one fake
// engine whose load takes delay seconds.
func engineConfig(t *testing.T, srv *bustest.Server, delay float64) *Config {
	dir := t.TempDir()
	return &Config{
		Container:  "engines",
		Bus:        srv.Config("control-engines"),
		Supervisor: SupervisorConfig{ProgramDir: filepath.Join(dir, "supervisor")},
		StateFile:  filepath.Join(dir, "control.json"),
		LeaseS:     testLease.Seconds(), HeartbeatS: testHeartbeat.Seconds(), PollS: 0.2, RunKeepS: 600,
		GPU: "auto", ResultsDir: filepath.Join(dir, "results"), MaxContext: 262144,
		Services: map[string]ServiceConfig{
			"fake": {Kind: KindFake, MiB: 2048, LoadDelayS: delay, Script: "/opt/test/vram-ballast.py", LogDir: filepath.Join(dir, "logs")},
		},
	}
}

// startRig starts a controller for cfg. sup and mem may be shared with a
// controller this one replaces (a devai-control restart).
func startRig(t *testing.T, srv *bustest.Server, cfg *Config, sup *fakeSupervisord, mem *fakeMemory) *rig {
	t.Helper()
	if mem == nil {
		mem = &fakeMemory{used: 2}
	}
	if sup == nil && cfg.Supervisor.ProgramDir != "" {
		sup = newFakeSupervisord(t, cfg.Supervisor.ProgramDir, mem)
	}
	if sup != nil {
		cfg.Supervisor.Socket = sup.socket
	}
	logs := &syncBuffer{}
	c, err := New(cfg, Options{Memory: mem, Logger: log.New(logs, "", 0)})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	r := &rig{t: t, srv: srv, cfg: cfg, sup: sup, mem: mem, ctrl: c, cancel: cancel, done: make(chan struct{}), logs: logs}
	go func() {
		defer close(r.done)
		if err := c.Run(ctx); err != nil {
			t.Errorf("Run: %v", err)
		}
	}()
	t.Cleanup(r.kill)
	// Wait until it answers.
	nc := r.connect("operator")
	deadline := time.Now().Add(5 * time.Second)
	for {
		svc := ""
		for n := range cfg.Services {
			svc = n
		}
		ctx, cc := context.WithTimeout(context.Background(), 500*time.Millisecond)
		_, err := bus.Do(ctx, nc, bus.CommandSubject(cfg.Container, svc, bus.VerbState), bus.Request{V: 1})
		cc()
		if err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("devai-control did not answer: %v\n%s", err, logs.String())
		}
		time.Sleep(50 * time.Millisecond)
	}
	return r
}

// kill ends the controller the way SIGKILL would: nothing is cleaned up.
func (r *rig) kill() {
	r.cancel()
	<-r.done
}

func (r *rig) connect(user string) *nats.Conn {
	r.t.Helper()
	nc, err := bus.Connect(r.srv.Config(user), "test-"+user)
	if err != nil {
		r.t.Fatal(err)
	}
	r.t.Cleanup(nc.Close)
	return nc
}

// ask sends one command as user and returns the reply.
func (r *rig) ask(user, service, verb string, req bus.Request, timeout time.Duration) bus.Reply {
	r.t.Helper()
	req.V = 1
	nc := r.connect(user)
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	rep, err := bus.Do(ctx, nc, bus.CommandSubject(r.cfg.Container, service, verb), req)
	if err != nil {
		r.t.Fatalf("%s %s: %v", service, verb, err)
	}
	return rep
}

func (r *rig) state(service string) bus.State {
	r.t.Helper()
	rep := r.ask("operator", service, bus.VerbState, bus.Request{}, 5*time.Second)
	if !rep.OK {
		r.t.Fatalf("state: %+v", rep.Error)
	}
	return *rep.State
}

func (r *rig) waitStatus(service, status string, within time.Duration) bus.State {
	r.t.Helper()
	deadline := time.Now().Add(within)
	for {
		st := r.state(service)
		if st.Status == status {
			return st
		}
		if time.Now().After(deadline) {
			r.t.Fatalf("%s: status %q, want %q within %s\n%s", service, st.Status, status, within, r.logs.String())
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// requestAsJob runs `devai-bus request` in-process as user jobs.
func (r *rig) requestAsJob(subject, body string, job int64, timeout time.Duration) (int, bus.Reply) {
	var out, errb syncBuffer
	code := buscli.Request(context.Background(), buscli.RequestOptions{
		Bus: r.srv.Config("jobs"), Subject: subject, Body: body, Job: job, Timeout: timeout,
		ReplyGrace: testReplyGrace, Stdout: &out, Stderr: &errb,
	})
	var rep bus.Reply
	_ = json.Unmarshal(bytes.TrimSpace([]byte(out.String())), &rep)
	return code, rep
}

// hold runs `devai-bus hold` in-process in the background.
type holdRun struct {
	done chan int
	code int
	err  *syncBuffer
}

func (r *rig) startHold(subject string, job int64, opts func(*buscli.HoldOptions)) *holdRun {
	h := &holdRun{done: make(chan int, 1), err: &syncBuffer{}}
	o := buscli.HoldOptions{Bus: r.srv.Config("jobs"), Subject: subject, Job: job,
		LeaseInterval: leaseEvery, ConnectTimeout: 5 * time.Second, Stderr: h.err}
	if opts != nil {
		opts(&o)
	}
	ctx, cancel := context.WithCancel(context.Background())
	r.t.Cleanup(cancel)
	go func() { h.done <- buscli.Hold(ctx, o) }()
	return h
}

func (h *holdRun) wait(t *testing.T, within time.Duration) int {
	t.Helper()
	select {
	case c := <-h.done:
		return c
	case <-time.After(within):
		t.Fatalf("the hold did not end within %s\n%s", within, h.err.String())
	}
	return -1
}

func (h *holdRun) running(within time.Duration) bool {
	select {
	case c := <-h.done:
		h.done <- c
		return false
	case <-time.After(within):
		return true
	}
}

func loadBody(timeout int) string {
	return `{"launch":{"backend":"fake","model":"qwen3.8:27b-q4_K_M","ctx":131072,"kv_cache_dtype":"","mtp":null},"timeout_s":` + strconv.Itoa(timeout) + `}`
}

// watchSubject collects messages on a subject as user.
type watcher struct {
	mu   sync.Mutex
	msgs []*nats.Msg
}

func (r *rig) watch(user, subject string) *watcher {
	r.t.Helper()
	w := &watcher{}
	nc := r.connect(user)
	if _, err := nc.Subscribe(subject, func(m *nats.Msg) {
		w.mu.Lock()
		w.msgs = append(w.msgs, m)
		w.mu.Unlock()
	}); err != nil {
		r.t.Fatal(err)
	}
	_ = nc.Flush()
	return w
}

func (w *watcher) all() []*nats.Msg {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]*nats.Msg(nil), w.msgs...)
}

func (w *watcher) states(job int64) []string {
	var out []string
	for _, m := range w.all() {
		if s, err := bus.DecodeState(m.Data); err == nil && s.JobIs(job) {
			out = append(out, s.Status)
		}
	}
	return out
}

func discard() io.Writer { return io.Discard }

// killGroup is a test cleanup for an action's process group.
func killGroup(pid int) {
	if pid > 0 {
		_ = syscall.Kill(-pid, syscall.SIGKILL)
	}
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}
