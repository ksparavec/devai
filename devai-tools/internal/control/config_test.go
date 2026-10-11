package control

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sparavec/gpu-arbiter/pkg/bus"
	"github.com/sparavec/gpu-arbiter/pkg/bus/bustest"
)

const engYAML = `container: engines
bus: {url: "nats://devai-slurm:4222", user: control-engines, password_file: /etc/devai/bus/control-engines.pass}
supervisor: {socket: /run/supervisor.sock, program_dir: /run/devai/supervisor}
state_file: /run/devai/control.json
lease_timeout_s: 60
services:
  fake: {kind: fake, mib: 2048, load_delay_s: 0, script: /usr/local/lib/devai/vram-ballast.py, log_dir: /var/log/devai-fake}
`

func TestConfigParsesTheEnginesTestFile(t *testing.T) {
	c, err := ParseConfig([]byte(engYAML))
	if err != nil {
		t.Fatal(err)
	}
	if c.Lease() != time.Minute || c.Heartbeat() != 30*time.Second || c.RunKeep() != 10*time.Minute {
		t.Errorf("defaults: lease %s heartbeat %s keep %s", c.Lease(), c.Heartbeat(), c.RunKeep())
	}
}

func TestConfigRefusals(t *testing.T) {
	cases := map[string]string{
		"unknown key":        engYAML + "frobnicate: 1\n",
		"engine kind":        strings.Replace(engYAML, "kind: fake", "kind: vllm", 1),
		"fake without mib":   strings.Replace(engYAML, "mib: 2048, ", "", 1),
		"no state file":      strings.Replace(engYAML, "state_file: /run/devai/control.json\n", "", 1),
		"actions elsewhere":  strings.Replace(engYAML, "fake: {kind: fake, mib: 2048, load_delay_s: 0, script: /usr/local/lib/devai/vram-ballast.py, log_dir: /var/log/devai-fake}", "actions: {kind: actions, registry: /r.yaml, repo: /repo}", 1),
		"bad gpu":            engYAML + "gpu: maybe\n",
		"no bus user":        strings.Replace(engYAML, "user: control-engines, ", "", 1),
		"bad container name": strings.Replace(engYAML, "container: engines", "container: Engines!", 1),
	}
	for name, y := range cases {
		if _, err := ParseConfig([]byte(y)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	dir := t.TempDir()
	p := filepath.Join(dir, "control.yaml")
	if err := os.WriteFile(p, []byte(engYAML), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadConfig(p, "operator"); err == nil {
		t.Error("--container operator accepted for a file naming engines")
	}
	if _, err := LoadConfig(p, "engines"); err != nil {
		t.Errorf("LoadConfig: %v", err)
	}
}

func TestProgramKindVerbs(t *testing.T) {
	srv := bustest.Start(t)
	dir := t.TempDir()
	cfg := &Config{
		Container: "operator", Bus: srv.Config("control-operator"),
		Supervisor: SupervisorConfig{ProgramDir: filepath.Join(dir, "supervisor")},
		StateFile:  filepath.Join(dir, "control.json"), LeaseS: 2, HeartbeatS: 1, PollS: 0.2, RunKeepS: 600,
		GPU: "none", ResultsDir: filepath.Join(dir, "results"), MaxContext: 262144,
		Services: map[string]ServiceConfig{"apache": {Kind: KindProgram, Program: "apache"}},
	}
	mem := &fakeMemory{}
	sup := newFakeSupervisord(t, cfg.Supervisor.ProgramDir, mem)
	sup.addProgram("apache", "RUNNING")
	r := startRig(t, srv, cfg, sup, mem)
	if st := r.state("apache"); st.Status != bus.StatusServing {
		t.Errorf("running program: %+v", st)
	}
	if rep := r.ask("operator", "apache", "stop", bus.Request{}, 5*time.Second); !rep.OK || rep.State.Status != bus.StatusStopped {
		t.Errorf("stop: %+v", rep)
	}
	if sup.state("apache") != "STOPPED" {
		t.Errorf("supervisord state %q", sup.state("apache"))
	}
	if rep := r.ask("operator", "apache", "start", bus.Request{}, 5*time.Second); !rep.OK || rep.State.Status != bus.StatusServing {
		t.Errorf("start: %+v", rep)
	}
	if rep := r.ask("operator", "apache", "restart", bus.Request{}, 5*time.Second); !rep.OK || rep.State.Status != bus.StatusServing {
		t.Errorf("restart: %+v", rep)
	}
	if rep := r.ask("operator", "apache", "reload", bus.Request{}, 5*time.Second); !rep.OK {
		t.Errorf("reload: %+v", rep)
	}
	if rep := r.ask("operator", "apache", "load", bus.Request{}, 5*time.Second); rep.OK || rep.Error.Code != bus.CodeBadRequest {
		t.Errorf("load on a program: %+v", rep)
	}
	_ = context.Background
}

func TestSupervisorFaultsAreTyped(t *testing.T) {
	mem := &fakeMemory{}
	sup := newFakeSupervisord(t, t.TempDir(), mem)
	s := NewSupervisor(sup.socket)
	_, err := s.GetProcessInfo(context.Background(), "nothing")
	if !IsFault(err, FaultBadName) {
		t.Errorf("err = %v, want BAD_NAME", err)
	}
	sup.addProgram("p", "RUNNING")
	if err := s.StartProcess(context.Background(), "p", true); !IsFault(err, FaultAlreadyStarted) {
		t.Errorf("err = %v, want ALREADY_STARTED", err)
	}
	info, err := s.GetProcessInfo(context.Background(), "p")
	if err != nil || info.StateName != "RUNNING" || !info.Running() {
		t.Errorf("info %+v %v", info, err)
	}
}
