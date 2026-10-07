// Package control is devai-control, the control service in devai-engines,
// devai-laya-trainer and devai-operator (docs/plans/platform-rebuild.md C1,
// C7, C8, I33; docs/slurm.md Sec. 4). It takes commands from the bus,
// drives its container's supervisord, and publishes each service's state.
//
// Phase 5 implements the kinds `program` (a supervisord program), `fake`
// (Phase 5's test engine: scripts/vram-ballast.py, as vllm) and `actions`
// (the operator's registry, C8). The engine and trainer kinds are Phase 8's.
package control

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"sort"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/sparavec/gpu-arbiter/pkg/bus"
)

// Service kinds.
const (
	KindProgram = "program"
	KindFake    = "fake"
	KindActions = "actions"
)

// Config is /etc/devai/control.yaml (C7).
type Config struct {
	Container  string                   `yaml:"container"`
	Bus        bus.Config               `yaml:"bus"`
	Supervisor SupervisorConfig         `yaml:"supervisor"`
	StateFile  string                   `yaml:"state_file"`
	LeaseS     float64                  `yaml:"lease_timeout_s"`
	Services   map[string]ServiceConfig `yaml:"services"`

	// Extensions with defaults: the heartbeat (C1: 30 s), the liveness poll,
	// how long an ended run is kept (C1: 10 minutes), the GPU memory reader.
	HeartbeatS float64 `yaml:"heartbeat_s"`
	PollS      float64 `yaml:"poll_s"`
	RunKeepS   float64 `yaml:"run_keep_s"`
	// GPU: "auto" (nvidia-smi when it is on PATH; C1 "Memory readings") or
	// "none" (every memory check skipped).
	GPU string `yaml:"gpu"`
	// ResultsDir is where action runs write action.log and result.json (C4).
	ResultsDir string `yaml:"results_dir"`
	// MaxContext bounds a load's ctx (C2: 1024 to MAX_CONTEXT_LEN).
	MaxContext int `yaml:"max_context"`
}

// SupervisorConfig names supervisord's XML-RPC socket and the directory
// devai-control writes its program files into (supervisord includes it).
type SupervisorConfig struct {
	Socket     string `yaml:"socket"`
	ProgramDir string `yaml:"program_dir"`
}

// ServiceConfig is one entry of services:. Fields not used by a kind must
// stay unset.
type ServiceConfig struct {
	Kind    string `yaml:"kind"`
	Program string `yaml:"program"`
	Port    int    `yaml:"port"`
	Log     string `yaml:"log"`
	LogDir  string `yaml:"log_dir"`

	// fake: the VRAM ballast as the engine. `load` holds MiB and is
	// "serving" once it prints `ready`, after LoadDelayS seconds.
	MiB        int     `yaml:"mib"`
	LoadDelayS float64 `yaml:"load_delay_s"`
	Script     string  `yaml:"script"`
	Python     string  `yaml:"python"`

	// actions (C8).
	Registry string   `yaml:"registry"`
	Repo     string   `yaml:"repo"`
	Podman   string   `yaml:"podman"`
	OpsEnv   []string `yaml:"ops_env"`
}

// Durations of the configuration.
func (c *Config) Lease() time.Duration     { return secs(c.LeaseS) }
func (c *Config) Heartbeat() time.Duration { return secs(c.HeartbeatS) }
func (c *Config) Poll() time.Duration      { return secs(c.PollS) }
func (c *Config) RunKeep() time.Duration   { return secs(c.RunKeepS) }

func secs(s float64) time.Duration { return time.Duration(s * float64(time.Second)) }

// LoadConfig reads and checks control.yaml strictly: an unknown key is an
// error, and --container must equal the file's container (C7).
func LoadConfig(path, container string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	c, err := ParseConfig(data)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if container != c.Container {
		return nil, fmt.Errorf("--container %q does not equal %s's container %q", container, path, c.Container)
	}
	return c, nil
}

// ParseConfig decodes and validates control.yaml and fills the defaults.
func ParseConfig(data []byte) (*Config, error) {
	var c Config
	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)
	if err := dec.Decode(&c); err != nil {
		return nil, err
	}
	if c.LeaseS == 0 {
		c.LeaseS = 60
	}
	if c.HeartbeatS == 0 {
		c.HeartbeatS = 30
	}
	if c.PollS == 0 {
		c.PollS = 2
	}
	if c.RunKeepS == 0 {
		c.RunKeepS = 600
	}
	if c.GPU == "" {
		c.GPU = "auto"
	}
	if c.ResultsDir == "" {
		c.ResultsDir = "/var/cache/devai/jobs/results"
	}
	if c.MaxContext == 0 {
		c.MaxContext = 262144
	}
	return &c, c.validate()
}

func (c *Config) validate() error {
	var errs []error
	if !bus.ValidName(c.Container) {
		errs = append(errs, fmt.Errorf("container %q is not a valid name", c.Container))
	}
	if err := c.Bus.Validate(); err != nil {
		errs = append(errs, fmt.Errorf("bus: %w", err))
	}
	if c.StateFile == "" {
		errs = append(errs, errors.New("state_file is required"))
	}
	if c.LeaseS < 0 || c.HeartbeatS <= 0 || c.PollS <= 0 || c.RunKeepS < 0 {
		errs = append(errs, errors.New("lease_timeout_s, heartbeat_s, poll_s and run_keep_s must be positive"))
	}
	if c.GPU != "auto" && c.GPU != "none" {
		errs = append(errs, fmt.Errorf("gpu %q: want auto or none", c.GPU))
	}
	if len(c.Services) == 0 {
		errs = append(errs, errors.New("no services"))
	}
	names := make([]string, 0, len(c.Services))
	for n := range c.Services {
		names = append(names, n)
	}
	sort.Strings(names)
	needSupervisor := false
	for _, n := range names {
		s := c.Services[n]
		if !bus.ValidName(n) {
			errs = append(errs, fmt.Errorf("service %q is not a valid name", n))
		}
		switch s.Kind {
		case KindProgram:
			needSupervisor = true
			if s.Program == "" {
				errs = append(errs, fmt.Errorf("service %s: a program service names its program", n))
			}
		case KindFake:
			needSupervisor = true
			if s.MiB <= 0 || s.Script == "" || s.LogDir == "" {
				errs = append(errs, fmt.Errorf("service %s: kind fake needs mib, script and log_dir", n))
			}
			if s.LoadDelayS < 0 {
				errs = append(errs, fmt.Errorf("service %s: load_delay_s must not be negative", n))
			}
		case KindActions:
			if s.Registry == "" || s.Repo == "" {
				errs = append(errs, fmt.Errorf("service %s: kind actions needs registry and repo", n))
			}
			if n != bus.ServiceActions || c.Container != bus.ContainerOperator {
				errs = append(errs, fmt.Errorf("service %s: the actions kind is devai-operator's service `actions` (C1)", n))
			}
		default:
			errs = append(errs, fmt.Errorf("service %s: kind %q is not one of program, fake, actions (engines and the trainer come in Phase 8)", n, s.Kind))
		}
	}
	if needSupervisor && (c.Supervisor.Socket == "" || c.Supervisor.ProgramDir == "") {
		errs = append(errs, errors.New("supervisor.socket and supervisor.program_dir are required"))
	}
	return errors.Join(errs...)
}
