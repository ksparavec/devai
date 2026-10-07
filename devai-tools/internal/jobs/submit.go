// Package jobs builds devai's Slurm jobs (platform-rebuild C3) and reads
// them back for devai-jobs. The REST calls are gpu-arbiter/pkg/slurmrest's,
// the one REST client (I2, C8).
package jobs

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"

	"github.com/sparavec/gpu-arbiter/pkg/slurmrest"
)

// Kinds (C3).
const (
	KindEngine  = "engine"
	KindTrainer = "trainer"
	KindProbe   = "probe"
	KindBench   = "bench"
	KindAction  = "action"
)

// Paths and bus settings every job carries (C3).
const (
	JobScriptDir = "/usr/local/lib/devai/jobs"
	ResultsDir   = "/var/cache/devai/jobs/results"
	JobBusURL    = "nats://127.0.0.1:4222"
	JobBusUser   = "jobs"
	JobBusPass   = "/run/devai/bus/jobs.pass"
	MaxComment   = 1024
)

var (
	backendRE = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{0,31}$`)
	actionRE  = regexp.MustCompile(`^[a-z0-9-]{1,64}$`)
	taskRE    = regexp.MustCompile(`^[A-Za-z0-9_.-]{1,64}$`)
)

// Spec describes one job to submit.
type Spec struct {
	Kind      string
	Backend   string          // engine, probe (and the comment of bench and some actions)
	Load      json.RawMessage // engine: the body of the load request (DEVAI_LOAD)
	FTJob     string          // trainer
	Action    string          // probe, bench, action
	Params    json.RawMessage // probe, bench, action: a JSON object
	GPU       bool            // action: a GPU action (C8)
	TimeLimit int             // minutes; 0 = the kind's default
	// Comment keys a bench run or an action entry's `comment` names (C3).
	Model string
	Ctx   int
	MTP   bool
	Run   string // bench
	Task  string // bench
	// Requester and Git go into every comment.
	Requester string
	Git       string
	Hold      bool
}

// Built is a ready job description and the hold key it carries.
type Built struct {
	Desc    slurmrest.JobDesc
	Comment map[string]any
	HoldKey string
}

// NewHoldKey is 32 random hex characters (C3).
func NewHoldKey() (string, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

func object(raw json.RawMessage, what string) (map[string]json.RawMessage, error) {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil || m == nil {
		return nil, fmt.Errorf("%s must be a JSON object", what)
	}
	return m, nil
}

// Build turns a Spec into the job of C3: script, name, licenses, time
// limit and warning signal per kind, requeue false, the results directory,
// the job environment and the JSON comment.
func Build(s Spec) (Built, error) {
	b := Built{Comment: map[string]any{"v": 1, "kind": s.Kind, "requester": s.Requester, "git": s.Git}}
	env := []string{
		"PATH=/usr/local/bin:/usr/bin:/bin",
		"DEVAI_BUS_URL=" + JobBusURL,
		"DEVAI_BUS_USER=" + JobBusUser,
		"DEVAI_BUS_PASSWORD_FILE=" + JobBusPass,
	}
	d := slurmrest.JobDesc{
		Script:                  "#!/bin/bash\nexec " + JobScriptDir + "/" + s.Kind + ".sh\n",
		Requeue:                 false,
		CurrentWorkingDirectory: ResultsDir,
		StandardOutput:          ResultsDir + "/%j/job.log",
		StandardError:           ResultsDir + "/%j/job.log",
		Hold:                    s.Hold,
	}
	needAction := func() error {
		if !actionRE.MatchString(s.Action) {
			return fmt.Errorf("--action %q: want [a-z0-9-]+", s.Action)
		}
		params := s.Params
		if len(params) == 0 {
			params = json.RawMessage(`{}`)
		}
		if _, err := object(params, "--params"); err != nil {
			return err
		}
		compact, _ := json.Marshal(json.RawMessage(params))
		env = append(env, "DEVAI_ACTION="+s.Action, "DEVAI_PARAMS="+string(compact))
		b.Comment["action"] = s.Action
		key, err := NewHoldKey()
		if err != nil {
			return err
		}
		b.HoldKey = key
		b.Comment["hold_key"] = key
		return nil
	}
	launchKeys := func() {
		if s.Backend != "" {
			b.Comment["backend"] = s.Backend
		}
		if s.Model != "" {
			b.Comment["model"] = s.Model
		}
		if s.Ctx > 0 {
			b.Comment["ctx"] = s.Ctx
		}
		if s.MTP {
			b.Comment["mtp"] = true
		}
	}
	switch s.Kind {
	case KindEngine:
		if !backendRE.MatchString(s.Backend) {
			return b, fmt.Errorf("--backend %q is not a backend name", s.Backend)
		}
		m, err := object(s.Load, "--load")
		if err != nil {
			return b, err
		}
		if _, ok := m["v"]; ok {
			return b, errors.New("--load must not carry v (devai-bus adds it)")
		}
		if _, ok := m["job"]; ok {
			return b, errors.New("--load must not carry job (devai-bus adds it)")
		}
		var launch struct {
			Model string          `json:"model"`
			Ctx   int             `json:"ctx"`
			MTP   json.RawMessage `json:"mtp"`
		}
		_ = json.Unmarshal(m["launch"], &launch)
		s.Model, s.Ctx = launch.Model, launch.Ctx
		s.MTP = len(launch.MTP) > 0 && string(launch.MTP) != "null"
		launchKeys()
		compact, _ := json.Marshal(json.RawMessage(s.Load))
		env = append(env, "DEVAI_BACKEND="+s.Backend, "DEVAI_LOAD="+string(compact))
		d.Name = "engine-" + s.Backend
		d.Licenses = "engine"
	case KindTrainer:
		if !regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`).MatchString(s.FTJob) {
			return b, fmt.Errorf("--ft-job %q is not a fine-tuning job id", s.FTJob)
		}
		env = append(env, "DEVAI_FT_JOB="+s.FTJob)
		b.Comment["ft_job"] = s.FTJob
		d.Name = "trainer"
		d.Licenses = "engine"
		if s.TimeLimit > 0 {
			d.TimeLimit = slurmrest.Num(int64(s.TimeLimit))
			d.KillWarningSignal, d.KillWarningDelay, d.KillWarningFlags = "USR2", slurmrest.Num(30), []string{"BATCH_JOB"}
		}
	case KindProbe:
		if !backendRE.MatchString(s.Backend) {
			return b, fmt.Errorf("--backend %q is not a backend name", s.Backend)
		}
		if err := needAction(); err != nil {
			return b, err
		}
		launchKeys()
		env = append(env, "DEVAI_BACKEND="+s.Backend)
		d.Name = "probe-" + s.Backend
		d.Licenses = "engine"
		d.TimeLimit = slurmrest.Num(int64(orDefault(s.TimeLimit, 60)))
	case KindBench:
		if err := needAction(); err != nil {
			return b, err
		}
		if !taskRE.MatchString(s.Task) {
			return b, fmt.Errorf("--task %q is not a task name", s.Task)
		}
		if !backendRE.MatchString(s.Backend) || s.Model == "" || s.Ctx <= 0 {
			return b, errors.New("a bench job names its launch: --backend, --model and --ctx (and --mtp)")
		}
		// C3: a bench comment always carries backend, model, ctx and mtp,
		// so its C5 hold is accepted and a held re-run names its launch.
		launchKeys()
		b.Comment["ctx"], b.Comment["mtp"] = s.Ctx, s.MTP
		b.Comment["run"], b.Comment["task"] = s.Run, s.Task
		d.Name = "bench-" + s.Task
		d.Licenses = "bench"
		d.TimeLimit = slurmrest.Num(int64(orDefault(s.TimeLimit, 32)))
		d.KillWarningSignal, d.KillWarningDelay, d.KillWarningFlags = "INT", slurmrest.Num(120), []string{"BATCH_JOB"}
	case KindAction:
		if err := needAction(); err != nil {
			return b, err
		}
		launchKeys()
		b.Comment["gpu"] = s.GPU
		d.Name = "action-" + s.Action
		if s.GPU {
			if s.TimeLimit <= 0 {
				return b, errors.New("a GPU action needs a time limit (--time-limit-min): the partition has no maximum")
			}
			d.Licenses = "engine"
		}
		if s.TimeLimit > 0 {
			d.TimeLimit = slurmrest.Num(int64(s.TimeLimit))
		}
	default:
		return b, fmt.Errorf("unknown job kind %q (engine, trainer, probe, bench, action)", s.Kind)
	}
	comment, err := json.Marshal(b.Comment)
	if err != nil {
		return b, err
	}
	if len(comment) >= MaxComment {
		return b, fmt.Errorf("the job comment is %d bytes; C3 keeps it under %d", len(comment), MaxComment)
	}
	d.Comment = string(comment)
	d.Environment = env
	b.Desc = d
	return b, nil
}

func orDefault(v, def int) int {
	if v > 0 {
		return v
	}
	return def
}

// ParseComment decodes a job's JSON comment; a comment that is not a JSON
// object yields an empty map.
func ParseComment(s string) map[string]any {
	var m map[string]any
	if err := json.Unmarshal([]byte(s), &m); err != nil || m == nil {
		return map[string]any{}
	}
	return m
}
