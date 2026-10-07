package jobs

import (
	"encoding/json"
	"strings"
	"testing"
)

func envMap(env []string) map[string]string {
	m := map[string]string{}
	for _, kv := range env {
		k, v, _ := strings.Cut(kv, "=")
		m[k] = v
	}
	return m
}

func TestBuildPerKindMatchesC3(t *testing.T) {
	cases := []struct {
		spec       Spec
		name, lic  string
		limit      int64 // 0: none
		warn       string
		delay      int64
		commentKey []string
		envKeys    []string
		holdKey    bool
	}{
		{Spec{Kind: KindEngine, Backend: "vllm", Load: json.RawMessage(`{"launch":{"backend":"vllm","model":"Qwen3.8-27B","ctx":131072,"mtp":{"n":1}},"timeout_s":600}`)},
			"engine-vllm", "engine", 0, "", 0, []string{"backend", "model", "ctx", "mtp"}, []string{"DEVAI_BACKEND", "DEVAI_LOAD"}, false},
		{Spec{Kind: KindTrainer, FTJob: "ftjob-1", TimeLimit: 15},
			"trainer", "engine", 15, "USR2", 30, []string{"ft_job"}, []string{"DEVAI_FT_JOB"}, false},
		{Spec{Kind: KindProbe, Backend: "ollama", Action: "probe-ollama-model", Params: json.RawMessage(`{"model":"m"}`)},
			"probe-ollama", "engine", 60, "", 0, []string{"action", "hold_key"}, []string{"DEVAI_ACTION", "DEVAI_PARAMS", "DEVAI_BACKEND"}, true},
		{Spec{Kind: KindBench, Action: "bench-task", Run: "r1", Task: "gsm8k", Backend: "vllm", Model: "m", Ctx: 8192, MTP: true},
			"bench-gsm8k", "bench", 32, "INT", 120, []string{"run", "task", "backend", "model", "ctx", "mtp", "hold_key"}, []string{"DEVAI_ACTION", "DEVAI_PARAMS"}, true},
		{Spec{Kind: KindAction, Action: "model-pull", Params: json.RawMessage(`{"name":"x"}`)},
			"action-model-pull", "", 0, "", 0, []string{"action", "gpu", "hold_key"}, []string{"DEVAI_ACTION", "DEVAI_PARAMS"}, true},
		{Spec{Kind: KindAction, Action: "quant-local", GPU: true, TimeLimit: 240},
			"action-quant-local", "engine", 240, "", 0, []string{"gpu"}, nil, true},
	}
	mtpFalse, err := Build(Spec{Kind: KindBench, Action: "bench-task", Run: "r", Task: "t", Backend: "ollama", Model: "m", Ctx: 4096})
	if err != nil {
		t.Fatal(err)
	}
	if c := ParseComment(mtpFalse.Desc.Comment); c["mtp"] != false || c["ctx"] != float64(4096) {
		t.Errorf("a bench comment always carries ctx and mtp: %s", mtpFalse.Desc.Comment)
	}
	for _, c := range cases {
		b, err := Build(c.spec)
		if err != nil {
			t.Errorf("%s: %v", c.spec.Kind, err)
			continue
		}
		d := b.Desc
		if d.Name != c.name || d.Licenses != c.lic || d.Requeue {
			t.Errorf("%s: name %q licenses %q requeue %v", c.spec.Kind, d.Name, d.Licenses, d.Requeue)
		}
		if d.Script != "#!/bin/bash\nexec /usr/local/lib/devai/jobs/"+c.spec.Kind+".sh\n" {
			t.Errorf("%s: script %q", c.spec.Kind, d.Script)
		}
		if c.limit == 0 && d.TimeLimit != nil || c.limit != 0 && (d.TimeLimit == nil || d.TimeLimit.Number != c.limit) {
			t.Errorf("%s: time limit %+v, want %d", c.spec.Kind, d.TimeLimit, c.limit)
		}
		if d.KillWarningSignal != c.warn || (c.warn != "" && (d.KillWarningDelay.Number != c.delay || d.KillWarningFlags[0] != "BATCH_JOB")) {
			t.Errorf("%s: warning %q %+v %v", c.spec.Kind, d.KillWarningSignal, d.KillWarningDelay, d.KillWarningFlags)
		}
		if d.StandardOutput != "/var/cache/devai/jobs/results/%j/job.log" || d.CurrentWorkingDirectory != "/var/cache/devai/jobs/results" {
			t.Errorf("%s: output %q cwd %q", c.spec.Kind, d.StandardOutput, d.CurrentWorkingDirectory)
		}
		env := envMap(d.Environment)
		for k, v := range map[string]string{"PATH": "/usr/local/bin:/usr/bin:/bin", "DEVAI_BUS_URL": "nats://127.0.0.1:4222",
			"DEVAI_BUS_USER": "jobs", "DEVAI_BUS_PASSWORD_FILE": "/run/devai/bus/jobs.pass"} {
			if env[k] != v {
				t.Errorf("%s: env %s=%q", c.spec.Kind, k, env[k])
			}
		}
		for _, k := range c.envKeys {
			if _, ok := env[k]; !ok {
				t.Errorf("%s: env lacks %s", c.spec.Kind, k)
			}
		}
		comment := ParseComment(d.Comment)
		if comment["v"] != float64(1) || comment["kind"] != c.spec.Kind {
			t.Errorf("%s: comment %s", c.spec.Kind, d.Comment)
		}
		for _, k := range c.commentKey {
			if _, ok := comment[k]; !ok {
				t.Errorf("%s: comment lacks %s: %s", c.spec.Kind, k, d.Comment)
			}
		}
		if hk, _ := comment["hold_key"].(string); c.holdKey != (len(hk) == 32) || hk != b.HoldKey {
			t.Errorf("%s: hold key %q", c.spec.Kind, hk)
		}
	}
}

func TestBuildRefusals(t *testing.T) {
	long := `{"model":"` + strings.Repeat("x", 2000) + `"}`
	cases := map[string]Spec{
		"load carries v":            {Kind: KindEngine, Backend: "vllm", Load: json.RawMessage(`{"v":1,"launch":{}}`)},
		"load carries job":          {Kind: KindEngine, Backend: "vllm", Load: json.RawMessage(`{"job":1,"launch":{}}`)},
		"load not an object":        {Kind: KindEngine, Backend: "vllm", Load: json.RawMessage(`[]`)},
		"bad backend":               {Kind: KindEngine, Backend: "V LLM", Load: json.RawMessage(`{}`)},
		"gpu action without limit":  {Kind: KindAction, Action: "x", GPU: true},
		"bad action id":             {Kind: KindAction, Action: "X Y"},
		"params not an object":      {Kind: KindAction, Action: "x", Params: json.RawMessage(`[1]`)},
		"bench without its launch":  {Kind: KindBench, Action: "bench-task", Task: "gsm8k"},
		"bench without its ctx":     {Kind: KindBench, Action: "bench-task", Task: "gsm8k", Backend: "vllm", Model: "m"},
		"unknown kind":              {Kind: "other"},
		"comment over 1 KiB":        {Kind: KindAction, Action: "x", Model: strings.Repeat("m", 1100)},
		"trainer without ft_job":    {Kind: KindTrainer},
		"bench with a bad task":     {Kind: KindBench, Action: "b", Task: "a b", Backend: "vllm", Model: "m"},
		"probe params over a limit": {Kind: KindProbe, Backend: "vllm", Action: "p", Params: json.RawMessage(long), Model: strings.Repeat("m", 1100)},
	}
	for name, s := range cases {
		if _, err := Build(s); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}
