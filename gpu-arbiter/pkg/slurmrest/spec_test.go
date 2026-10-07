package slurmrest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
)

// The spec Debian's slurmrestd 24.11.5 prints (platform-rebuild Phase 5,
// step 2): `slurmrestd -d v0.0.42 -s slurmctld,slurmdbd
// --generate-openapi-spec`, stored in the repository.
func specPath() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..", "..", "deploy", "slurm", "openapi-v0.0.42.json")
}

type openapi struct {
	Paths      map[string]map[string]json.RawMessage `json:"paths"`
	Components struct {
		Schemas map[string]json.RawMessage `json:"schemas"`
	} `json:"components"`
}

type operation struct {
	Parameters []struct {
		In     string          `json:"in"`
		Name   string          `json:"name"`
		Schema json.RawMessage `json:"schema"`
	} `json:"parameters"`
	RequestBody *struct {
		Content map[string]struct {
			Schema json.RawMessage `json:"schema"`
		} `json:"content"`
	} `json:"requestBody"`
	Responses map[string]struct {
		Content map[string]struct {
			Schema json.RawMessage `json:"schema"`
		} `json:"content"`
	} `json:"responses"`
}

type schema struct {
	Ref        string                     `json:"$ref"`
	Type       string                     `json:"type"`
	Properties map[string]json.RawMessage `json:"properties"`
	Items      json.RawMessage            `json:"items"`
	Enum       []string                   `json:"enum"`
}

func loadSpec(t *testing.T) *openapi {
	t.Helper()
	b, err := os.ReadFile(specPath())
	if err != nil {
		t.Fatalf("read the stored spec: %v", err)
	}
	var s openapi
	if err := json.Unmarshal(b, &s); err != nil {
		t.Fatalf("parse the stored spec: %v", err)
	}
	return &s
}

// match finds the spec path template for a concrete path; a literal
// segment beats a {parameter} ("/job/submit" is not "/job/{job_id}").
func (s *openapi) match(path string) (string, bool) {
	want := strings.Split(path, "/")
	best, bestLit := "", -1
	for tmpl := range s.Paths {
		got := strings.Split(tmpl, "/")
		if len(got) != len(want) {
			continue
		}
		ok, lit := true, 0
		for i := range got {
			if strings.HasPrefix(got[i], "{") && strings.HasSuffix(got[i], "}") {
				if want[i] == "" {
					ok = false
				}
				continue
			}
			if got[i] != want[i] {
				ok = false
				break
			}
			lit++
		}
		if ok && lit > bestLit {
			best, bestLit = tmpl, lit
		}
	}
	return best, bestLit >= 0
}

func (s *openapi) op(t *testing.T, method, tmpl string) operation {
	t.Helper()
	raw, ok := s.Paths[tmpl][strings.ToLower(method)]
	if !ok {
		t.Fatalf("spec has no %s %s", method, tmpl)
	}
	var o operation
	if err := json.Unmarshal(raw, &o); err != nil {
		t.Fatal(err)
	}
	return o
}

func (s *openapi) resolve(raw json.RawMessage) schema {
	var sc schema
	_ = json.Unmarshal(raw, &sc)
	for sc.Ref != "" {
		name := strings.TrimPrefix(sc.Ref, "#/components/schemas/")
		next := schema{}
		_ = json.Unmarshal(s.Components.Schemas[name], &next)
		sc = next
	}
	return sc
}

// validate checks value against a schema: every object key must be a
// declared property, every enum value a declared one, and types must agree.
func (s *openapi) validate(where string, raw json.RawMessage, value any) []string {
	sc := s.resolve(raw)
	var errs []string
	switch v := value.(type) {
	case map[string]any:
		if sc.Type != "" && sc.Type != "object" {
			return []string{fmt.Sprintf("%s: object where the spec has %s", where, sc.Type)}
		}
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			p, ok := sc.Properties[k]
			if !ok {
				errs = append(errs, fmt.Sprintf("%s: field %q is not in the spec", where, k))
				continue
			}
			errs = append(errs, s.validate(where+"."+k, p, v[k])...)
		}
	case []any:
		if sc.Type != "array" {
			return []string{fmt.Sprintf("%s: array where the spec has %q", where, sc.Type)}
		}
		for i, item := range v {
			errs = append(errs, s.validate(fmt.Sprintf("%s[%d]", where, i), sc.Items, item)...)
		}
	case string:
		if sc.Type != "string" && !(sc.Type == "" && len(sc.Enum) > 0) {
			return []string{fmt.Sprintf("%s: string where the spec has %q", where, sc.Type)}
		}
		if len(sc.Enum) > 0 {
			found := false
			for _, e := range sc.Enum {
				if e == v {
					found = true
				}
			}
			if !found {
				errs = append(errs, fmt.Sprintf("%s: %q is not one of the spec's values", where, v))
			}
		}
	case json.Number:
		if sc.Type != "integer" && sc.Type != "number" {
			return []string{fmt.Sprintf("%s: number where the spec has %q", where, sc.Type)}
		}
	case bool:
		if sc.Type != "boolean" {
			return []string{fmt.Sprintf("%s: boolean where the spec has %q", where, sc.Type)}
		}
	case nil:
		errs = append(errs, where+": null")
	}
	return errs
}

func decodeNumbers(t *testing.T, b []byte) any {
	t.Helper()
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		t.Fatalf("decode %s: %v", b, err)
	}
	return v
}

func TestStoredSpecIsDebians2411(t *testing.T) {
	s := loadSpec(t)
	for _, p := range []string{
		"/slurm/v0.0.42/job/submit", "/slurm/v0.0.42/job/{job_id}", "/slurm/v0.0.42/jobs/",
		"/slurm/v0.0.42/node/{node_name}", "/slurmdb/v0.0.42/jobs/", "/slurmdb/v0.0.42/job/{job_id}",
	} {
		if _, ok := s.Paths[p]; !ok {
			t.Errorf("stored spec lacks %s", p)
		}
	}
}

// The field names C3 and C5 use, checked against the spec (Phase 5 step 2).
func TestContractFieldNamesExistInTheSpec(t *testing.T) {
	s := loadSpec(t)
	desc := s.resolve(json.RawMessage(`{"$ref":"#/components/schemas/v0.0.42_job_desc_msg"}`))
	for _, f := range []string{
		"script", "name", "comment", "licenses", "requeue", "time_limit", "environment",
		"standard_output", "standard_error", "current_working_directory",
		"kill_warning_signal", "kill_warning_delay", "kill_warning_flags", "hold",
	} {
		if _, ok := desc.Properties[f]; !ok {
			t.Errorf("v0.0.42_job_desc_msg has no %q", f)
		}
	}
	submit := s.resolve(json.RawMessage(`{"$ref":"#/components/schemas/v0.0.42_job_submit_req"}`))
	if _, ok := submit.Properties["job"]; !ok {
		t.Error("v0.0.42_job_submit_req has no `job`")
	}
	del := s.op(t, "DELETE", "/slurm/v0.0.42/job/{job_id}")
	names := map[string]bool{}
	for _, p := range del.Parameters {
		names[p.Name] = true
		if p.Name == "flags" && !strings.Contains(string(p.Schema), `"BATCH_JOB"`) {
			t.Error("DELETE job flags lacks BATCH_JOB")
		}
	}
	if !names["signal"] || !names["flags"] {
		t.Errorf("DELETE job parameters = %v, want signal and flags", names)
	}
	flags := s.resolve(desc.Properties["kill_warning_flags"])
	if !strings.Contains(string(flags.Items), `"BATCH_JOB"`) {
		t.Error("kill_warning_flags lacks BATCH_JOB")
	}
	states := s.resolve(json.RawMessage(`{"$ref":"#/components/schemas/v0.0.42_node_states"}`))
	if !strings.Contains(string(states.Items), `"RESUME"`) {
		t.Error("node states lack RESUME")
	}
}
