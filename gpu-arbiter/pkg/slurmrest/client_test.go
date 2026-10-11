package slurmrest

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

var testKey = []byte("0123456789abcdef0123456789abcdef")

type recorded struct {
	Method, Path, Template string
	Query                  map[string][]string
	Body                   any
}

// fakeRest is a slurmrestd stand-in that checks EVERY request against the
// stored v0.0.42 spec and answers with canned bodies, themselves checked
// against the spec's response schema.
type fakeRest struct {
	t       *testing.T
	spec    *openapi
	mu      sync.Mutex
	reqs    []recorded
	answers map[string]string // "METHOD template" -> body
	status  int
}

func newFake(t *testing.T) (*fakeRest, *Client) {
	f := &fakeRest{t: t, spec: loadSpec(t), answers: map[string]string{}, status: 200}
	srv := httptest.NewServer(f)
	t.Cleanup(srv.Close)
	c := New(srv.URL, "devai", testKey)
	return f, c
}

func (f *fakeRest) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	t := f.t
	tmpl, ok := f.spec.match(r.URL.Path)
	if !ok {
		t.Errorf("request to %s, which is not in the spec", r.URL.Path)
		http.Error(w, "no such path", 404)
		return
	}
	op := f.spec.op(t, r.Method, tmpl)
	declared := map[string]bool{}
	for _, p := range op.Parameters {
		if p.In == "query" {
			declared[p.Name] = true
		}
		if p.In == "query" && p.Name != "" {
			for _, v := range r.URL.Query()[p.Name] {
				var vv any = v
				for _, e := range f.spec.validate(r.Method+" "+tmpl+"?"+p.Name, p.Schema, vv) {
					t.Error(e)
				}
			}
		}
	}
	for k := range r.URL.Query() {
		if !declared[k] {
			t.Errorf("%s %s: query parameter %q is not in the spec", r.Method, tmpl, k)
		}
	}
	if r.Header.Get("X-SLURM-USER-NAME") != "devai" {
		t.Errorf("X-SLURM-USER-NAME = %q", r.Header.Get("X-SLURM-USER-NAME"))
	}
	claims, err := VerifyJWT(testKey, r.Header.Get("X-SLURM-USER-TOKEN"), time.Now())
	if err != nil || claims.Sun != "devai" || claims.Exp-claims.Iat != 300 {
		t.Errorf("token: %+v %v", claims, err)
	}
	rec := recorded{Method: r.Method, Path: r.URL.Path, Template: tmpl, Query: r.URL.Query()}
	body, _ := io.ReadAll(r.Body)
	if len(body) > 0 {
		if op.RequestBody == nil {
			t.Errorf("%s %s sent a body the spec does not take", r.Method, tmpl)
		} else {
			v := decodeNumbers(t, body)
			rec.Body = v
			for _, e := range f.spec.validate(r.Method+" "+tmpl, op.RequestBody.Content["application/json"].Schema, v) {
				t.Error(e)
			}
		}
	}
	f.mu.Lock()
	f.reqs = append(f.reqs, rec)
	ans := f.answers[r.Method+" "+tmpl]
	status := f.status
	f.mu.Unlock()
	if ans == "" {
		ans = `{"errors":[],"warnings":[]}`
	}
	resp := op.Responses["200"]
	for _, e := range f.spec.validate("response of "+r.Method+" "+tmpl, resp.Content["application/json"].Schema, decodeNumbers(t, []byte(ans))) {
		t.Error(e)
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = io.WriteString(w, ans)
}

func (f *fakeRest) last(t *testing.T) recorded {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.reqs) == 0 {
		t.Fatal("no request")
	}
	return f.reqs[len(f.reqs)-1]
}

func TestJWTRoundTrip(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	tok, err := SignJWT(testKey, "devai", now, TokenTTL)
	if err != nil {
		t.Fatal(err)
	}
	c, err := VerifyJWT(testKey, tok, now.Add(time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	if c.Sun != "devai" || c.Iat != now.Unix() || c.Exp != now.Unix()+300 {
		t.Errorf("claims = %+v", c)
	}
	if _, err := VerifyJWT([]byte("another key of thirty-two bytes!"), tok, now); err == nil {
		t.Error("a token verified with the wrong key")
	}
	if _, err := VerifyJWT(testKey, tok, now.Add(301*time.Second)); err == nil {
		t.Error("an expired token verified")
	}
	if _, err := SignJWT(nil, "devai", now, TokenTTL); err == nil {
		t.Error("signed with an empty key")
	}
}

func TestSubmitEngineJobMatchesC3(t *testing.T) {
	f, c := newFake(t)
	f.answers["POST /slurm/v0.0.42/job/submit"] = `{"job_id":42,"step_id":"batch","job_submit_user_msg":"","errors":[],"warnings":[]}`
	id, err := c.Submit(context.Background(), JobDesc{
		Script:                  "#!/bin/bash\nexec /usr/local/lib/devai/jobs/engine.sh\n",
		Name:                    "engine-fake",
		Licenses:                "engine",
		Requeue:                 false,
		CurrentWorkingDirectory: "/var/cache/devai/jobs/results",
		StandardOutput:          "/var/cache/devai/jobs/results/%j/job.log",
		StandardError:           "/var/cache/devai/jobs/results/%j/job.log",
		Environment:             []string{"PATH=/usr/local/bin:/usr/bin:/bin", "DEVAI_BACKEND=fake"},
		Comment:                 `{"v":1,"kind":"engine"}`,
	})
	if err != nil || id != 42 {
		t.Fatalf("Submit = %d, %v", id, err)
	}
	body := f.last(t).Body.(map[string]any)
	job := body["job"].(map[string]any)
	if job["requeue"] != false {
		t.Errorf("requeue = %v, want false sent explicitly", job["requeue"])
	}
	if _, ok := body["script"]; ok {
		t.Error("script sent at the top level (deprecated in v0.0.42)")
	}
}

func TestSubmitWithTimeLimitAndBatchWarning(t *testing.T) {
	f, c := newFake(t)
	f.answers["POST /slurm/v0.0.42/job/submit"] = `{"job_id":7}`
	_, err := c.Submit(context.Background(), JobDesc{
		Script: "#!/bin/bash\nexec /usr/local/lib/devai/jobs/bench.sh\n", Name: "bench-gsm8k",
		Licenses: "bench", TimeLimit: Num(32),
		KillWarningSignal: "INT", KillWarningDelay: Num(120), KillWarningFlags: []string{"BATCH_JOB"},
		Environment: []string{"PATH=/usr/bin"},
	})
	if err != nil {
		t.Fatal(err)
	}
	job := f.last(t).Body.(map[string]any)["job"].(map[string]any)
	if job["kill_warning_signal"] != "INT" {
		t.Errorf("kill_warning_signal = %v", job["kill_warning_signal"])
	}
}

func TestCallsUseTheSpecsPathsAndParameters(t *testing.T) {
	f, c := newFake(t)
	ctx := context.Background()
	f.answers["GET /slurm/v0.0.42/job/{job_id}"] = `{"jobs":[{"job_id":5,"name":"engine-fake","job_state":["RUNNING"],"state_reason":"None","comment":"{\"v\":1}","licenses":"engine","hold":false,"exit_code":{"status":["SUCCESS"],"return_code":{"set":true,"infinite":false,"number":0},"signal":{"id":{"set":false,"infinite":false,"number":0},"name":""}},"start_time":{"set":true,"infinite":false,"number":1}}],"last_backfill":{"set":true,"infinite":false,"number":0},"last_update":{"set":true,"infinite":false,"number":0}}`
	j, err := c.Job(ctx, 5)
	if err != nil || j.JobID != 5 || !j.HasState("RUNNING") || j.Licenses != "engine" {
		t.Fatalf("Job = %+v, %v", j, err)
	}

	f.answers["GET /slurm/v0.0.42/jobs/"] = `{"jobs":[{"job_id":5,"job_state":["PENDING"],"state_reason":"Licenses"}],"last_backfill":{"set":true,"infinite":false,"number":0},"last_update":{"set":true,"infinite":false,"number":0}}`
	js, err := c.Jobs(ctx)
	if err != nil || len(js) != 1 || js[0].StateReason != "Licenses" {
		t.Fatalf("Jobs = %+v, %v", js, err)
	}

	if err := c.Signal(ctx, 5, "SIGUSR1", true); err != nil {
		t.Fatal(err)
	}
	r := f.last(t)
	if r.Method != "DELETE" || r.Query["signal"][0] != "SIGUSR1" || r.Query["flags"][0] != "BATCH_JOB" {
		t.Errorf("Signal sent %+v", r)
	}
	if err := c.Cancel(ctx, 5); err != nil {
		t.Fatal(err)
	}
	if r := f.last(t); r.Method != "DELETE" || len(r.Query) != 0 {
		t.Errorf("Cancel sent %+v", r)
	}
	if err := c.Hold(ctx, 5); err != nil {
		t.Fatal(err)
	}
	if b := f.last(t).Body.(map[string]any); b["hold"] != true {
		t.Errorf("Hold sent %v", b)
	}
	if err := c.Release(ctx, 5); err != nil {
		t.Fatal(err)
	}
	if b := f.last(t).Body.(map[string]any); b["hold"] != false {
		t.Errorf("Release sent %v", b)
	}

	f.answers["GET /slurm/v0.0.42/node/{node_name}"] = `{"nodes":[{"name":"devai","state":["IDLE","DRAIN"],"reason":"devai-guard: busy","cpus":24,"real_memory":64000}],"last_update":{"set":true,"infinite":false,"number":0}}`
	n, err := c.Node(ctx, "devai")
	if err != nil || !n.HasState("DRAIN") || n.Reason != "devai-guard: busy" {
		t.Fatalf("Node = %+v, %v", n, err)
	}
	if err := c.ResumeNode(ctx, "devai"); err != nil {
		t.Fatal(err)
	}
	if b := f.last(t).Body.(map[string]any); b["state"].([]any)[0] != "RESUME" {
		t.Errorf("ResumeNode sent %v", b)
	}

	f.answers["GET /slurmdb/v0.0.42/jobs/"] = `{"jobs":[{"job_id":5,"name":"engine-fake","user":"devai","comment":{"administrator":"gpu peak=1 mean=1 util=0 energy=0","job":"{\"v\":1,\"kind\":\"engine\"}","system":""},"state":{"current":["COMPLETED"],"reason":"None"},"time":{"start":1,"end":2,"elapsed":1,"submission":1,"eligible":1}}]}`
	db, err := c.DBJobs(ctx, DBQuery{Users: "devai", StartTime: "now-1days"})
	if err != nil || len(db) != 1 || db[0].Comment.Job != `{"v":1,"kind":"engine"}` {
		t.Fatalf("DBJobs = %+v, %v", db, err)
	}
	if q := f.last(t).Query; q["users"][0] != "devai" || q["start_time"][0] != "now-1days" {
		t.Errorf("DBJobs query = %v", q)
	}
	f.answers["GET /slurmdb/v0.0.42/job/{job_id}"] = f.answers["GET /slurmdb/v0.0.42/jobs/"]
	if j, err := c.DBJob(ctx, 5); err != nil || j.State.Current[0] != "COMPLETED" {
		t.Fatalf("DBJob = %+v, %v", j, err)
	}
}

func TestErrorsAreReturned(t *testing.T) {
	f, c := newFake(t)
	f.answers["GET /slurm/v0.0.42/job/{job_id}"] = `{"jobs":[],"last_backfill":{"set":true,"infinite":false,"number":0},"last_update":{"set":true,"infinite":false,"number":0},"errors":[{"description":"","error_number":2017,"error":"Invalid job id specified","source":"_handle_job_get"}]}`
	_, err := c.Job(context.Background(), 99)
	if err == nil || !IsInvalidJob(err) {
		t.Fatalf("err = %v, want an invalid-job error", err)
	}
	f.status = 500
	f.answers["POST /slurm/v0.0.42/job/submit"] = `{"errors":[{"error":"Access/permission denied","error_number":2010,"description":"devai jobs run as devai only","source":"slurm_submit_batch_job()"}]}`
	_, err = c.Submit(context.Background(), JobDesc{Script: "#!/bin/bash\n", Environment: []string{"PATH=/bin"}})
	if err == nil || !strings.Contains(err.Error(), "devai jobs run as devai only") {
		t.Fatalf("err = %v", err)
	}
	var e *Error
	if !asError(err, &e) || e.Status != 500 {
		t.Errorf("not an *Error with the status: %v", err)
	}
}

func asError(err error, target **Error) bool { return errors.As(err, target) }

func TestFromEnv(t *testing.T) {
	dir := t.TempDir()
	key := dir + "/jwt_hs256.key"
	if err := writeFile(key, testKey); err != nil {
		t.Fatal(err)
	}
	t.Setenv(EnvURL, "http://127.0.0.1:6820/")
	t.Setenv(EnvKeyFile, key)
	t.Setenv(EnvUser, "")
	c, err := FromEnv()
	if err != nil {
		t.Fatal(err)
	}
	if c.User != "devai" || c.BaseURL != "http://127.0.0.1:6820" {
		t.Errorf("client = %+v", c)
	}
	t.Setenv(EnvKeyFile, "")
	if _, err := FromEnv(); err == nil {
		t.Error("FromEnv without a key file succeeded")
	}
}

func writeFile(p string, b []byte) error { return os.WriteFile(p, b, 0o600) }
