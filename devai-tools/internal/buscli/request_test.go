package buscli

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/nats-io/nats.go"

	"github.com/sparavec/gpu-arbiter/pkg/bus"
	"github.com/sparavec/gpu-arbiter/pkg/bus/bustest"
)

func TestParseArgsTakesFlagsAnywhere(t *testing.T) {
	a, err := ParseArgs([]string{"devai.engines.vllm.load", "--job", "7", `{"x":1}`, "--timeout=900s"}, []string{"job", "timeout"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Pos) != 2 || a.Vals["job"] != "7" || a.Vals["timeout"] != "900s" {
		t.Errorf("parsed %+v", a)
	}
	if _, err := ParseArgs([]string{"--bogus"}, nil, nil); err == nil {
		t.Error("an unknown flag was accepted")
	}
	if _, err := ParseArgs([]string{"--job"}, []string{"job"}, nil); err == nil {
		t.Error("a flag without its value was accepted")
	}
}

// devai-bus request refuses a body that carries job or v (C3).
func TestBuildRequestAddsVAndJobAndRefusesThem(t *testing.T) {
	b, err := BuildRequest(`{"timeout_s":30}`, 42)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	_ = json.Unmarshal(b, &m)
	if m["v"] != float64(1) || m["job"] != float64(42) || m["timeout_s"] != float64(30) {
		t.Errorf("built %s", b)
	}
	for _, body := range []string{`{"job":7}`, `{"v":1}`, `[]`, `not json`, `{"timeout":30}`} {
		if _, err := BuildRequest(body, 7); err == nil {
			t.Errorf("body %s accepted", body)
		}
	}
}

func TestRequestOnTheWireCarriesVAndJob(t *testing.T) {
	srv := bustest.Start(t)
	ctl, err := bus.Connect(srv.Config("control-engines"), "test")
	if err != nil {
		t.Fatal(err)
	}
	defer ctl.Close()
	got := make(chan []byte, 1)
	_, _ = ctl.Subscribe("devai.engines.fake.>", func(m *nats.Msg) {
		got <- m.Data
		b, _ := bus.Marshal(bus.ErrReply(bus.CodeBusy, "busy", ""))
		_ = m.Respond(b)
	})
	_ = ctl.Flush()
	var out, errb bytes.Buffer
	code := Request(context.Background(), RequestOptions{Bus: srv.Config("jobs"), Subject: "devai.engines.fake.unload",
		Body: `{"force":true}`, Job: 1234, Timeout: 3 * time.Second, Stdout: &out, Stderr: &errb})
	if code != ExitFailed {
		t.Errorf("exit %d for a refusal, want 1 (%s)", code, errb.String())
	}
	var wire map[string]any
	_ = json.Unmarshal(<-got, &wire)
	if wire["v"] != float64(1) || wire["job"] != float64(1234) || wire["force"] != true {
		t.Errorf("on the wire: %v", wire)
	}
	if !strings.Contains(out.String(), `"code":"busy"`) {
		t.Errorf("the reply was not printed: %s", out.String())
	}
}

func TestRequestTimesOutWithOne(t *testing.T) {
	srv := bustest.Start(t)
	ctl, _ := bus.Connect(srv.Config("control-engines"), "test")
	defer ctl.Close()
	_, _ = ctl.Subscribe("devai.engines.fake.>", func(*nats.Msg) {}) // never answers
	_ = ctl.Flush()
	start := time.Now()
	code := Request(context.Background(), RequestOptions{Bus: srv.Config("jobs"), Subject: "devai.engines.fake.state",
		Body: `{}`, Job: 1, Timeout: time.Second})
	if code != ExitFailed {
		t.Errorf("exit %d, want 1", code)
	}
	if d := time.Since(start); d > 3*time.Second {
		t.Errorf("took %s", d)
	}
}

func TestRequestWithNoBusExitsTwo(t *testing.T) {
	srv := bustest.Start(t)
	cfg := srv.Config("jobs")
	srv.Shutdown()
	code := Request(context.Background(), RequestOptions{Bus: cfg, Subject: "devai.engines.fake.state",
		Body: `{}`, Job: 1, Timeout: time.Second})
	if code != ExitUnreachable {
		t.Errorf("exit %d, want 2", code)
	}
}

func TestLogWritesOneLinePerEventWithItsReceiveTime(t *testing.T) {
	srv := bustest.Start(t)
	dir := t.TempDir()
	ctx, cancel := context.WithCancel(context.Background())
	ready := make(chan struct{})
	done := make(chan int, 1)
	day := time.Date(2026, 10, 7, 12, 0, 0, 0, time.UTC)
	go func() {
		done <- Log(ctx, LogOptions{Bus: srv.Config("buslog"), Dir: dir, Ready: ready, Now: func() time.Time { return day }})
	}()
	<-ready
	pub, _ := bus.Connect(srv.Config("control-engines"), "test")
	defer pub.Close()
	_ = pub.Publish("devai.events.engines.fake", []byte(`{"status":"serving"}`))
	_ = pub.Flush()
	jobs, _ := bus.Connect(srv.Config("jobs"), "test")
	defer jobs.Close()
	_ = jobs.Publish("devai.events.slurm.guard", []byte(`{"verdict":"idle"}`))
	_ = jobs.Flush()
	file := filepath.Join(dir, "2026-10-07.jsonl")
	deadline := time.Now().Add(3 * time.Second)
	var lines []string
	for time.Now().Before(deadline) {
		b, _ := os.ReadFile(file)
		lines = strings.Split(strings.TrimSpace(string(b)), "\n")
		if len(lines) == 2 && lines[1] != "" {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	cancel()
	<-done
	if len(lines) != 2 {
		t.Fatalf("log lines: %q", lines)
	}
	var l LogLine
	if err := json.Unmarshal([]byte(lines[1]), &l); err != nil {
		t.Fatal(err)
	}
	if l.Subject != "devai.events.slurm.guard" || l.Received != bus.Timestamp(day) || string(l.Event) != `{"verdict":"idle"}` {
		t.Errorf("line %+v", l)
	}
}

func TestPublishRefusedByPermissionsFails(t *testing.T) {
	srv := bustest.Start(t)
	var errb bytes.Buffer
	code := Publish(context.Background(), PublishOptions{Bus: srv.Config("jobs"), Subject: "devai.events.engines.fake",
		Body: `{}`, Stderr: &errb})
	if code != ExitFailed {
		t.Errorf("exit %d, want 1 (%s)", code, errb.String())
	}
	code = Publish(context.Background(), PublishOptions{Bus: srv.Config("jobs"), Subject: "devai.events.slurm.guard", Body: `{}`})
	if code != ExitOK {
		t.Errorf("guard event: exit %d", code)
	}
}
