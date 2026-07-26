package main

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// nonFlusher is an http.ResponseWriter that deliberately does NOT
// implement http.Flusher. A comment frame that sits in a buffer defeats
// the entire purpose, so the keepalive must decline to arm rather than
// write frames nobody will see until the response ends.
type nonFlusher struct {
	h      http.Header
	status int
	n      int
}

func (n *nonFlusher) Header() http.Header {
	if n.h == nil {
		n.h = http.Header{}
	}
	return n.h
}
func (n *nonFlusher) Write(b []byte) (int, error) { n.n += len(b); return len(b), nil }
func (n *nonFlusher) WriteHeader(s int)           { n.status = s }

func TestWantsSSEKeepaliveGate(t *testing.T) {
	tests := []struct {
		name string
		path string
		body string
		want bool
	}{
		{"openai streaming", "/v1/chat/completions", `{"model":"m","stream":true}`, true},
		{"anthropic streaming", "/v1/messages", `{"model":"m","stream":true}`, true},
		{"completions streaming", "/v1/completions", `{"model":"m","stream":true}`, true},
		{"openai non-streaming", "/v1/chat/completions", `{"model":"m","stream":false}`, false},
		{"openai stream absent", "/v1/chat/completions", `{"model":"m"}`, false},

		// The one that would corrupt a live wire format. Ollama's native
		// surface answers in newline-delimited JSON and defaults stream to
		// TRUE when the field is absent, so an SSE comment frame injected
		// there is a parse error for every Ollama-native client.
		{"ollama native chat", "/api/chat", `{"model":"m","stream":true}`, false},
		{"ollama native generate", "/api/generate", `{"model":"m","stream":true}`, false},
		{"ollama native stream absent", "/api/chat", `{"model":"m"}`, false},

		{"malformed json", "/v1/chat/completions", `{"model":`, false},
		{"empty body", "/v1/chat/completions", ``, false},
		{"stream wrong type", "/v1/chat/completions", `{"stream":"yes"}`, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := wantsSSEKeepalive(tc.path, []byte(tc.body)); got != tc.want {
				t.Fatalf("wantsSSEKeepalive(%q, %q) = %v, want %v",
					tc.path, tc.body, got, tc.want)
			}
		})
	}
}

// The fast path must be byte-for-byte unchanged. Nearly every request
// finds the backend warm; if those started committing a 200 they would
// lose their real HTTP status codes for no benefit at all.
func TestKeepaliveWritesNothingWhenLaunchBeatsGrace(t *testing.T) {
	rec := httptest.NewRecorder()
	k := startSSEKeepalive(rec, 500*time.Millisecond, 10*time.Millisecond)
	if k == nil {
		t.Fatal("keepalive should have armed")
	}
	// Launch "completes" well inside the grace window.
	time.Sleep(20 * time.Millisecond)
	if committed := k.stop(); committed {
		t.Fatal("must not commit the response inside the grace window")
	}
	if rec.Body.Len() != 0 {
		t.Fatalf("wrote %q during the grace window; want nothing", rec.Body.String())
	}
	// The caller still owns the status code.
	rec.WriteHeader(http.StatusServiceUnavailable)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", rec.Code)
	}
}

func TestKeepaliveEmitsCommentFramesOnSlowLaunch(t *testing.T) {
	rec := httptest.NewRecorder()
	k := startSSEKeepalive(rec, 10*time.Millisecond, 10*time.Millisecond)
	time.Sleep(120 * time.Millisecond)
	committed := k.stop()

	if !committed {
		t.Fatal("a launch past the grace window must commit the response")
	}
	body := rec.Body.String()
	if !strings.HasPrefix(body, ": keepalive 1\n\n") {
		t.Fatalf("first frame = %q, want %q", body, ": keepalive 1\n\n")
	}
	if n := strings.Count(body, ": keepalive "); n < 2 {
		t.Fatalf("only %d frames in %q; expected repeats", n, body)
	}
	// Comment frames only -- nothing a client would parse as content.
	for _, line := range strings.Split(strings.TrimSpace(body), "\n") {
		if line != "" && !strings.HasPrefix(line, ": ") {
			t.Fatalf("non-comment line %q leaked into the stream", line)
		}
	}
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("Content-Type = %q, want text/event-stream", ct)
	}
	if rec.Header().Get("X-Accel-Buffering") != "no" {
		t.Fatal("X-Accel-Buffering: no is required or an intermediary may buffer the frames")
	}
}

// stop() must join the writer goroutine. If it returned early the proxy
// would write the real response while heartbeats were still interleaving
// comment frames into it.
func TestStopJoinsTheWriterGoroutine(t *testing.T) {
	rec := httptest.NewRecorder()
	k := startSSEKeepalive(rec, time.Millisecond, time.Millisecond)
	time.Sleep(30 * time.Millisecond)
	k.stop()

	after := rec.Body.Len()
	time.Sleep(30 * time.Millisecond)
	if rec.Body.Len() != after {
		t.Fatalf("keepalive wrote %d more bytes after stop()",
			rec.Body.Len()-after)
	}
}

func TestStopIsIdempotentAndNilSafe(t *testing.T) {
	var k *sseKeepalive
	if k.stop() {
		t.Fatal("nil keepalive must report not-committed")
	}
	rec := httptest.NewRecorder()
	k = startSSEKeepalive(rec, time.Millisecond, time.Millisecond)
	time.Sleep(20 * time.Millisecond)
	first := k.stop()
	if second := k.stop(); second != first {
		t.Fatalf("stop() not idempotent: %v then %v", first, second)
	}
}

func TestKeepaliveDisabledAndUnflushable(t *testing.T) {
	rec := httptest.NewRecorder()
	if k := startSSEKeepalive(rec, 0, 0); k != nil {
		t.Fatal("interval <= 0 must disable the feature")
	}
	nf := &nonFlusher{}
	if k := startSSEKeepalive(nf, time.Millisecond, time.Millisecond); k != nil {
		t.Fatal("a non-flushable writer must not be armed")
	}
	time.Sleep(20 * time.Millisecond)
	if nf.n != 0 || nf.status != 0 {
		t.Fatalf("wrote to a non-flushable writer: %d bytes, status %d", nf.n, nf.status)
	}
}

// Committing is one-way: after the first frame the status code is spent,
// so a launch failure has to be reported in-band. A stream that simply
// stops is indistinguishable from a hang and the client would wait out
// its own timeout instead of failing fast.
func TestWriteSSELaunchErrorOpenAIShape(t *testing.T) {
	rec := httptest.NewRecorder()
	writeSSELaunchError(rec, "/v1/chat/completions", errors.New("engine crashed at load"))

	body := rec.Body.String()
	if !strings.HasPrefix(body, "data: {") {
		t.Fatalf("body = %q, want a data: frame", body)
	}
	if !strings.Contains(body, "engine crashed at load") {
		t.Fatalf("error message missing from %q", body)
	}
	if !strings.HasSuffix(body, "data: [DONE]\n\n") {
		t.Fatalf("body = %q, must terminate with [DONE] so the client stops reading", body)
	}
}

func TestWriteSSELaunchErrorAnthropicShape(t *testing.T) {
	rec := httptest.NewRecorder()
	writeSSELaunchError(rec, "/v1/messages", errors.New("boom"))

	body := rec.Body.String()
	if !strings.HasPrefix(body, "event: error\ndata: {") {
		t.Fatalf("body = %q, want a named error event for the Anthropic surface", body)
	}
	if strings.Contains(body, "[DONE]") {
		t.Fatal("[DONE] is an OpenAI-ism and is not part of the Anthropic stream")
	}
}

// A client that hangs up mid-cold-start must not keep the goroutine
// beating into a dead socket for the rest of a 600s launch.
func TestKeepaliveStopsOnWriteError(t *testing.T) {
	fw := &failingWriter{h: http.Header{}}
	k := startSSEKeepalive(fw, time.Millisecond, time.Millisecond)
	if k == nil {
		t.Fatal("should have armed")
	}
	time.Sleep(40 * time.Millisecond)
	if got := fw.writes(); got != 1 {
		t.Fatalf("kept writing to a broken connection: %d attempts, want 1", got)
	}
	k.stop()
}

// The counter is read from the test goroutine while the keepalive
// goroutine may still be writing, so it needs its own lock. Production
// never needs one: stop() joins the writer before the caller touches the
// ResponseWriter again.
type failingWriter struct {
	h  http.Header
	mu sync.Mutex
	n  int
}

func (f *failingWriter) Header() http.Header { return f.h }
func (f *failingWriter) Write(b []byte) (int, error) {
	f.mu.Lock()
	f.n++
	f.mu.Unlock()
	return 0, errors.New("connection reset by peer")
}
func (f *failingWriter) WriteHeader(int) {}
func (f *failingWriter) Flush()          {}
func (f *failingWriter) writes() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.n
}

func TestItoa(t *testing.T) {
	for _, tc := range []struct {
		in   int
		want string
	}{{0, "0"}, {1, "1"}, {9, "9"}, {10, "10"}, {12345, "12345"}} {
		if got := itoa(tc.in); got != tc.want {
			t.Fatalf("itoa(%d) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
