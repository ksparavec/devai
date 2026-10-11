package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func sha12(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])[:12]
}

func intPtr(n int64) *int64 { return &n }

func TestFingerprintRequest(t *testing.T) {
	cases := []struct {
		name string
		path string
		body string
		want requestFingerprint
	}{
		{
			name: "anthropic messages, system as string",
			path: "/v1/messages",
			body: `{"model":"claude-haiku-4-5","max_tokens":1,"system":"You are a title generator.",` +
				`"messages":[{"role":"user","content":"quota"}],"stream":false}`,
			want: requestFingerprint{Path: "/v1/messages", Model: "claude-haiku-4-5", MaxTokens: intPtr(1),
				NMessages: 1, HasSystem: true, SystemSHA256: sha12("You are a title generator.")},
		},
		{
			name: "anthropic messages, system as blocks, tools, stream",
			path: "/v1/messages",
			body: `{"model":"m","max_tokens":32000,"stream":true,` +
				`"system":[{"type":"text","text":"part one"},{"type":"text","text":"part two","cache_control":{"type":"ephemeral"}}],` +
				`"tools":[{"name":"Bash"}],"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]},{"role":"assistant","content":"ok"}]}`,
			want: requestFingerprint{Path: "/v1/messages", Model: "m", MaxTokens: intPtr(32000),
				NMessages: 2, HasSystem: true, SystemSHA256: sha12("part one\npart two"), HasTools: true, Stream: true},
		},
		{
			name: "openai chat, system-role messages, max_completion_tokens",
			path: "/v1/chat/completions",
			body: `{"model":"q","max_completion_tokens":64,"messages":[{"role":"system","content":"sys a"},` +
				`{"role":"user","content":"u"},{"role":"system","content":[{"type":"text","text":"sys b"}]}]}`,
			want: requestFingerprint{Path: "/v1/chat/completions", Model: "q", MaxTokens: intPtr(64),
				NMessages: 3, HasSystem: true, SystemSHA256: sha12("sys a\nsys b")},
		},
		{
			name: "no system, empty tools list",
			path: "/api/chat",
			body: `{"model":"qwen3.5:9b","messages":[{"role":"user","content":"hi"}],"tools":[]}`,
			want: requestFingerprint{Path: "/api/chat", Model: "qwen3.5:9b", NMessages: 1},
		},
		{
			name: "malformed JSON keeps path and size only",
			path: "/v1/messages",
			body: `{"model":`,
			want: requestFingerprint{Path: "/v1/messages"},
		},
		{
			name: "fields of the wrong type are ignored, not fatal",
			path: "/v1/messages",
			body: `{"model":7,"max_tokens":"lots","messages":{},"system":42,"tools":"x","stream":"yes"}`,
			want: requestFingerprint{Path: "/v1/messages"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.want.BodyBytes = len(tc.body)
			got := fingerprintRequest(tc.path, []byte(tc.body))
			gotJSON, _ := json.Marshal(got)
			wantJSON, _ := json.Marshal(tc.want)
			if !bytes.Equal(gotJSON, wantJSON) {
				t.Errorf("fingerprint\n got %s\nwant %s", gotJSON, wantJSON)
			}
		})
	}
}

// The whole point of 16a: a fingerprint line can be logged on a shared
// host without leaking what anyone asked. No message or system text may
// appear in it, only the hash.
func TestFingerprintLine_CarriesNoContent(t *testing.T) {
	const secretSystem = "SYSTEM-SECRET-7f3a"
	const secretUser = "USER-SECRET-91bc"
	body := `{"model":"m","system":"` + secretSystem + `","messages":[{"role":"user","content":"` + secretUser + `"}],` +
		`"tools":[{"name":"t","description":"TOOL-SECRET-22aa"}]}`
	line := fingerprintLine(fingerprintRequest("/v1/messages", []byte(body)))
	for _, secret := range []string{secretSystem, secretUser, "TOOL-SECRET-22aa"} {
		if strings.Contains(line, secret) {
			t.Errorf("fingerprint line leaks %q: %s", secret, line)
		}
	}
	if !strings.Contains(line, sha12(secretSystem)) {
		t.Errorf("fingerprint line lacks the system hash: %s", line)
	}
}

func TestFingerprintModeFromEnv(t *testing.T) {
	cases := []struct {
		fp, raw string
		want    fingerprintMode
	}{
		{"", "", fingerprintOff},
		{"off", "", fingerprintOff},
		{"on", "", fingerprintOn},
		{"ON", "", fingerprintOn},
		{"on", "on", fingerprintRaw},
		{"", "on", fingerprintRaw},
		{"1", "", fingerprintOff},
	}
	for _, tc := range cases {
		t.Setenv("DEVAI_REQUEST_FINGERPRINT", tc.fp)
		t.Setenv("DEVAI_REQUEST_FINGERPRINT_RAW", tc.raw)
		if got := fingerprintModeFromEnv(); got != tc.want {
			t.Errorf("fp=%q raw=%q: mode %v, want %v", tc.fp, tc.raw, got, tc.want)
		}
	}
}

// postThroughHandler drives the real request handler with one request on
// a serving vLLM backend and returns what the router logged and what the
// engine received. setup, when given, adjusts the arbiter first.
func postThroughHandler(t *testing.T, mode fingerprintMode, method, body string, setup func(*arbiter)) (logged string, upstreamBody []byte, status int) {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	t.Cleanup(server.Close)

	bs := testBackend("vllm", server)
	bs.running = true
	bs.currentModel = "test-model"
	a := testArbiter(bs)
	a.fingerprint = mode
	a.fingerprintDir = t.TempDir()
	if setup != nil {
		setup(a)
	}

	var buf bytes.Buffer
	prev := log.Writer()
	log.SetOutput(&buf)
	t.Cleanup(func() { log.SetOutput(prev) })

	req := httptest.NewRequest(method, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	a.makeRequestHandler("vllm")(w, req)
	return buf.String(), upstreamBody, w.Code
}

const fingerprintTestBody = `{"model":"test-model","messages":[{"role":"user","content":"PROMPT-TEXT-5e1d"}]}`

func TestFingerprint_OneLinePerPostWhenOn(t *testing.T) {
	logged, upstream, _ := postThroughHandler(t, fingerprintOn, http.MethodPost, fingerprintTestBody, nil)
	if n := strings.Count(logged, "fingerprint: "); n != 1 {
		t.Errorf("%d fingerprint lines, want 1:\n%s", n, logged)
	}
	if strings.Contains(logged, "raw_file") {
		t.Errorf("raw capture in plain mode:\n%s", logged)
	}
	if strings.Contains(logged, "PROMPT-TEXT-5e1d") {
		t.Errorf("prompt text reached the log:\n%s", logged)
	}
	if string(upstream) != fingerprintTestBody {
		t.Errorf("engine received %q, want the client's body unchanged", upstream)
	}
}

func TestFingerprint_NothingWhenOff(t *testing.T) {
	logged, upstream, _ := postThroughHandler(t, fingerprintOff, http.MethodPost, fingerprintTestBody, nil)
	if strings.Contains(logged, "fingerprint") {
		t.Errorf("fingerprint logged while off:\n%s", logged)
	}
	if string(upstream) != fingerprintTestBody {
		t.Errorf("engine received %q, want the client's body unchanged", upstream)
	}
}

// The forwarded body is the same whatever the switch says, also for a
// body the rewrite chain changes (suffix peeled, tools present).
func TestFingerprint_ForwardedBodyIndependentOfMode(t *testing.T) {
	body := `{"model":"test-model::nothink","stream":false,"messages":[{"role":"user","content":"hi"}],` +
		`"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}`
	_, off, _ := postThroughHandler(t, fingerprintOff, http.MethodPost, body, nil)
	_, on, _ := postThroughHandler(t, fingerprintOn, http.MethodPost, body, nil)
	_, raw, _ := postThroughHandler(t, fingerprintRaw, http.MethodPost, body, nil)
	if off == nil {
		t.Fatal("the engine received nothing")
	}
	if !bytes.Equal(off, on) || !bytes.Equal(off, raw) {
		t.Errorf("forwarded bodies differ by mode:\n off %s\n  on %s\n raw %s", off, on, raw)
	}
	if string(off) == body {
		t.Errorf("the chain left the body unchanged; this test needs one it rewrites")
	}
}

func TestFingerprint_RawModeWritesTheBodyToAFile(t *testing.T) {
	pretty := "{\n  \"model\": \"test-model\",\n  \"system\": \"S\",\n  \"messages\": [{\"role\": \"user\", \"content\": \"PROMPT-TEXT-5e1d\"}]\n}"
	var dir string
	logged, upstream, _ := postThroughHandler(t, fingerprintRaw, http.MethodPost, pretty,
		func(a *arbiter) { dir = a.fingerprintDir })
	if n := strings.Count(logged, "fingerprint: "); n != 1 {
		t.Fatalf("%d fingerprint lines, want 1:\n%s", n, logged)
	}
	if strings.Contains(logged, "PROMPT-TEXT-5e1d") {
		t.Errorf("raw mode put body content into the log:\n%s", logged)
	}
	line := logged[strings.Index(logged, "fingerprint: ")+len("fingerprint: "):]
	var fp requestFingerprint
	if err := json.Unmarshal([]byte(strings.TrimSpace(line)), &fp); err != nil {
		t.Fatalf("fingerprint line is not JSON: %v\n%s", err, line)
	}
	if !strings.HasSuffix(fp.RawFile, "-vllm-"+sha12("S")+".json") {
		t.Errorf("raw_file %q does not name backend and system hash", fp.RawFile)
	}
	got, err := os.ReadFile(filepath.Join(dir, fp.RawFile))
	if err != nil {
		t.Fatalf("raw file: %v", err)
	}
	if string(got) != pretty {
		t.Errorf("raw file is not the body byte for byte:\n%q", got)
	}
	if info, _ := os.Stat(filepath.Join(dir, fp.RawFile)); info.Mode().Perm() != 0o600 {
		t.Errorf("raw file mode %o, want 0600", info.Mode().Perm())
	}
	if string(upstream) != pretty {
		t.Errorf("engine received %q, want the client's body unchanged", upstream)
	}
}

func TestFingerprint_RawCaptureStopsAtTheCap(t *testing.T) {
	logged, _, _ := postThroughHandler(t, fingerprintRaw, http.MethodPost, fingerprintTestBody,
		func(a *arbiter) { a.fingerprintRawBytes.Store(fingerprintRawCapBytes) })
	if strings.Contains(logged, "raw_file") {
		t.Errorf("a body was written past the cap:\n%s", logged)
	}
	if !strings.Contains(logged, "raw capture stopped") {
		t.Errorf("reaching the cap was not logged:\n%s", logged)
	}
	if n := strings.Count(logged, "fingerprint: {"); n != 1 {
		t.Errorf("%d fingerprint lines past the cap, want 1:\n%s", n, logged)
	}
}

func TestFingerprint_LongModelIsCut(t *testing.T) {
	long := strings.Repeat("m", 10000)
	fp := fingerprintRequest("/v1/messages", []byte(`{"model":"`+long+`"}`))
	if len(fp.Model) > fingerprintModelMaxBytes+len("...(cut)") || !strings.HasSuffix(fp.Model, "...(cut)") {
		t.Errorf("model kept %d bytes", len(fp.Model))
	}
}

func TestFingerprint_GetIsNotFingerprinted(t *testing.T) {
	logged, _, _ := postThroughHandler(t, fingerprintOn, http.MethodGet, "", nil)
	if strings.Contains(logged, "fingerprint") {
		t.Errorf("a GET was fingerprinted:\n%s", logged)
	}
}

// A request refused by admission control is never read, so it is not
// fingerprinted (docs/router.md says so).
func TestFingerprint_RefusedAt429IsNotFingerprinted(t *testing.T) {
	logged, _, status := postThroughHandler(t, fingerprintOn, http.MethodPost, fingerprintTestBody,
		func(a *arbiter) {
			a.maxConcurrent = 1
			a.backends["vllm"].activeReqs = 1
		})
	if status != http.StatusTooManyRequests {
		t.Fatalf("status %d, want 429", status)
	}
	if strings.Contains(logged, "fingerprint") {
		t.Errorf("a 429'd request was fingerprinted:\n%s", logged)
	}
}

func TestFingerprintEnvWarnings(t *testing.T) {
	t.Setenv("DEVAI_REQUEST_FINGERPRINT", "true")
	t.Setenv("DEVAI_REQUEST_FINGERPRINT_RAW", "OFF")
	w := fingerprintEnvWarnings()
	if len(w) != 1 || !strings.Contains(w[0], `DEVAI_REQUEST_FINGERPRINT="true"`) {
		t.Errorf("warnings %q, want one naming DEVAI_REQUEST_FINGERPRINT", w)
	}
}
