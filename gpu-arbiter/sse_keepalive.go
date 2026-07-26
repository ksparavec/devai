package main

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

// SSE keepalive during cold start.
//
// The router holds a client for the ENTIRE backend-launch window before a
// single byte is sent: makeRequestHandler -> ensureBackendRunning ->
// containerRecreate -> waitForHealthy -> proxy.ServeHTTP. An NVFP4 cold
// start is bounded by HEALTH_TIMEOUT_SECONDS, which defaults to 600s. A
// browser or corporate proxy with a 30-60s idle timeout drops the
// connection long before that, and the client retries -- which lands on a
// router that is still loading, so the expensive load is wasted and the
// cycle repeats. devai ships explicit HTTP_PROXY/HTTPS_PROXY support, so
// intermediaries are an expected part of the deployment.
//
// The fix is the standard one: emit SSE comment lines (`: keepalive N`)
// while waiting. Every OpenAI/Anthropic client ignores comment frames, so
// this is invisible to correct clients and resets the idle timer of every
// intermediary.
//
// Three constraints shape the design.
//
// It is gated on SSE responses only. Ollama's NATIVE surface (/api/chat,
// /api/generate) streams newline-delimited JSON, not SSE, and defaults
// `stream` to true when the field is absent -- injecting comment frames
// there would corrupt the stream for every Ollama-native client. So the
// gate requires an explicit `"stream": true` AND a `/v1/` path, which is
// exactly the set of surfaces that answer in SSE (Ollama's own
// OpenAI-compat endpoint included).
//
// It waits out a grace period first. Nearly every request finds the
// backend already warm and returns in milliseconds; those must keep their
// real HTTP status codes. Nothing is written -- and no header is
// committed -- until the launch has already been slow enough to be at
// risk, so the fast path is byte-for-byte unchanged.
//
// Committing is one-way. Once the first comment frame is written the
// response is a 200 text/event-stream and a later launch failure can no
// longer be a 5xx; it has to be reported in-band. writeSSELaunchError
// does that in the wire shape of whichever API was addressed.
const (
	sseKeepaliveIntervalDefault = 10
	sseKeepaliveGraceDefault    = 5
)

// wantsSSEKeepalive reports whether this request should get heartbeats
// during a slow launch. Requires an explicit `"stream": true` -- OpenAI
// defaults it to false, so absence means the client is not streaming --
// and a /v1/ path, since /api/ is Ollama-native NDJSON.
func wantsSSEKeepalive(path string, body []byte) bool {
	if !strings.HasPrefix(path, "/v1/") || len(body) == 0 {
		return false
	}
	var parsed struct {
		Stream *bool `json:"stream"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return false
	}
	return parsed.Stream != nil && *parsed.Stream
}

// sseKeepalive writes SSE comment frames to w until stopped.
//
// Exactly one goroutine writes to w at a time: the ticker goroutine owns
// w from start() until stop() returns, and stop() joins that goroutine
// before returning, so the caller may safely resume writing (or hand w to
// the proxy) afterwards. No lock is needed on w itself; the mutex guards
// only the `committed` flag, which stop() reads from the caller's
// goroutine.
type sseKeepalive struct {
	w       http.ResponseWriter
	flusher http.Flusher

	mu        sync.Mutex
	committed bool

	stopOnce sync.Once
	quit     chan struct{}
	done     chan struct{}
}

// startSSEKeepalive arms a heartbeat. It returns nil -- and writes
// nothing, ever -- when the feature is disabled or the ResponseWriter
// cannot flush, in which case a comment frame could sit in a buffer and
// defeat the entire point.
func startSSEKeepalive(w http.ResponseWriter, grace, interval time.Duration) *sseKeepalive {
	if interval <= 0 {
		return nil
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil
	}
	k := &sseKeepalive{
		w:       w,
		flusher: flusher,
		quit:    make(chan struct{}),
		done:    make(chan struct{}),
	}
	go k.run(grace, interval)
	return k
}

func (k *sseKeepalive) run(grace, interval time.Duration) {
	defer close(k.done)

	select {
	case <-k.quit:
		// The launch finished inside the grace window -- the common case.
		// Nothing was written, so the caller still owns the status code.
		return
	case <-time.After(grace):
	}

	n := 0
	for {
		n++
		if !k.beat(n) {
			return
		}
		select {
		case <-k.quit:
			return
		case <-time.After(interval):
		}
	}
}

// beat writes one comment frame, committing the response on the first
// one. Returns false when the write fails -- typically a client that has
// gone away, in which case further beats are pointless.
func (k *sseKeepalive) beat(n int) bool {
	k.mu.Lock()
	first := !k.committed
	if first {
		k.committed = true
	}
	k.mu.Unlock()

	if first {
		h := k.w.Header()
		h.Set("Content-Type", "text/event-stream")
		h.Set("Cache-Control", "no-cache")
		h.Set("Connection", "keep-alive")
		// Defeat proxy buffering; without it an intermediary can hold the
		// comment frames and the heartbeat never reaches the client.
		h.Set("X-Accel-Buffering", "no")
		k.w.WriteHeader(http.StatusOK)
		log.Printf("sse: backend still launching, sending keepalive frames to hold the connection")
	}
	if _, err := k.w.Write([]byte(": keepalive " + itoa(n) + "\n\n")); err != nil {
		return false
	}
	k.flusher.Flush()
	return true
}

// stop halts the heartbeat and waits for the writer goroutine to exit, so
// the caller regains exclusive ownership of the ResponseWriter. It
// reports whether the response was already committed as SSE -- if so, the
// caller must not attempt to set a status code.
//
// Safe to call more than once, and on a nil receiver, which is what the
// disabled/non-flushable path returns.
func (k *sseKeepalive) stop() bool {
	if k == nil {
		return false
	}
	k.stopOnce.Do(func() { close(k.quit) })
	<-k.done
	k.mu.Lock()
	defer k.mu.Unlock()
	return k.committed
}

// writeSSELaunchError reports a launch failure in-band, for the case
// where heartbeats already committed a 200. Without this the client would
// see a stream that simply stops -- indistinguishable from a hang, and it
// would sit there until its own timeout rather than failing fast.
//
// The frame shape follows the API that was addressed: Anthropic clients
// expect a named `error` event, OpenAI clients a `data:` payload followed
// by `[DONE]`.
func writeSSELaunchError(w http.ResponseWriter, path string, err error) {
	log.Printf("error (in-band, response already committed as SSE): %v", err)

	payload, _ := json.Marshal(map[string]any{
		"type": "error",
		"error": map[string]any{
			"type":    "server_error",
			"message": err.Error(),
		},
	})

	var frame string
	if strings.HasPrefix(path, "/v1/messages") {
		frame = "event: error\ndata: " + string(payload) + "\n\n"
	} else {
		frame = "data: " + string(payload) + "\n\ndata: [DONE]\n\n"
	}
	if _, werr := w.Write([]byte(frame)); werr != nil {
		return
	}
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

// itoa avoids pulling strconv in for one call site in the hot loop.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
