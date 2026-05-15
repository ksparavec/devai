// Head-side request proxy.
//
// Per docs/plans/gpu-arbiter-cluster-mode.md Phase 2 step 5: forward
// the full request body to the chosen worker's /v1/cluster/inbound
// endpoint, including the (model, ctx, backend, reasoning) overrides
// untouched (so the existing override-parsing on the worker still
// works). Stream the response back to the client. For SSE preserve
// framing.

package main

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

// ClusterProxy implements HeadForwarder. Carries a token store so
// each forward attaches the bearer token the worker expects.
type ClusterProxy struct {
	Token  *TokenStore
	Client *http.Client
}

// NewClusterProxy returns a ClusterProxy with a sane default HTTP
// client. Tests can replace Client to inject a fake transport.
func NewClusterProxy(tokens *TokenStore) *ClusterProxy {
	return &ClusterProxy{
		Token: tokens,
		Client: &http.Client{
			// No overall timeout: streaming responses can run
			// arbitrarily long. The connection-level transport
			// timeouts are inherited from http.DefaultTransport.
			Timeout: 0,
		},
	}
}

// Forward proxies the (already-read body) request from the head's
// frontend to the chosen worker's inbound endpoint, then streams
// the response back.
func (p *ClusterProxy) Forward(
	w http.ResponseWriter, r *http.Request, worker WorkerEntry, parsed MinimalRequest,
) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "read body: "+err.Error(), http.StatusBadGateway)
		return
	}
	tok, err := p.Token.Read()
	if err != nil {
		http.Error(w, "read token: "+err.Error(), http.StatusInternalServerError)
		return
	}
	target := worker.Endpoint + "/v1/cluster/inbound"
	upstream, err := http.NewRequestWithContext(
		r.Context(), http.MethodPost, target, bytes.NewReader(body),
	)
	if err != nil {
		http.Error(w, "build upstream req: "+err.Error(), http.StatusInternalServerError)
		return
	}
	// Carry through Content-Type and any client headers that affect
	// the model server's interpretation (Accept, X-Request-ID).
	for _, k := range []string{"Content-Type", "Accept", "X-Request-Id"} {
		if v := r.Header.Get(k); v != "" {
			upstream.Header.Set(k, v)
		}
	}
	upstream.Header.Set("Authorization", "Bearer "+tok)
	upstream.Header.Set("X-Devai-Worker-Id", worker.WorkerID)
	upstream.Header.Set("X-Devai-Original-Path", r.URL.Path)

	start := time.Now()
	resp, err := p.Client.Do(upstream)
	if err != nil {
		w.Header().Set("Retry-After", "1")
		http.Error(w, fmt.Sprintf("upstream %s: %v", worker.WorkerID, err),
			http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// Mirror response headers to the client. Content-Length is left
	// off the wire when streaming so the client treats the response
	// as chunked.
	for k, vs := range resp.Header {
		for _, v := range vs {
			w.Header().Add(k, v)
		}
	}
	if resp.Header.Get("Content-Type") == "text/event-stream" {
		w.Header().Del("Content-Length")
	}
	w.WriteHeader(resp.StatusCode)

	// Stream the body. http.Flusher is needed for SSE; if the
	// ResponseWriter doesn't support it, we still copy bytes
	// (just no chunk-level flushing).
	flusher, _ := w.(http.Flusher)
	buf := make([]byte, 4096)
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				log.Printf("[proxy] client write failed mid-stream: %v", werr)
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			log.Printf("[proxy] upstream read failed mid-stream: %v", rerr)
			return
		}
	}
	log.Printf("[proxy] forwarded model=%s ctx=%d -> worker=%s status=%d in %s",
		parsed.Model, parsed.Context, worker.WorkerID,
		resp.StatusCode, time.Since(start).Round(time.Millisecond))
}
