// Bearer-token loading + validation for the cluster control plane.
//
// Token lives in /run/devai/cluster-token (tmpfs-backed, mode 0600,
// rendered from cluster-token.sops.env by the shared sops/age
// scaffold). We re-read on every Validate so a rotated token
// becomes effective by the next heartbeat without restart.

package main

import (
	"crypto/subtle"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// TokenStore loads a bearer token from a file and caches it for a
// short interval. New() returns a store that re-reads the file when
// the cached value is older than CacheTTL.
//
// All exported methods are safe for concurrent use.
type TokenStore struct {
	Path     string
	CacheTTL time.Duration

	mu        sync.RWMutex
	cached    string
	lastRead  time.Time
	lastError error
}

// NewTokenStore returns a store backed by `path`. CacheTTL caps how
// long a stale-on-write token can authenticate -- 30s is a sane
// default; tests pass 0 to force every Validate to re-read.
func NewTokenStore(path string, cacheTTL time.Duration) *TokenStore {
	return &TokenStore{Path: path, CacheTTL: cacheTTL}
}

// Read returns the current token. Re-reads the file when the cached
// value is older than CacheTTL, OR when CacheTTL == 0.
func (t *TokenStore) Read() (string, error) {
	t.mu.RLock()
	if t.CacheTTL > 0 && !t.lastRead.IsZero() && time.Since(t.lastRead) < t.CacheTTL && t.lastError == nil {
		v := t.cached
		t.mu.RUnlock()
		return v, nil
	}
	t.mu.RUnlock()

	t.mu.Lock()
	defer t.mu.Unlock()
	// Double-check after promotion to write lock.
	if t.CacheTTL > 0 && !t.lastRead.IsZero() && time.Since(t.lastRead) < t.CacheTTL && t.lastError == nil {
		return t.cached, nil
	}
	data, err := os.ReadFile(t.Path)
	t.lastRead = time.Now()
	if err != nil {
		t.lastError = fmt.Errorf("read token at %s: %w", t.Path, err)
		t.cached = ""
		return "", t.lastError
	}
	tok := strings.TrimSpace(string(data))
	if tok == "" {
		t.lastError = fmt.Errorf("token at %s is empty", t.Path)
		t.cached = ""
		return "", t.lastError
	}
	t.cached = tok
	t.lastError = nil
	return tok, nil
}

// Validate compares the Authorization header on `r` against the
// loaded token in constant time. Returns nil on success, an error
// describing the failure on rejection.
func (t *TokenStore) Validate(r *http.Request) error {
	expected, err := t.Read()
	if err != nil {
		return fmt.Errorf("token store: %w", err)
	}
	got := bearerFromHeader(r.Header.Get("Authorization"))
	if got == "" {
		return errors.New("missing or malformed Authorization header")
	}
	if subtle.ConstantTimeCompare([]byte(got), []byte(expected)) != 1 {
		return errors.New("bearer token does not match")
	}
	return nil
}

// AuthMiddleware wraps a handler and rejects requests whose bearer
// token doesn't match. Used by both worker (/v1/cluster/inbound) and
// head (/v1/cluster/{register,heartbeat}). On reject, responds with
// 401 + a brief WWW-Authenticate hint.
func (t *TokenStore) AuthMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := t.Validate(r); err != nil {
			w.Header().Set("WWW-Authenticate", `Bearer realm="devai-cluster"`)
			http.Error(w, "Unauthorized: "+err.Error(), http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// bearerFromHeader extracts the token from a "Bearer <token>" header
// (case-insensitive scheme). Returns "" when the header is missing
// or malformed.
func bearerFromHeader(authHeader string) string {
	authHeader = strings.TrimSpace(authHeader)
	if authHeader == "" {
		return ""
	}
	const prefix = "bearer "
	if len(authHeader) <= len(prefix) {
		return ""
	}
	if !strings.EqualFold(authHeader[:len(prefix)], prefix) {
		return ""
	}
	return strings.TrimSpace(authHeader[len(prefix):])
}
