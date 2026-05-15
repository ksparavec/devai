package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestBearerFromHeader(t *testing.T) {
	tests := []struct {
		name string
		hdr  string
		want string
	}{
		{"empty", "", ""},
		{"no-prefix", "tok", ""},
		{"bearer-lowercase", "bearer abc", "abc"},
		{"Bearer-canonical", "Bearer secret-token-xyz", "secret-token-xyz"},
		{"BEARER-uppercase", "BEARER UPPER", "UPPER"},
		{"with-trailing-whitespace", "Bearer   abc  ", "abc"},
		{"only-bearer", "Bearer", ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := bearerFromHeader(tc.hdr)
			if got != tc.want {
				t.Fatalf("got %q, want %q", got, tc.want)
			}
		})
	}
}

func writeTempToken(t *testing.T, contents string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "token")
	if err := os.WriteFile(p, []byte(contents), 0o600); err != nil {
		t.Fatalf("write token: %v", err)
	}
	return p
}

func TestTokenStore_ReadFreshAndCached(t *testing.T) {
	p := writeTempToken(t, "secret-1\n")
	ts := NewTokenStore(p, 24*time.Hour)
	got, err := ts.Read()
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if got != "secret-1" {
		t.Fatalf("got %q, want secret-1", got)
	}
	// Rewrite token; cached value should still win.
	if err := os.WriteFile(p, []byte("secret-2\n"), 0o600); err != nil {
		t.Fatalf("rewrite: %v", err)
	}
	got2, err := ts.Read()
	if err != nil {
		t.Fatalf("read 2: %v", err)
	}
	if got2 != "secret-1" {
		t.Fatalf("got %q, want cached secret-1", got2)
	}
}

func TestTokenStore_ReadAlwaysFresh_WhenTTLZero(t *testing.T) {
	p := writeTempToken(t, "secret-1\n")
	ts := NewTokenStore(p, 0)
	if v, _ := ts.Read(); v != "secret-1" {
		t.Fatalf("first read: got %q", v)
	}
	if err := os.WriteFile(p, []byte("secret-2"), 0o600); err != nil {
		t.Fatalf("rewrite: %v", err)
	}
	if v, _ := ts.Read(); v != "secret-2" {
		t.Fatalf("second read: got %q want secret-2", v)
	}
}

func TestTokenStore_EmptyFileFails(t *testing.T) {
	p := writeTempToken(t, "  \n")
	ts := NewTokenStore(p, 0)
	_, err := ts.Read()
	if err == nil {
		t.Fatalf("expected error for empty token")
	}
}

func TestTokenStore_MissingFile(t *testing.T) {
	ts := NewTokenStore("/no/such/file/devai-token", 0)
	_, err := ts.Read()
	if err == nil {
		t.Fatalf("expected error for missing file")
	}
}

func TestTokenStore_ValidateMatchesAndRejects(t *testing.T) {
	p := writeTempToken(t, "the-token")
	ts := NewTokenStore(p, time.Hour)

	good := httptest.NewRequest(http.MethodGet, "/whatever", nil)
	good.Header.Set("Authorization", "Bearer the-token")
	if err := ts.Validate(good); err != nil {
		t.Fatalf("good token: %v", err)
	}

	bad := httptest.NewRequest(http.MethodGet, "/whatever", nil)
	bad.Header.Set("Authorization", "Bearer wrong")
	if err := ts.Validate(bad); err == nil {
		t.Fatalf("bad token: expected error")
	}

	missing := httptest.NewRequest(http.MethodGet, "/whatever", nil)
	if err := ts.Validate(missing); err == nil {
		t.Fatalf("missing header: expected error")
	}
}

func TestTokenStore_AuthMiddleware(t *testing.T) {
	p := writeTempToken(t, "the-token")
	ts := NewTokenStore(p, time.Hour)

	called := false
	next := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	})
	mw := ts.AuthMiddleware(next)

	srv := httptest.NewServer(mw)
	defer srv.Close()

	// Reject without token.
	resp, err := http.Get(srv.URL)
	if err != nil {
		t.Fatalf("get without token: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("status without token = %d, want 401", resp.StatusCode)
	}
	if called {
		t.Fatalf("next was invoked despite missing token")
	}
	if got := resp.Header.Get("WWW-Authenticate"); got == "" {
		t.Fatalf("missing WWW-Authenticate header")
	}

	// Accept with token.
	called = false
	req, _ := http.NewRequest(http.MethodGet, srv.URL, nil)
	req.Header.Set("Authorization", "Bearer the-token")
	resp2, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("get with token: %v", err)
	}
	resp2.Body.Close()
	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("status with token = %d, want 200", resp2.StatusCode)
	}
	if !called {
		t.Fatalf("next not invoked despite valid token")
	}
}
