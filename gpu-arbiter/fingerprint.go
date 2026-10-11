package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Request fingerprints (platform-rebuild Phase 16a): one content-free log
// line per POST, so the throwaway requests Claude Code sends (quota check,
// title generation, suggestion mode, ...) can be found by their shape and
// captured as fixtures before any short-circuit pattern is trusted. Observe
// only: the request is neither changed nor answered differently.
//
// DEVAI_REQUEST_FINGERPRINT=on logs the shape. The system text appears only
// as the first 12 hex digits of its SHA-256; message, tool and system text
// never appear. DEVAI_REQUEST_FINGERPRINT_RAW=on also writes each body,
// byte for byte, to its own file in the router container's own filesystem
// (fingerprintRawDir) and names the file in the line: content-revealing,
// meant only for capturing a fixture. Bodies never go to the log, because
// conmon splits long lines and `podman logs --timestamps` (the logger
// sidecar) splices a stamp into every 8 KiB chunk, and because the log is
// persisted. Both default to off. See docs/router.md.

type fingerprintMode int

const (
	fingerprintOff fingerprintMode = iota
	fingerprintOn
	fingerprintRaw
)

const (
	// fingerprintRawDir lives in the container's writable layer, so it
	// goes away with the container; the router also empties it at start
	// unless raw mode is on.
	fingerprintRawDir = "/tmp/devai-fingerprints"
	// fingerprintRawCapBytes bounds what raw mode writes per router
	// process: capture is a short operator step, not a recorder.
	fingerprintRawCapBytes = 256 << 20
	// fingerprintModelMaxBytes keeps a fingerprint line short whatever
	// the client puts in `model`; a longer value is cut and marked.
	fingerprintModelMaxBytes = 256
)

var fingerprintEnvVars = []string{"DEVAI_REQUEST_FINGERPRINT", "DEVAI_REQUEST_FINGERPRINT_RAW"}

func fingerprintModeFromEnv() fingerprintMode {
	if strings.EqualFold(os.Getenv("DEVAI_REQUEST_FINGERPRINT_RAW"), "on") {
		return fingerprintRaw
	}
	if strings.EqualFold(os.Getenv("DEVAI_REQUEST_FINGERPRINT"), "on") {
		return fingerprintOn
	}
	return fingerprintOff
}

// fingerprintEnvWarnings names every switch set to something other than
// on or off, which counts as off.
func fingerprintEnvWarnings() []string {
	var out []string
	for _, k := range fingerprintEnvVars {
		v := os.Getenv(k)
		if v != "" && !strings.EqualFold(v, "on") && !strings.EqualFold(v, "off") {
			out = append(out, fmt.Sprintf("%s=%q is neither on nor off; treated as off", k, v))
		}
	}
	return out
}

type requestFingerprint struct {
	Path         string `json:"path"`
	Model        string `json:"model"`
	MaxTokens    *int64 `json:"max_tokens"`
	NMessages    int    `json:"n_messages"`
	HasSystem    bool   `json:"has_system"`
	SystemSHA256 string `json:"system_sha256_12"`
	HasTools     bool   `json:"has_tools"`
	Stream       bool   `json:"stream"`
	BodyBytes    int    `json:"body_bytes"`
	RawFile      string `json:"raw_file,omitempty"`
}

// fingerprintRequest reads each field on its own, so a body that is not
// JSON, or a field of an unexpected type, leaves that part of the
// fingerprint at its zero value instead of failing. The system text is the
// top-level `system` (a string or a list of text blocks) followed by every
// role:"system" message, in order, joined with newlines: Claude Code sends
// both, and this is the client's body before the router folds them.
func fingerprintRequest(path string, body []byte) requestFingerprint {
	fp := requestFingerprint{Path: path, BodyBytes: len(body)}
	var fields map[string]json.RawMessage
	if json.Unmarshal(body, &fields) != nil {
		return fp
	}
	if json.Unmarshal(fields["model"], &fp.Model) == nil && len(fp.Model) > fingerprintModelMaxBytes {
		fp.Model = fp.Model[:fingerprintModelMaxBytes] + "...(cut)"
	}
	for _, key := range []string{"max_tokens", "max_completion_tokens"} {
		var n int64
		if raw, ok := fields[key]; ok && json.Unmarshal(raw, &n) == nil {
			fp.MaxTokens = &n
			break
		}
	}
	var stream bool
	if json.Unmarshal(fields["stream"], &stream) == nil {
		fp.Stream = stream
	}
	var tools []json.RawMessage
	if json.Unmarshal(fields["tools"], &tools) == nil {
		fp.HasTools = len(tools) > 0
	}

	var system []string
	system = append(system, contentTexts(fields["system"])...)
	var messages []struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if json.Unmarshal(fields["messages"], &messages) == nil {
		fp.NMessages = len(messages)
		for _, m := range messages {
			if m.Role == "system" {
				system = append(system, contentTexts(m.Content)...)
			}
		}
	}
	if text := strings.Join(system, "\n"); text != "" {
		fp.HasSystem = true
		sum := sha256.Sum256([]byte(text))
		fp.SystemSHA256 = hex.EncodeToString(sum[:])[:12]
	}
	return fp
}

// contentTexts returns the text of a content value: a plain string, or
// the `text` of each block in a list. Anything else yields nothing.
func contentTexts(raw json.RawMessage) []string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		if s == "" {
			return nil
		}
		return []string{s}
	}
	var blocks []struct {
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &blocks) != nil {
		return nil
	}
	var out []string
	for _, b := range blocks {
		if b.Text != "" {
			out = append(out, b.Text)
		}
	}
	return out
}

func fingerprintLine(fp requestFingerprint) string {
	b, _ := json.Marshal(fp)
	return string(b)
}

func (a *arbiter) logFingerprint(backendName, path string, body []byte) {
	if a.fingerprint == fingerprintOff {
		return
	}
	fp := fingerprintRequest(path, body)
	if a.fingerprint == fingerprintRaw {
		fp.RawFile = a.captureRawBody(backendName, body, fp.SystemSHA256)
	}
	log.Printf("[%s] fingerprint: %s", backendName, fingerprintLine(fp))
}

// captureRawBody writes the body verbatim to its own file and returns the
// file's name, or "" when nothing was written (the cap was reached, or the
// write failed; either is logged).
func (a *arbiter) captureRawBody(backendName string, body []byte, systemSHA string) string {
	if a.fingerprintRawBytes.Add(int64(len(body))) > fingerprintRawCapBytes {
		if a.fingerprintCapHit.CompareAndSwap(false, true) {
			log.Printf("[%s] fingerprint: raw capture stopped, %d MiB written this run (the cap)",
				backendName, fingerprintRawCapBytes>>20)
		}
		return ""
	}
	if systemSHA == "" {
		systemSHA = "nosystem"
	}
	name := fmt.Sprintf("%d-%s-%s.json", time.Now().UnixNano(), backendName, systemSHA)
	dir := a.fingerprintDir
	if dir == "" {
		dir = fingerprintRawDir
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		log.Printf("[%s] fingerprint: raw capture failed: %v", backendName, err)
		return ""
	}
	if err := os.WriteFile(filepath.Join(dir, name), body, 0o600); err != nil {
		log.Printf("[%s] fingerprint: raw capture failed: %v", backendName, err)
		return ""
	}
	return name
}
