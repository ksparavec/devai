// Package envfile implements add-or-replace mutation of a KEY=VALUE .env
// file, preserving every other line (comments, blank lines, unrelated
// keys) untouched. Used by devai-gpu-vendor rather than a Makefile sed
// snippet, since a fresh checkout's .env may not have any of the target
// keys yet -- a blind substitution can't cleanly cover "key absent".
package envfile

import (
	"fmt"
	"os"
	"regexp"
	"strings"
)

var keyLineRE = regexp.MustCompile(`^([A-Za-z_][A-Za-z0-9_]*)=`)

// SetKeys reads the .env file at path (an absent file is treated as
// empty, not an error) and writes back a version where every key in
// order has the value kv[key]: an existing uncommented "KEY=..." line is
// replaced in place; a key with no existing line is appended at the end.
// Every other line is preserved byte-for-byte.
func SetKeys(path string, order []string, kv map[string]string) error {
	var lines []string
	data, err := os.ReadFile(path)
	switch {
	case err == nil:
		lines = strings.Split(string(data), "\n")
		if len(lines) > 0 && lines[len(lines)-1] == "" {
			lines = lines[:len(lines)-1]
		}
	case os.IsNotExist(err):
		lines = nil
	default:
		return err
	}

	pending := make(map[string]bool, len(order))
	for _, k := range order {
		pending[k] = true
	}

	for i, line := range lines {
		m := keyLineRE.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		key := m[1]
		if pending[key] {
			lines[i] = fmt.Sprintf("%s=%s", key, kv[key])
			delete(pending, key)
		}
	}

	for _, k := range order {
		if pending[k] {
			lines = append(lines, fmt.Sprintf("%s=%s", k, kv[k]))
		}
	}

	return os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644)
}
