package backup

import (
	"path/filepath"
	"strings"
	"testing"
)

// The sops/age scaffold moved to attic/sops-age/ on 2026-10-07
// (platform-rebuild Phase 1, M12), and with it the age key left the
// manifest: nothing under ~/.config/sops is backed up any more.
func TestManifestCarriesNoAgeKey(t *testing.T) {
	for _, e := range Manifest("/repo", "/home/u") {
		if strings.Contains(e.SourcePath, filepath.Join(".config", "sops")) ||
			strings.HasPrefix(e.ArchivePath, "sops-age") {
			t.Errorf("unexpected sops/age entry in manifest: %+v", e)
		}
	}
}

func TestManifestCacheFilesUnderDeploy(t *testing.T) {
	entries := Manifest("/repo", "/home/u")
	want := []string{
		"deploy/.bench-cache.json",
		"deploy/.ollama-reasoning-cache.json",
		"deploy/.vllm-reasoning-cache.json",
		"deploy/.sglang-reasoning-cache.json",
		"deploy/.model-status.json",
	}
	got := map[string]bool{}
	for _, e := range entries {
		got[e.ArchivePath] = true
	}
	for _, w := range want {
		if !got[w] {
			t.Errorf("missing expected manifest entry %q", w)
		}
	}
}

func TestManifestSessionsIsDirEntry(t *testing.T) {
	entries := Manifest("/repo", "/home/u")
	for _, e := range entries {
		if e.ArchivePath == "devai-home/sessions" {
			if !e.Dir {
				t.Error("sessions entry should have Dir=true")
			}
			return
		}
	}
	t.Fatal("expected a devai-home/sessions entry")
}

func TestArchiveRootsRoundTrip(t *testing.T) {
	roots := ArchiveRoots("/repo", "/home/u")
	if roots["deploy"] != filepath.Join("/repo", "deploy") {
		t.Errorf("deploy root = %q", roots["deploy"])
	}
	if roots["devai-home"] != filepath.Join("/home/u", ".devai") {
		t.Errorf("devai-home root = %q", roots["devai-home"])
	}
	if _, ok := roots["sops-age"]; ok {
		t.Error("sops-age is a retired root; ArchiveRoots must not map it")
	}
	if len(roots) != 2 {
		t.Errorf("ArchiveRoots = %v, want exactly deploy and devai-home", roots)
	}
}
