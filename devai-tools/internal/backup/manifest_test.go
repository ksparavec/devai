package backup

import (
	"path/filepath"
	"testing"
)

func TestManifestIncludesAgeKeyByDefault(t *testing.T) {
	entries := Manifest("/repo", "/home/u", false)
	found := false
	for _, e := range entries {
		if e.IsAgeKey {
			found = true
			want := filepath.Join("/home/u", ".config", "sops", "age", "keys.txt")
			if e.SourcePath != want {
				t.Errorf("age key source = %q, want %q", e.SourcePath, want)
			}
			if e.ArchivePath != "sops-age/keys.txt" {
				t.Errorf("age key archive path = %q, want sops-age/keys.txt", e.ArchivePath)
			}
		}
	}
	if !found {
		t.Fatal("expected an IsAgeKey entry by default")
	}
}

func TestManifestExcludeAgeKey(t *testing.T) {
	entries := Manifest("/repo", "/home/u", true)
	for _, e := range entries {
		if e.IsAgeKey {
			t.Fatalf("expected no age key entry with excludeAgeKey=true, got %+v", e)
		}
	}
}

func TestManifestCacheFilesUnderDeploy(t *testing.T) {
	entries := Manifest("/repo", "/home/u", false)
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
	entries := Manifest("/repo", "/home/u", false)
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
	if roots["sops-age"] != filepath.Join("/home/u", ".config", "sops", "age") {
		t.Errorf("sops-age root = %q", roots["sops-age"])
	}
}
