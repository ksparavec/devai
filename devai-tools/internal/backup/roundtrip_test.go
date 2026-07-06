package backup

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// setupFixture builds a fake repo + home tree matching the shapes Manifest
// expects, with real content in every source file.
func setupFixture(t *testing.T) (repoRoot, homeDir string) {
	t.Helper()
	repoRoot = t.TempDir()
	homeDir = t.TempDir()

	deploy := filepath.Join(repoRoot, "deploy")
	mustMkdirAll(t, deploy)
	mustWriteFile(t, filepath.Join(deploy, ".bench-cache.json"), `{"bench":true}`)
	mustWriteFile(t, filepath.Join(deploy, ".ollama-reasoning-cache.json"), `{"ollama":true}`)
	mustWriteFile(t, filepath.Join(deploy, "prod.sops.env"), "ENC[...]")

	devaiHome := filepath.Join(homeDir, ".devai")
	mustMkdirAll(t, filepath.Join(devaiHome, "sessions"))
	mustWriteFile(t, filepath.Join(devaiHome, "preferences.yaml"), "vram: 24\n")
	mustWriteFile(t, filepath.Join(devaiHome, "sessions", "claude.jsonl"), `{"turn":1}`)

	ageDir := filepath.Join(homeDir, ".config", "sops", "age")
	mustMkdirAll(t, ageDir)
	mustWriteFile(t, filepath.Join(ageDir, "keys.txt"), "AGE-SECRET-KEY-1TESTTESTTEST\n")

	return repoRoot, homeDir
}

func mustMkdirAll(t *testing.T, dir string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
}

func mustWriteFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestSnapshotListVerifyRestoreRoundTrip(t *testing.T) {
	repoRoot, homeDir := setupFixture(t)
	destDir := t.TempDir()

	manifest := Manifest(repoRoot, homeDir, false)
	archivePath, err := Snapshot(destDir, manifest, "20260101T000000Z")
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	archives, err := List(destDir)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(archives) != 1 || archives[0].Path != archivePath {
		t.Fatalf("List = %+v, want one entry for %s", archives, archivePath)
	}
	if len(archives[0].TopDirs) == 0 {
		t.Fatal("expected non-empty TopDirs")
	}

	if err := Verify(archivePath); err != nil {
		t.Fatalf("Verify: %v", err)
	}

	// Capture originals, then delete every source so restore is the only
	// way the content comes back.
	origBench := mustReadFile(t, filepath.Join(repoRoot, "deploy", ".bench-cache.json"))
	origPrefs := mustReadFile(t, filepath.Join(homeDir, ".devai", "preferences.yaml"))
	origSession := mustReadFile(t, filepath.Join(homeDir, ".devai", "sessions", "claude.jsonl"))
	origAgeKey := mustReadFile(t, filepath.Join(homeDir, ".config", "sops", "age", "keys.txt"))

	if err := os.RemoveAll(repoRoot); err != nil {
		t.Fatal(err)
	}
	if err := os.RemoveAll(homeDir); err != nil {
		t.Fatal(err)
	}

	if err := Restore(archivePath, repoRoot, homeDir, "20260101T010101Z"); err != nil {
		t.Fatalf("Restore: %v", err)
	}

	assertFileEquals(t, filepath.Join(repoRoot, "deploy", ".bench-cache.json"), origBench)
	assertFileEquals(t, filepath.Join(homeDir, ".devai", "preferences.yaml"), origPrefs)
	assertFileEquals(t, filepath.Join(homeDir, ".devai", "sessions", "claude.jsonl"), origSession)
	assertFileEquals(t, filepath.Join(homeDir, ".config", "sops", "age", "keys.txt"), origAgeKey)

	info, err := os.Stat(filepath.Join(homeDir, ".config", "sops", "age", "keys.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Errorf("age key restored mode = %o, want 0600", info.Mode().Perm())
	}
}

func TestRestoreRenamesExistingRatherThanOverwriting(t *testing.T) {
	repoRoot, homeDir := setupFixture(t)
	destDir := t.TempDir()

	manifest := Manifest(repoRoot, homeDir, false)
	archivePath, err := Snapshot(destDir, manifest, "20260101T000000Z")
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	// Mutate the live file so we can tell original vs restored apart, and
	// confirm the pre-restore version survives under the .before-restore
	// suffix instead of being clobbered.
	benchPath := filepath.Join(repoRoot, "deploy", ".bench-cache.json")
	mustWriteFile(t, benchPath, `{"bench":"mutated-before-restore"}`)

	if err := Restore(archivePath, repoRoot, homeDir, "20260101T020202Z"); err != nil {
		t.Fatalf("Restore: %v", err)
	}

	asideContent := mustReadFile(t, benchPath+".before-restore-20260101T020202Z")
	if string(asideContent) != `{"bench":"mutated-before-restore"}` {
		t.Errorf("renamed-aside content = %q", asideContent)
	}
	restored := mustReadFile(t, benchPath)
	if string(restored) != `{"bench":true}` {
		t.Errorf("restored content = %q, want original archived content", restored)
	}
}

func TestConcurrentWriterDuringSnapshotProducesValidTar(t *testing.T) {
	repoRoot, homeDir := setupFixture(t)
	destDir := t.TempDir()
	manifest := Manifest(repoRoot, homeDir, false)

	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		path := filepath.Join(repoRoot, "deploy", ".bench-cache.json")
		i := 0
		for {
			select {
			case <-stop:
				return
			default:
				i++
				_ = os.WriteFile(path, []byte(`{"bench":`+time.Now().Format(time.RFC3339Nano)+`}`), 0o644)
			}
			if i > 2000 {
				return
			}
		}
	}()

	archivePath, err := Snapshot(destDir, manifest, "20260101T000000Z")
	close(stop)
	wg.Wait()

	if err != nil {
		t.Fatalf("Snapshot with concurrent writer: %v", err)
	}
	if err := Verify(archivePath); err != nil {
		t.Fatalf("Verify after concurrent-writer snapshot: %v", err)
	}
}

func TestSnapshotMissingSourcesToleratedOnFreshHost(t *testing.T) {
	repoRoot := t.TempDir()
	homeDir := t.TempDir()
	destDir := t.TempDir()

	manifest := Manifest(repoRoot, homeDir, false)
	archivePath, err := Snapshot(destDir, manifest, "20260101T000000Z")
	if err != nil {
		t.Fatalf("Snapshot on fresh host (no sources exist): %v", err)
	}
	if err := Verify(archivePath); err == nil {
		t.Fatal("expected Verify to reject an archive with zero entries")
	}
}

func mustReadFile(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return b
}

func assertFileEquals(t *testing.T, path string, want []byte) {
	t.Helper()
	got := mustReadFile(t, path)
	if string(got) != string(want) {
		t.Errorf("%s = %q, want %q", path, got, want)
	}
}
