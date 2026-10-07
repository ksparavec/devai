package backup

import (
	"archive/tar"
	"bytes"
	"os"
	"path/filepath"
	"strings"
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

	// An age key left on the host from the retired sops/age scaffold
	// (attic/sops-age/). It is NOT a manifest source any more; the round
	// trip below proves it is neither archived nor restored.
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

	manifest := Manifest(repoRoot, homeDir)
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
	origSops := mustReadFile(t, filepath.Join(repoRoot, "deploy", "prod.sops.env"))

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
	assertFileEquals(t, filepath.Join(repoRoot, "deploy", "prod.sops.env"), origSops)

	if _, err := os.Lstat(filepath.Join(homeDir, ".config", "sops")); !os.IsNotExist(err) {
		t.Errorf("restore recreated ~/.config/sops (err=%v); the age key is no longer backed up", err)
	}
}

// An archive written before the sops/age scaffold moved to the attic
// (2026-10-07) carries sops-age/keys.txt. Restoring it must still bring
// back everything else, skip the key with a warning naming the attic, and
// write nothing under ~/.config/sops.
func TestRestoreSkipsRetiredSopsAgeMembers(t *testing.T) {
	parent := t.TempDir()
	repoRoot := filepath.Join(parent, "repo")
	homeDir := filepath.Join(parent, "home")
	archivePath := filepath.Join(t.TempDir(), "old.tar.gz")
	writeRawArchive(t, archivePath, []*tar.Header{
		{Name: "sops-age/keys.txt", Typeflag: tar.TypeReg, Mode: 0o600, Size: 7},
		{Name: "deploy/.bench-cache.json", Typeflag: tar.TypeReg, Mode: 0o644, Size: 7},
	})

	var warnings bytes.Buffer
	prev := warnOut
	warnOut = &warnings
	t.Cleanup(func() { warnOut = prev })

	if err := Restore(archivePath, repoRoot, homeDir, "20260101T000000Z"); err != nil {
		t.Fatalf("Restore of an archive with a sops-age member: %v", err)
	}

	assertFileEquals(t, filepath.Join(repoRoot, "deploy", ".bench-cache.json"), []byte("payload"))
	if _, err := os.Lstat(filepath.Join(homeDir, ".config", "sops")); !os.IsNotExist(err) {
		t.Errorf("Restore wrote under ~/.config/sops (err=%v)", err)
	}
	for _, want := range []string{"sops-age/keys.txt", "attic/sops-age"} {
		if !strings.Contains(warnings.String(), want) {
			t.Errorf("skip warning %q does not name %q", warnings.String(), want)
		}
	}
}

func TestRestoreRenamesExistingRatherThanOverwriting(t *testing.T) {
	repoRoot, homeDir := setupFixture(t)
	destDir := t.TempDir()

	manifest := Manifest(repoRoot, homeDir)
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
	manifest := Manifest(repoRoot, homeDir)

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

	manifest := Manifest(repoRoot, homeDir)
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
