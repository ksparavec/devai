package backup

import (
	"archive/tar"
	"compress/gzip"
	"os"
	"path/filepath"
	"testing"
)

// writeRawArchive hand-builds a tar.gz from literal headers, bypassing
// Snapshot entirely, so tests can construct archives Snapshot itself would
// never produce (path traversal, symlink entries).
func writeRawArchive(t *testing.T, path string, headers []*tar.Header) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	gz := gzip.NewWriter(f)
	tw := tar.NewWriter(gz)
	for _, hdr := range headers {
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatal(err)
		}
		if hdr.Typeflag == tar.TypeReg {
			if _, err := tw.Write([]byte("payload")); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestVerifyRejectsPathTraversal(t *testing.T) {
	cases := []struct {
		name string
		hdr  *tar.Header
	}{
		{"relative-traversal", &tar.Header{Name: "../../evil", Typeflag: tar.TypeReg, Mode: 0o644, Size: 7}},
		{"absolute-path", &tar.Header{Name: "/etc/passwd", Typeflag: tar.TypeReg, Mode: 0o644, Size: 7}},
		{"embedded-traversal", &tar.Header{Name: "deploy/../../evil", Typeflag: tar.TypeReg, Mode: 0o644, Size: 7}},
		{"symlink-escape", &tar.Header{Name: "deploy/link", Typeflag: tar.TypeSymlink, Linkname: "/etc/passwd"}},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			archivePath := filepath.Join(dir, "evil.tar.gz")
			writeRawArchive(t, archivePath, []*tar.Header{c.hdr})

			if err := Verify(archivePath); err == nil {
				t.Errorf("Verify accepted malicious entry %q", c.hdr.Name)
			}
		})
	}
}

func TestRestoreRejectsMaliciousArchiveWithNoWrites(t *testing.T) {
	cases := []struct {
		name string
		hdr  *tar.Header
	}{
		{"relative-traversal", &tar.Header{Name: "../../evil", Typeflag: tar.TypeReg, Mode: 0o644, Size: 7}},
		{"absolute-path", &tar.Header{Name: "/etc/passwd", Typeflag: tar.TypeReg, Mode: 0o644, Size: 7}},
		{"symlink-escape", &tar.Header{Name: "deploy/link", Typeflag: tar.TypeSymlink, Linkname: "/etc/passwd"}},
		// "sops-age" is one archive-name segment but maps to a real root
		// three directories deep (homeDir/.config/sops/age): a single ".."
		// cancels cleanly in the archive-name's own coordinate space (so
		// ValidateName sees plain "evil" and accepts it) while still
		// escaping the real "sops-age" root into a sibling directory
		// (homeDir/.config/sops/evil) once joined. Only resolveTarget's
		// containment check on the resolved path catches this.
		{"root-depth-mismatch-escape", &tar.Header{Name: "sops-age/../evil", Typeflag: tar.TypeReg, Mode: 0o644, Size: 7}},
		// A bare root name resolves to the root DIRECTORY. Honouring it
		// would rename the whole deploy/ tree aside -- every probe and
		// bench cache with it -- and write a single file in its place.
		// Manifest never emits such an entry.
		{"bare-root-name", &tar.Header{Name: "deploy", Typeflag: tar.TypeReg, Mode: 0o644, Size: 7}},
		// Same clobber reached via a trailing slash. tar's writer only
		// allows one on a directory entry, so that is what a foreign
		// archive would carry here.
		{"root-name-trailing-slash", &tar.Header{Name: "deploy/", Typeflag: tar.TypeDir, Mode: 0o755}},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			parent := t.TempDir()
			repoRoot := filepath.Join(parent, "repo")
			homeDir := filepath.Join(parent, "home")
			mustMkdirAll(t, filepath.Join(repoRoot, "deploy"))
			mustMkdirAll(t, filepath.Join(homeDir, ".devai"))
			mustMkdirAll(t, filepath.Join(homeDir, ".config", "sops", "age"))

			archiveDir := t.TempDir()
			archivePath := filepath.Join(archiveDir, "evil.tar.gz")
			writeRawArchive(t, archivePath, []*tar.Header{c.hdr})

			before := snapshotTree(t, parent)

			if err := Restore(archivePath, repoRoot, homeDir, "20260101T000000Z"); err == nil {
				t.Fatalf("Restore accepted malicious entry %q", c.hdr.Name)
			}

			after := snapshotTree(t, parent)
			if before != after {
				t.Errorf("Restore modified the filesystem despite rejecting the archive:\nbefore=%q\nafter=%q", before, after)
			}
		})
	}
}

// TestRestoreLeavesRootTreeIntactOnBareRootEntry is the concrete
// consequence of the bare-root-name case above: the pre-existing caches
// under deploy/ must still be there, unrenamed, after the rejection.
func TestRestoreLeavesRootTreeIntactOnBareRootEntry(t *testing.T) {
	parent := t.TempDir()
	repoRoot := filepath.Join(parent, "repo")
	homeDir := filepath.Join(parent, "home")
	deploy := filepath.Join(repoRoot, "deploy")
	mustMkdirAll(t, deploy)
	mustMkdirAll(t, filepath.Join(homeDir, ".devai"))
	cache := filepath.Join(deploy, ".bench-cache.json")
	if err := os.WriteFile(cache, []byte(`{"real":"data"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	archivePath := filepath.Join(t.TempDir(), "evil.tar.gz")
	writeRawArchive(t, archivePath, []*tar.Header{
		{Name: "deploy", Typeflag: tar.TypeReg, Mode: 0o644, Size: 7},
	})

	if err := Restore(archivePath, repoRoot, homeDir, "20260101T000000Z"); err == nil {
		t.Fatal("Restore accepted a bare root-name entry")
	}

	fi, err := os.Lstat(deploy)
	if err != nil || !fi.IsDir() {
		t.Fatalf("deploy/ is no longer a directory: stat=%v err=%v", fi, err)
	}
	got, err := os.ReadFile(cache)
	if err != nil || string(got) != `{"real":"data"}` {
		t.Errorf("bench cache lost or clobbered: %q, err=%v", got, err)
	}
}

// A pre-existing symlink at an intermediate component lets a perfectly
// clean archive name write outside its declared root: the name never
// contains "..", so only a filesystem-aware check catches it.
func TestRestoreRejectsPreExistingSymlinkComponent(t *testing.T) {
	parent := t.TempDir()
	repoRoot := filepath.Join(parent, "repo")
	homeDir := filepath.Join(parent, "home")
	deploy := filepath.Join(repoRoot, "deploy")
	outside := filepath.Join(parent, "outside")
	mustMkdirAll(t, deploy)
	mustMkdirAll(t, filepath.Join(homeDir, ".devai"))
	mustMkdirAll(t, outside)
	if err := os.Symlink(outside, filepath.Join(deploy, "sub")); err != nil {
		t.Skipf("symlinks unsupported here: %v", err)
	}

	archivePath := filepath.Join(t.TempDir(), "evil.tar.gz")
	writeRawArchive(t, archivePath, []*tar.Header{
		{Name: "deploy/sub/pwned.json", Typeflag: tar.TypeReg, Mode: 0o644, Size: 7},
	})

	if err := Restore(archivePath, repoRoot, homeDir, "20260101T000000Z"); err == nil {
		t.Fatal("Restore followed a pre-existing symlink component")
	}
	if _, err := os.Stat(filepath.Join(outside, "pwned.json")); err == nil {
		t.Error("Restore wrote outside the declared root through the symlink")
	}
}

// snapshotTree returns a stable listing of every path + size under root, to
// detect any filesystem mutation a rejected restore should never cause.
func snapshotTree(t *testing.T, root string) string {
	t.Helper()
	var out string
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(root, path)
		out += rel + ":" + info.Mode().String() + "\n"
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return out
}
