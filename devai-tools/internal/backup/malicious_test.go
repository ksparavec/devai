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
