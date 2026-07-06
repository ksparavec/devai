package backup

import (
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

type plannedEntry struct {
	target string
	mode   os.FileMode
	data   []byte
}

// Restore extracts archivePath into repoRoot/homeDir, the inverse of
// Manifest via ArchiveRoots. It validates every entry before touching
// anything (Verify, then a second structural pass), then for each target
// that already exists renames it aside to "<path>.before-restore-<ts>"
// instead of deleting it, so a partial archive never destroys unrelated
// files. Finally it re-asserts 0600 on the age key as defense-in-depth.
// Callers gate this behind an explicit confirmation flag; Restore itself
// does not prompt.
func Restore(archivePath, repoRoot, homeDir, timestamp string) error {
	if err := Verify(archivePath); err != nil {
		return fmt.Errorf("archive failed validation, aborting before any writes: %w", err)
	}

	planned, err := planEntries(archivePath, ArchiveRoots(repoRoot, homeDir))
	if err != nil {
		return fmt.Errorf("archive failed validation, aborting before any writes: %w", err)
	}

	for _, p := range planned {
		if _, err := os.Lstat(p.target); err == nil {
			aside := p.target + ".before-restore-" + timestamp
			if err := os.Rename(p.target, aside); err != nil {
				return fmt.Errorf("rename aside %s: %w", p.target, err)
			}
		}
	}

	for _, p := range planned {
		if err := os.MkdirAll(filepath.Dir(p.target), 0o755); err != nil {
			return fmt.Errorf("mkdir for %s: %w", p.target, err)
		}
		if err := os.WriteFile(p.target, p.data, p.mode); err != nil {
			return fmt.Errorf("write %s: %w", p.target, err)
		}
	}

	ageKeyPath := filepath.Join(homeDir, ".config", "sops", "age", "keys.txt")
	if _, err := os.Stat(ageKeyPath); err == nil {
		_ = os.Chmod(ageKeyPath, 0o600)
	}

	return nil
}

func planEntries(archivePath string, roots map[string]string) ([]plannedEntry, error) {
	f, err := os.Open(archivePath)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer func() { _ = gz.Close() }()

	tr := tar.NewReader(gz)
	var planned []plannedEntry
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if err := ValidateHeader(hdr); err != nil {
			return nil, err
		}
		// Belt-and-suspenders on top of ValidateHeader/ValidateName: reject
		// any entry name containing ".." right before it's used to build a
		// filesystem path, so the guard sits directly against the value
		// that reaches resolveTarget below rather than only inside a
		// helper several calls away.
		if strings.Contains(hdr.Name, "..") {
			return nil, fmt.Errorf("rejected: path traversal %q", hdr.Name)
		}
		target, err := resolveTarget(roots, hdr.Name)
		if err != nil {
			return nil, err
		}
		data, err := io.ReadAll(tr)
		if err != nil {
			return nil, fmt.Errorf("read entry %s: %w", hdr.Name, err)
		}
		mode := os.FileMode(hdr.Mode)
		if mode == 0 {
			mode = 0o644
		}
		planned = append(planned, plannedEntry{target: target, mode: mode, data: data})
	}
	return planned, nil
}

func resolveTarget(roots map[string]string, name string) (string, error) {
	parts := strings.SplitN(filepath.ToSlash(name), "/", 2)
	root, ok := roots[parts[0]]
	if !ok {
		return "", fmt.Errorf("unknown archive root %q in entry %q", parts[0], name)
	}
	if len(parts) == 1 {
		return root, nil
	}
	target := filepath.Join(root, parts[1])
	if target != root && !strings.HasPrefix(target, root+string(os.PathSeparator)) {
		return "", fmt.Errorf("rejected: entry %q escapes root %q", name, root)
	}
	return target, nil
}
