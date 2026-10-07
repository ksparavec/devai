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

// skippedEntry is an archive member under a retired root: reported, never
// written.
type skippedEntry struct {
	name    string // archive member name
	root    string // its retired top-level segment
	seeAlso string // where that root's content lives now
}

// warnOut receives Restore's skip warnings. A variable so tests can capture
// them; os.Stderr otherwise.
var warnOut io.Writer = os.Stderr

// Restore extracts archivePath into repoRoot/homeDir, the inverse of
// Manifest via ArchiveRoots. It validates every entry before touching
// anything (Verify, then a second structural pass), then for each target
// that already exists renames it aside to "<path>.before-restore-<ts>"
// instead of deleting it, so a partial archive never destroys unrelated
// files. Members under a retired archive root (retiredArchiveRoots, e.g.
// the sops/age key older snapshots carry) are skipped with a warning on
// stderr, never written. Callers gate this behind an explicit confirmation
// flag; Restore itself does not prompt.
func Restore(archivePath, repoRoot, homeDir, timestamp string) error {
	if err := Verify(archivePath); err != nil {
		return fmt.Errorf("archive failed validation, aborting before any writes: %w", err)
	}

	planned, skipped, err := planEntries(archivePath, ArchiveRoots(repoRoot, homeDir))
	if err != nil {
		return fmt.Errorf("archive failed validation, aborting before any writes: %w", err)
	}
	for _, s := range skipped {
		_, _ = fmt.Fprintf(warnOut,
			"WARNING: skipping %s: archive root %q is retired and nothing restores it any more (see %s)\n",
			s.name, s.root, s.seeAlso)
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

	return nil
}

func planEntries(archivePath string, roots map[string]string) ([]plannedEntry, []skippedEntry, error) {
	f, err := os.Open(archivePath)
	if err != nil {
		return nil, nil, err
	}
	defer func() { _ = f.Close() }()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, nil, err
	}
	defer func() { _ = gz.Close() }()

	tr := tar.NewReader(gz)
	var planned []plannedEntry
	var skipped []skippedEntry
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, err
		}
		if err := ValidateHeader(hdr); err != nil {
			return nil, nil, err
		}
		// Belt-and-suspenders on top of ValidateHeader/ValidateName: reject
		// any entry name containing ".." right before it's used to build a
		// filesystem path, so the guard sits directly against the value
		// that reaches resolveTarget below rather than only inside a
		// helper several calls away.
		if strings.Contains(hdr.Name, "..") {
			return nil, nil, fmt.Errorf("rejected: path traversal %q", hdr.Name)
		}
		// Only AFTER the traversal guard: a retired root must never become
		// a way to slip a hostile name past it ("sops-age/../evil" is still
		// rejected above, not skipped here).
		if root := strings.SplitN(filepath.ToSlash(hdr.Name), "/", 2)[0]; retiredArchiveRoots[root] != "" {
			skipped = append(skipped, skippedEntry{name: hdr.Name, root: root, seeAlso: retiredArchiveRoots[root]})
			continue
		}
		target, err := resolveTarget(roots, hdr.Name)
		if err != nil {
			return nil, nil, err
		}
		data, err := io.ReadAll(tr)
		if err != nil {
			return nil, nil, fmt.Errorf("read entry %s: %w", hdr.Name, err)
		}
		mode := os.FileMode(hdr.Mode)
		if mode == 0 {
			mode = 0o644
		}
		planned = append(planned, plannedEntry{target: target, mode: mode, data: data})
	}
	return planned, skipped, nil
}

func resolveTarget(roots map[string]string, name string) (string, error) {
	parts := strings.SplitN(filepath.ToSlash(name), "/", 2)
	root, ok := roots[parts[0]]
	if !ok {
		return "", fmt.Errorf("unknown archive root %q in entry %q", parts[0], name)
	}
	// A bare root name ("deploy") resolves to the root directory itself.
	// Manifest never emits one -- every ArchivePath it declares has a
	// "<root>/<leaf>" shape -- and honouring it would make Restore rename
	// the whole deploy/ tree aside (taking every probe and bench cache
	// with it) and write a single file in its place.
	if len(parts) == 1 {
		return "", fmt.Errorf("rejected: entry %q names archive root %q with no path under it", name, parts[0])
	}
	target := filepath.Join(root, parts[1])
	// Strict containment: target must be strictly BELOW root. Equality is
	// the same root-clobbering case as the bare-name check above, reached
	// via a trailing slash ("deploy/") or a self-cancelling segment.
	if !strings.HasPrefix(target, root+string(os.PathSeparator)) {
		return "", fmt.Errorf("rejected: entry %q escapes root %q", name, root)
	}
	if err := validateNoSymlinkComponents(root, target); err != nil {
		return "", err
	}
	return target, nil
}
