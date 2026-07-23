package backup

import (
	"archive/tar"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ValidateName rejects a tar entry name that would escape its extraction
// root: absolute paths and ".." traversal segments. It works on the name
// alone (no real filesystem root needed), so both Verify (read-only) and
// Restore (before any writes) can share it.
func ValidateName(name string) error {
	if name == "" {
		return fmt.Errorf("rejected: empty entry name")
	}
	if filepath.IsAbs(name) {
		return fmt.Errorf("rejected: absolute path %q", name)
	}
	clean := filepath.ToSlash(filepath.Clean(name))
	if clean == ".." || strings.HasPrefix(clean, "../") {
		return fmt.Errorf("rejected: path traversal %q", name)
	}
	return nil
}

// ValidateHeader applies ValidateName plus a type check: the tool never
// writes symlink/hardlink tar entries itself, so any archive containing one
// is either foreign or hostile.
func ValidateHeader(hdr *tar.Header) error {
	if err := ValidateName(hdr.Name); err != nil {
		return err
	}
	switch hdr.Typeflag {
	case tar.TypeSymlink, tar.TypeLink:
		return fmt.Errorf("rejected: symlink/hardlink entry %q", hdr.Name)
	}
	return nil
}

// validateNoSymlinkComponents rejects a resolved target whose intermediate
// path components (strictly between root and the leaf) already exist as
// symlinks on disk. ValidateName/ValidateHeader work on the archive name
// alone and cannot see this: the name stays perfectly clean while
// Restore's MkdirAll + WriteFile follow the pre-existing link and write
// outside root. Only Restore can run this check -- Verify is read-only and
// has no extraction root to resolve against, so the two stay consistent by
// Restore applying ValidateHeader plus this, never less.
//
// The leaf itself is deliberately not checked: an existing leaf is renamed
// aside first, and os.Rename does not follow symlinks.
func validateNoSymlinkComponents(root, target string) error {
	rel, err := filepath.Rel(root, target)
	if err != nil {
		return fmt.Errorf("resolve %q under %q: %w", target, root, err)
	}
	parts := strings.Split(filepath.ToSlash(rel), "/")
	cur := root
	for _, part := range parts[:len(parts)-1] {
		cur = filepath.Join(cur, part)
		fi, err := os.Lstat(cur)
		if err != nil {
			if os.IsNotExist(err) {
				// Nothing exists from here down, so nothing can be a
				// pre-existing symlink; MkdirAll will create real dirs.
				return nil
			}
			return fmt.Errorf("stat %q: %w", cur, err)
		}
		if fi.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("rejected: entry path %q traverses existing symlink %q", target, cur)
		}
	}
	return nil
}
