package backup

import (
	"archive/tar"
	"fmt"
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
