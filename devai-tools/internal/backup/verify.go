package backup

import (
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io"
	"os"
)

// Verify opens archivePath read-only and walks every entry, validating each
// against ValidateHeader without extracting. It returns the first rejected
// entry's error, or nil if every entry is safe.
func Verify(archivePath string) error {
	f, err := os.Open(archivePath)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return fmt.Errorf("not a gzip archive: %w", err)
	}
	defer func() { _ = gz.Close() }()

	tr := tar.NewReader(gz)
	count := 0
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("corrupt tar stream: %w", err)
		}
		if err := ValidateHeader(hdr); err != nil {
			return err
		}
		count++
	}
	if count == 0 {
		return fmt.Errorf("archive contains no entries")
	}
	return nil
}
