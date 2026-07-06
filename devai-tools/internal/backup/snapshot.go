package backup

import (
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
)

// Snapshot writes a gzipped tar archive containing every existing manifest
// entry into destDir, named "<timestamp>.tar.gz". Each source file is read
// fully into memory before its tar header+body is written, rather than
// streamed against a pre-Stat'd size -- all backed-up files are small
// (hundreds of KB max), and buffering removes the race where a concurrent
// writer changes a file's size between Stat and copy, which would corrupt
// the tar stream itself (not just the JSON content).
func Snapshot(destDir string, manifest []Entry, timestamp string) (string, error) {
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return "", fmt.Errorf("create dest dir: %w", err)
	}
	archivePath := filepath.Join(destDir, timestamp+".tar.gz")

	f, err := os.Create(archivePath)
	if err != nil {
		return "", err
	}
	gz := gzip.NewWriter(f)
	tw := tar.NewWriter(gz)

	writeErr := func() error {
		for _, e := range manifest {
			if err := addEntry(tw, e); err != nil {
				return err
			}
		}
		return nil
	}()

	closeErr := tw.Close()
	if writeErr == nil {
		writeErr = closeErr
	}
	closeErr = gz.Close()
	if writeErr == nil {
		writeErr = closeErr
	}
	closeErr = f.Close()
	if writeErr == nil {
		writeErr = closeErr
	}

	if writeErr != nil {
		_ = os.Remove(archivePath)
		return "", writeErr
	}
	return archivePath, nil
}

func addEntry(tw *tar.Writer, e Entry) error {
	info, err := os.Lstat(e.SourcePath)
	if err != nil {
		if os.IsNotExist(err) {
			if e.IsAgeKey {
				fmt.Fprintf(os.Stderr,
					"WARNING: age key not found at %s -- skipping. "+
						"sops+age has no backdoor: losing this key means losing "+
						"every secret encrypted with its public key. Run "+
						"`make age-keygen-host` if this host should have one.\n",
					e.SourcePath)
			}
			return nil
		}
		return fmt.Errorf("stat %s: %w", e.SourcePath, err)
	}

	if e.Dir {
		return addDirEntry(tw, e.SourcePath, e.ArchivePath)
	}

	// Cache files under ~/.devai/ are symlinks back into deploy/, already
	// backed up directly from their deploy/ source -- skip to avoid double
	// counting, and skip any other stray symlink defensively.
	if info.Mode()&os.ModeSymlink != 0 {
		return nil
	}

	return writeFileEntry(tw, e.SourcePath, e.ArchivePath, info)
}

func addDirEntry(tw *tar.Writer, root, archiveRoot string) error {
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			if os.IsNotExist(err) {
				return nil
			}
			return err
		}
		if d.IsDir() {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		return writeFileEntry(tw, path, filepath.ToSlash(filepath.Join(archiveRoot, rel)), info)
	})
	if os.IsNotExist(err) {
		return nil
	}
	return err
}

func writeFileEntry(tw *tar.Writer, sourcePath, archivePath string, info fs.FileInfo) error {
	data, err := os.ReadFile(sourcePath)
	if err != nil {
		return fmt.Errorf("read %s: %w", sourcePath, err)
	}
	hdr := &tar.Header{
		Name:    archivePath,
		Mode:    int64(info.Mode().Perm()),
		Size:    int64(len(data)),
		ModTime: info.ModTime(),
	}
	if err := tw.WriteHeader(hdr); err != nil {
		return fmt.Errorf("write header %s: %w", archivePath, err)
	}
	_, err = tw.Write(data)
	return err
}
