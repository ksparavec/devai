package backup

import (
	"archive/tar"
	"compress/gzip"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// ArchiveInfo is one archive as reported by List.
type ArchiveInfo struct {
	Path    string    `json:"path"`
	Size    int64     `json:"size"`
	ModTime time.Time `json:"mtime"`
	TopDirs []string  `json:"top_dirs"`
}

// List enumerates *.tar.gz archives in dir, reading only tar headers (never
// decompressing bodies) to report each archive's top-level directories.
func List(dir string) ([]ArchiveInfo, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "*.tar.gz"))
	if err != nil {
		return nil, err
	}
	sort.Strings(matches)

	out := make([]ArchiveInfo, 0, len(matches))
	for _, m := range matches {
		info, err := os.Stat(m)
		if err != nil {
			continue
		}
		topDirs, err := topLevelDirs(m)
		if err != nil {
			return nil, err
		}
		out = append(out, ArchiveInfo{Path: m, Size: info.Size(), ModTime: info.ModTime(), TopDirs: topDirs})
	}
	return out, nil
}

func topLevelDirs(archivePath string) ([]string, error) {
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
	seen := map[string]bool{}
	var dirs []string
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		top := strings.SplitN(filepath.ToSlash(hdr.Name), "/", 2)[0]
		if !seen[top] {
			seen[top] = true
			dirs = append(dirs, top)
		}
	}
	sort.Strings(dirs)
	return dirs, nil
}
