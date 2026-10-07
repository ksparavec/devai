// Package backup implements the devai-backup manifest and archive logic:
// what gets backed up, how a snapshot is written, and how an archive is
// verified/restored safely.
package backup

import "path/filepath"

// Entry describes one backup source: either a single file (Dir=false) or a
// directory walked recursively (Dir=true, e.g. the sessions/ tree).
type Entry struct {
	SourcePath  string // absolute filesystem path
	ArchivePath string // path prefix inside the tar archive
	Dir         bool
}

// ArchiveRoots maps the top-level archive path segment back to the real
// filesystem root it was written from, so Restore can invert Manifest.
func ArchiveRoots(repoRoot, homeDir string) map[string]string {
	return map[string]string{
		"deploy":     filepath.Join(repoRoot, "deploy"),
		"devai-home": filepath.Join(homeDir, ".devai"),
	}
}

// retiredArchiveRoots are top-level archive segments that older snapshots
// carry but that no longer map to anything on disk. Restore skips their
// members with a warning instead of failing on an unknown root, so an
// archive taken before the root was retired still restores everything
// else. "sops-age" held ~/.config/sops/age/keys.txt until the sops/age
// scaffold moved to attic/sops-age/ on 2026-10-07 (platform-rebuild
// Phase 1, M12); the value is where to look for it now.
var retiredArchiveRoots = map[string]string{
	"sops-age": "attic/sops-age",
}

// Manifest returns every backup source for a given repo checkout and home
// directory. Missing sources are tolerated (a fresh host may not have probe
// caches yet) -- Snapshot skips them silently.
func Manifest(repoRoot, homeDir string) []Entry {
	deploy := filepath.Join(repoRoot, "deploy")
	devaiHome := filepath.Join(homeDir, ".devai")

	entries := []Entry{
		{SourcePath: filepath.Join(deploy, ".bench-cache.json"), ArchivePath: "deploy/.bench-cache.json"},
		{SourcePath: filepath.Join(deploy, ".ollama-reasoning-cache.json"), ArchivePath: "deploy/.ollama-reasoning-cache.json"},
		{SourcePath: filepath.Join(deploy, ".vllm-reasoning-cache.json"), ArchivePath: "deploy/.vllm-reasoning-cache.json"},
		{SourcePath: filepath.Join(deploy, ".sglang-reasoning-cache.json"), ArchivePath: "deploy/.sglang-reasoning-cache.json"},
		// The vllm-devai backend's cache: platform-rebuild 4v probes the
		// 0.30.0 build into it, and 4v's rollback restores this file alone
		// from the archive taken before the first probe.
		{SourcePath: filepath.Join(deploy, ".vllm-devai-reasoning-cache.json"), ArchivePath: "deploy/.vllm-devai-reasoning-cache.json"},
		{SourcePath: filepath.Join(deploy, ".model-status.json"), ArchivePath: "deploy/.model-status.json"},
		{SourcePath: filepath.Join(devaiHome, "preferences.yaml"), ArchivePath: "devai-home/preferences.yaml"},
		{SourcePath: filepath.Join(devaiHome, "sessions"), ArchivePath: "devai-home/sessions", Dir: true},
	}

	if matches, err := filepath.Glob(filepath.Join(deploy, "*.sops.env")); err == nil {
		for _, m := range matches {
			entries = append(entries, Entry{SourcePath: m, ArchivePath: "deploy/" + filepath.Base(m)})
		}
	}

	return entries
}
