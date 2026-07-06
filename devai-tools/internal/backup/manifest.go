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
	IsAgeKey    bool // triggers the recovery-critical warning when missing
}

// ArchiveRoots maps the top-level archive path segment back to the real
// filesystem root it was written from, so Restore can invert Manifest.
func ArchiveRoots(repoRoot, homeDir string) map[string]string {
	return map[string]string{
		"deploy":     filepath.Join(repoRoot, "deploy"),
		"devai-home": filepath.Join(homeDir, ".devai"),
		"sops-age":   filepath.Join(homeDir, ".config", "sops", "age"),
	}
}

// Manifest returns every backup source for a given repo checkout and home
// directory. Missing sources are tolerated (a fresh host may not have probe
// caches or an age key yet) -- Snapshot skips them silently, except the age
// key, which gets a loud warning since losing it means losing every secret
// sops+age encrypted with its public key ("no backdoor" -- docs/secrets.md).
func Manifest(repoRoot, homeDir string, excludeAgeKey bool) []Entry {
	deploy := filepath.Join(repoRoot, "deploy")
	devaiHome := filepath.Join(homeDir, ".devai")

	entries := []Entry{
		{SourcePath: filepath.Join(deploy, ".bench-cache.json"), ArchivePath: "deploy/.bench-cache.json"},
		{SourcePath: filepath.Join(deploy, ".ollama-reasoning-cache.json"), ArchivePath: "deploy/.ollama-reasoning-cache.json"},
		{SourcePath: filepath.Join(deploy, ".vllm-reasoning-cache.json"), ArchivePath: "deploy/.vllm-reasoning-cache.json"},
		{SourcePath: filepath.Join(deploy, ".sglang-reasoning-cache.json"), ArchivePath: "deploy/.sglang-reasoning-cache.json"},
		{SourcePath: filepath.Join(deploy, ".model-status.json"), ArchivePath: "deploy/.model-status.json"},
		{SourcePath: filepath.Join(devaiHome, "preferences.yaml"), ArchivePath: "devai-home/preferences.yaml"},
		{SourcePath: filepath.Join(devaiHome, "sessions"), ArchivePath: "devai-home/sessions", Dir: true},
	}

	if matches, err := filepath.Glob(filepath.Join(deploy, "*.sops.env")); err == nil {
		for _, m := range matches {
			entries = append(entries, Entry{SourcePath: m, ArchivePath: "deploy/" + filepath.Base(m)})
		}
	}

	if !excludeAgeKey {
		entries = append(entries, Entry{
			SourcePath:  filepath.Join(homeDir, ".config", "sops", "age", "keys.txt"),
			ArchivePath: "sops-age/keys.txt",
			IsAgeKey:    true,
		})
	}

	return entries
}
