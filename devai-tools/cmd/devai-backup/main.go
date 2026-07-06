// Command devai-backup snapshots and restores devai's irreplaceable
// host-local state: probe/bench caches, ~/.devai/ preferences+sessions, and
// the sops/age secret scaffold. See docs/backup-restore.md.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/sparavec/devai-tools/internal/backup"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}

	var err error
	switch os.Args[1] {
	case "snapshot":
		err = runSnapshot(os.Args[2:])
	case "list":
		err = runList(os.Args[2:])
	case "verify":
		err = runVerify(os.Args[2:])
	case "restore":
		err = runRestore(os.Args[2:])
	case "-h", "--help", "help":
		usage()
		return
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `usage: devai-backup <command> [flags]

commands:
  snapshot [--dest DIR] [--exclude-age-key] [--repo-root DIR] [--home-dir DIR]
  list     [--dest DIR] [--home-dir DIR]
  verify   --archive PATH
  restore  --archive PATH --yes [--repo-root DIR] [--home-dir DIR]`)
}

func defaultHomeDir() string {
	if h := os.Getenv("HOME"); h != "" {
		return h
	}
	h, _ := os.UserHomeDir()
	return h
}

func defaultRepoRoot() string {
	wd, _ := os.Getwd()
	return wd
}

func defaultBackupDir(homeDir string) string {
	if d := os.Getenv("DEVAI_BACKUP_DIR"); d != "" {
		return d
	}
	return filepath.Join(homeDir, ".devai", "backups")
}

func runSnapshot(args []string) error {
	fs := flag.NewFlagSet("snapshot", flag.ExitOnError)
	dest := fs.String("dest", "", "destination directory (default $DEVAI_BACKUP_DIR or ~/.devai/backups)")
	excludeAgeKey := fs.Bool("exclude-age-key", false, "do not include ~/.config/sops/age/keys.txt")
	repoRoot := fs.String("repo-root", "", "devai repo root (default: cwd)")
	homeDir := fs.String("home-dir", "", "home directory override (default: $HOME)")
	if err := fs.Parse(args); err != nil {
		return err
	}

	hd := *homeDir
	if hd == "" {
		hd = defaultHomeDir()
	}
	rr := *repoRoot
	if rr == "" {
		rr = defaultRepoRoot()
	}
	d := *dest
	if d == "" {
		d = defaultBackupDir(hd)
	}

	manifest := backup.Manifest(rr, hd, *excludeAgeKey)
	timestamp := time.Now().UTC().Format("20060102T150405Z")
	path, err := backup.Snapshot(d, manifest, timestamp)
	if err != nil {
		return err
	}
	fmt.Println(path)
	return nil
}

func runList(args []string) error {
	fs := flag.NewFlagSet("list", flag.ExitOnError)
	dest := fs.String("dest", "", "archive directory (default $DEVAI_BACKUP_DIR or ~/.devai/backups)")
	homeDir := fs.String("home-dir", "", "home directory override (default: $HOME)")
	if err := fs.Parse(args); err != nil {
		return err
	}

	hd := *homeDir
	if hd == "" {
		hd = defaultHomeDir()
	}
	d := *dest
	if d == "" {
		d = defaultBackupDir(hd)
	}

	archives, err := backup.List(d)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(archives)
}

func runVerify(args []string) error {
	fs := flag.NewFlagSet("verify", flag.ExitOnError)
	archive := fs.String("archive", "", "path to archive (required)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *archive == "" {
		return fmt.Errorf("--archive is required")
	}
	if err := backup.Verify(*archive); err != nil {
		return err
	}
	fmt.Println("OK:", *archive)
	return nil
}

func runRestore(args []string) error {
	fs := flag.NewFlagSet("restore", flag.ExitOnError)
	archive := fs.String("archive", "", "path to archive (required)")
	yes := fs.Bool("yes", false, "confirm this destructive operation")
	repoRoot := fs.String("repo-root", "", "devai repo root (default: cwd)")
	homeDir := fs.String("home-dir", "", "home directory override (default: $HOME)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *archive == "" {
		return fmt.Errorf("--archive is required")
	}
	if !*yes {
		return fmt.Errorf("refusing to restore without --yes")
	}

	hd := *homeDir
	if hd == "" {
		hd = defaultHomeDir()
	}
	rr := *repoRoot
	if rr == "" {
		rr = defaultRepoRoot()
	}
	timestamp := time.Now().UTC().Format("20060102T150405Z")

	if err := backup.Restore(*archive, rr, hd, timestamp); err != nil {
		return err
	}
	fmt.Println("restored from", *archive)
	return nil
}
