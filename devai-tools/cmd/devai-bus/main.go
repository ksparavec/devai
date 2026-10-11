// Command devai-bus is the bus client of job scripts, the guard and the
// operator (docs/plans/platform-rebuild.md C1, C3; docs/slurm.md Sec. 4).
//
//	devai-bus request <subject> <json> --job N [--timeout D]
//	devai-bus publish <subject> <json>
//	devai-bus hold <devai.container.service> --job N --parent PID [--lease-only]
//	devai-bus watch [<subject>]
//	devai-bus log --dir DIR
//
// It reads DEVAI_BUS_URL, DEVAI_BUS_USER and DEVAI_BUS_PASSWORD_FILE.
// Exit codes (C3): 0 done; 1 refused or failed; 2 the bus could not be
// reached; 3 the service left its active state on its own.
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/sparavec/devai-tools/internal/buscli"
	"github.com/sparavec/gpu-arbiter/pkg/bus"
)

func usage() {
	fmt.Fprintln(os.Stderr, `usage: devai-bus <command> ...

  request <subject> <json> --job N [--timeout D]   send one command; print the reply
  publish <subject> <json>                         publish one event
  hold <devai.container.service> --job N --parent PID [--lease-only]
                                                   lease and wait for the job's model or run to end
  watch [<subject>]                                print events (default devai.events.>)
  log --dir DIR                                    write every event to DIR/<date>.jsonl

environment: DEVAI_BUS_URL, DEVAI_BUS_USER, DEVAI_BUS_PASSWORD_FILE
exit codes: 0 done, 1 refused or failed, 2 bus unreachable, 3 service left its active state`)
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(buscli.ExitFailed)
	}
	os.Exit(run(os.Args[1], os.Args[2:]))
}

func run(cmd string, argv []string) int {
	switch cmd {
	case "-h", "--help", "help":
		usage()
		return 0
	case "request":
		return runRequest(argv)
	case "publish":
		return runPublish(argv)
	case "hold":
		return runHold(argv)
	case "watch":
		return runWatch(argv)
	case "log":
		return runLog(argv)
	}
	usage()
	return buscli.ExitFailed
}

func busConfig() (bus.Config, bool) {
	c, err := bus.ConfigFromEnv()
	if err != nil {
		fmt.Fprintln(os.Stderr, "devai-bus:", err)
		return c, false
	}
	return c, true
}

func runRequest(argv []string) int {
	a, err := buscli.ParseArgs(argv, []string{"job", "timeout"}, nil)
	if err != nil || len(a.Pos) != 2 {
		fmt.Fprintln(os.Stderr, "usage: devai-bus request <subject> <json> --job N [--timeout D]", errText(err))
		return buscli.ExitFailed
	}
	js, ok := a.Vals["job"]
	if !ok {
		fmt.Fprintln(os.Stderr, "devai-bus request: --job is required (the operator's tools pass --job 0)")
		return buscli.ExitFailed
	}
	job, err := strconv.ParseInt(js, 10, 64)
	if err != nil || job < 0 {
		fmt.Fprintf(os.Stderr, "devai-bus request: bad --job %q\n", js)
		return buscli.ExitFailed
	}
	timeout := 30 * time.Second
	if t, ok := a.Vals["timeout"]; ok {
		if timeout, err = time.ParseDuration(t); err != nil || timeout <= 0 {
			fmt.Fprintf(os.Stderr, "devai-bus request: bad --timeout %q\n", t)
			return buscli.ExitFailed
		}
	}
	// Refuse a bad body before needing the bus.
	if _, err := buscli.BuildRequest(a.Pos[1], job); err != nil {
		fmt.Fprintln(os.Stderr, "devai-bus request:", err)
		return buscli.ExitFailed
	}
	cfg, ok := busConfig()
	if !ok {
		return buscli.ExitFailed
	}
	// SIGTERM keeps its default handling (C3): Slurm's TERM to the whole step
	// must not wait behind a long load.
	return buscli.Request(context.Background(), buscli.RequestOptions{
		Bus: cfg, Subject: a.Pos[0], Body: a.Pos[1], Job: job, Timeout: timeout,
		Stdout: os.Stdout, Stderr: os.Stderr,
	})
}

func runPublish(argv []string) int {
	a, err := buscli.ParseArgs(argv, nil, nil)
	if err != nil || len(a.Pos) != 2 {
		fmt.Fprintln(os.Stderr, "usage: devai-bus publish <subject> <json>", errText(err))
		return buscli.ExitFailed
	}
	cfg, ok := busConfig()
	if !ok {
		return buscli.ExitFailed
	}
	return buscli.Publish(context.Background(), buscli.PublishOptions{
		Bus: cfg, Subject: a.Pos[0], Body: a.Pos[1], Stderr: os.Stderr,
	})
}

func runHold(argv []string) int {
	// The hold must outlive Slurm's TERM to the whole step to see its run
	// end (C3); only Slurm's SIGKILL after KillWait, or its script's death,
	// ends it.
	signal.Ignore(syscall.SIGTERM, syscall.SIGINT)
	a, err := buscli.ParseArgs(argv, []string{"job", "parent", "lease-interval", "connect-timeout"}, []string{"lease-only"})
	if err != nil || len(a.Pos) != 1 {
		fmt.Fprintln(os.Stderr, "usage: devai-bus hold <devai.container.service> --job N --parent PID [--lease-only]", errText(err))
		return buscli.ExitFailed
	}
	job, err := strconv.ParseInt(a.Vals["job"], 10, 64)
	if err != nil || job <= 0 {
		fmt.Fprintf(os.Stderr, "devai-bus hold: bad --job %q\n", a.Vals["job"])
		return buscli.ExitFailed
	}
	parent := 0
	if p, ok := a.Vals["parent"]; ok {
		if parent, err = strconv.Atoi(p); err != nil || parent <= 1 {
			fmt.Fprintf(os.Stderr, "devai-bus hold: bad --parent %q\n", p)
			return buscli.ExitFailed
		}
	}
	if parent > 0 {
		// Die with the job script: SIGKILL when the parent dies, and exit at
		// once when the parent is already not the --parent pid (it died
		// before this line; the hold was re-parented to tini -s).
		if err := buscli.SetParentDeathSignal(syscall.SIGKILL); err != nil {
			fmt.Fprintln(os.Stderr, "devai-bus hold: prctl:", err)
			return buscli.ExitFailed
		}
		if os.Getppid() != parent {
			fmt.Fprintf(os.Stderr, "devai-bus hold: parent %d is gone; exiting\n", parent)
			return buscli.ExitFailed
		}
	}
	var interval, connect time.Duration
	if v, ok := a.Vals["lease-interval"]; ok {
		interval, _ = time.ParseDuration(v)
	}
	if v, ok := a.Vals["connect-timeout"]; ok {
		connect, _ = time.ParseDuration(v)
	}
	cfg, ok := busConfig()
	if !ok {
		return buscli.ExitFailed
	}
	hup := make(chan os.Signal, 1)
	if a.Bools["lease-only"] {
		signal.Notify(hup, syscall.SIGHUP)
	}
	o := buscli.HoldOptions{
		Bus: cfg, Subject: a.Pos[0], Job: job, LeaseOnly: a.Bools["lease-only"],
		LeaseInterval: interval, ConnectTimeout: connect, HUP: hup, Stderr: os.Stderr,
	}
	if parent > 0 {
		o.ParentAlive = func() bool { return os.Getppid() == parent }
	}
	return buscli.Hold(context.Background(), o)
}

func runWatch(argv []string) int {
	a, err := buscli.ParseArgs(argv, nil, nil)
	if err != nil || len(a.Pos) > 1 {
		fmt.Fprintln(os.Stderr, "usage: devai-bus watch [<subject>]", errText(err))
		return buscli.ExitFailed
	}
	cfg, ok := busConfig()
	if !ok {
		return buscli.ExitFailed
	}
	subject := ""
	if len(a.Pos) == 1 {
		subject = a.Pos[0]
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()
	return buscli.Watch(ctx, buscli.WatchOptions{Bus: cfg, Subject: subject, Stdout: os.Stdout, Stderr: os.Stderr})
}

func runLog(argv []string) int {
	a, err := buscli.ParseArgs(argv, []string{"dir"}, nil)
	if err != nil || len(a.Pos) != 0 || a.Vals["dir"] == "" {
		fmt.Fprintln(os.Stderr, "usage: devai-bus log --dir DIR", errText(err))
		return buscli.ExitFailed
	}
	cfg, ok := busConfig()
	if !ok {
		return buscli.ExitFailed
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()
	return buscli.Log(ctx, buscli.LogOptions{Bus: cfg, Dir: a.Vals["dir"], Stderr: os.Stderr})
}

func errText(err error) string {
	if err == nil {
		return ""
	}
	return "(" + err.Error() + ")"
}
