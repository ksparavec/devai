// Command devai-jobs is the job queue and history tool (docs/slurm.md Sec. 9;
// platform-rebuild Phase 5, basic; Phase 13 completes it). It runs inside
// devai-slurm (and, from Phase 6, devai-operator), never on the host (I13):
// the host calls bin/devai-jobs, which runs it through podman exec.
//
//	devai-jobs list [--since D] [--json]   history from slurmdbd (default: the last 24 h)
//	devai-jobs show <id> [--json]          one job: slurmctld's view, else slurmdbd's record
//	devai-jobs queue [--json]              jobs slurmctld still holds, with their reasons
//	devai-jobs cancel <id> [--json]
//	devai-jobs node [--json]               the node devai: state and drain reason
//	devai-jobs submit <kind> [flags] [--json]
//
// It reads SLURM_REST_URL, SLURM_JWT_KEY_FILE and SLURM_USER (C7).
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/sparavec/devai-tools/internal/buscli"
	"github.com/sparavec/devai-tools/internal/jobs"
	"github.com/sparavec/gpu-arbiter/pkg/slurmrest"
)

const nodeName = "devai"

func usage(w io.Writer) {
	fmt.Fprintln(w, `usage: devai-jobs <command> [args] [--json]

  list [--since D]                 job history from slurmdbd (D: a duration such as 24h, or a date)
  show <id>                        one job
  queue                            pending and running jobs with their reasons, and the node
  cancel <id>                      cancel a job
  node                             the node devai: state and drain reason
  submit engine  --backend B --load JSON
  submit trainer --ft-job ID [--time-limit-min N]
  submit probe   --backend B --action A [--params JSON] [--model M] [--ctx N] [--mtp] [--time-limit-min N]
  submit bench   --action A [--params JSON] --run R --task T --backend B --model M [--ctx N] [--mtp] [--time-limit-min N]
  submit action  --action A [--params JSON] [--gpu --time-limit-min N] [--backend B --model M [--ctx N] [--mtp]]
                 every submit also takes --hold, --requester NAME, --git COMMIT

environment: SLURM_REST_URL, SLURM_JWT_KEY_FILE, SLURM_USER (default devai)`)
}

func main() {
	if len(os.Args) < 2 {
		usage(os.Stderr)
		os.Exit(2)
	}
	os.Exit(run(context.Background(), os.Args[1], os.Args[2:], os.Stdout, os.Stderr))
}

func run(ctx context.Context, cmd string, argv []string, stdout, stderr io.Writer) int {
	if cmd == "-h" || cmd == "--help" || cmd == "help" {
		usage(stdout)
		return 0
	}
	valueFlags := []string{"since", "backend", "load", "ft-job", "action", "params", "model", "ctx",
		"run", "task", "time-limit-min", "requester", "git"}
	a, err := buscli.ParseArgs(argv, valueFlags, []string{"json", "mtp", "gpu", "hold"})
	if err != nil {
		fmt.Fprintln(stderr, "devai-jobs:", err)
		return 2
	}
	asJSON := a.Bools["json"]
	c, err := slurmrest.FromEnv()
	if err != nil {
		fmt.Fprintln(stderr, "devai-jobs:", err)
		return 1
	}
	ctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	out := &printer{w: stdout, json: asJSON}
	switch cmd {
	case "list":
		err = list(ctx, c, a, out)
	case "show":
		err = show(ctx, c, a, out)
	case "queue":
		err = queue(ctx, c, out)
	case "cancel":
		err = cancelJob(ctx, c, a, out)
	case "node":
		err = node(ctx, c, out)
	case "submit":
		err = submit(ctx, c, a, out)
	default:
		usage(stderr)
		return 2
	}
	if err != nil {
		if asJSON {
			b, _ := json.Marshal(map[string]any{"ok": false, "error": err.Error()})
			fmt.Fprintln(stdout, string(b))
		}
		fmt.Fprintln(stderr, "devai-jobs:", err)
		return 1
	}
	return 0
}

type printer struct {
	w    io.Writer
	json bool
}

func (p *printer) emit(v any, text func(io.Writer)) {
	if p.json {
		b, _ := json.MarshalIndent(v, "", "  ")
		fmt.Fprintln(p.w, string(b))
		return
	}
	text(p.w)
}

func jobID(a *buscli.Args) (int64, error) {
	if len(a.Pos) != 1 {
		return 0, fmt.Errorf("want one job id")
	}
	id, err := strconv.ParseInt(a.Pos[0], 10, 64)
	if err != nil || id <= 0 {
		return 0, fmt.Errorf("bad job id %q", a.Pos[0])
	}
	return id, nil
}

// sinceArg turns --since (a duration, or a date) into a UNIX time.
func sinceArg(s string, now time.Time) (string, error) {
	if s == "" {
		s = "24h"
	}
	if d, err := time.ParseDuration(s); err == nil {
		return strconv.FormatInt(now.Add(-d).Unix(), 10), nil
	}
	for _, layout := range []string{time.RFC3339, "2006-01-02T15:04", "2006-01-02"} {
		if t, err := time.ParseInLocation(layout, s, time.Local); err == nil {
			return strconv.FormatInt(t.Unix(), 10), nil
		}
	}
	return "", fmt.Errorf("--since %q: want a duration (24h) or a date (2026-10-07)", s)
}

func exitText(e slurmrest.ExitCode) string {
	if rc, ok := e.ReturnCode.Value(); ok {
		if e.Signal.Name != "" {
			return fmt.Sprintf("%d:%s", rc, e.Signal.Name)
		}
		return strconv.FormatInt(rc, 10)
	}
	return "-"
}

func unixText(n int64) string {
	if n <= 0 {
		return "-"
	}
	return time.Unix(n, 0).Format("2006-01-02 15:04:05")
}

func list(ctx context.Context, c *slurmrest.Client, a *buscli.Args, out *printer) error {
	since, err := sinceArg(a.Vals["since"], time.Now())
	if err != nil {
		return err
	}
	rows, err := c.DBJobs(ctx, slurmrest.DBQuery{StartTime: since})
	if err != nil {
		return err
	}
	type row struct {
		ID       int64          `json:"job_id"`
		Name     string         `json:"name"`
		State    []string       `json:"state"`
		Reason   string         `json:"reason"`
		Exit     string         `json:"exit_code"`
		Submit   int64          `json:"submit_time"`
		Start    int64          `json:"start_time"`
		End      int64          `json:"end_time"`
		Comment  map[string]any `json:"comment"`
		AdminGPU string         `json:"admin_comment"`
	}
	var rs []row
	for _, j := range rows {
		rs = append(rs, row{ID: j.JobID, Name: j.Name, State: j.State.Current, Reason: j.State.Reason,
			Exit: exitText(j.ExitCode), Submit: j.Time.Submission, Start: j.Time.Start, End: j.Time.End,
			Comment: jobs.ParseComment(j.Comment.Job), AdminGPU: j.Comment.Administrator})
	}
	out.emit(map[string]any{"jobs": rs}, func(w io.Writer) {
		fmt.Fprintf(w, "%-8s %-22s %-12s %-6s %-19s %-19s %s\n", "JOBID", "NAME", "STATE", "EXIT", "START", "END", "GPU")
		for _, r := range rs {
			fmt.Fprintf(w, "%-8d %-22s %-12s %-6s %-19s %-19s %s\n", r.ID, r.Name, strings.Join(r.State, ","),
				r.Exit, unixText(r.Start), unixText(r.End), r.AdminGPU)
		}
	})
	return nil
}

func show(ctx context.Context, c *slurmrest.Client, a *buscli.Args, out *printer) error {
	id, err := jobID(a)
	if err != nil {
		return err
	}
	j, err := c.Job(ctx, id)
	if err == nil {
		v := map[string]any{"source": "slurmctld", "job": j, "comment": jobs.ParseComment(j.Comment)}
		out.emit(v, func(w io.Writer) {
			fmt.Fprintf(w, "job %d  %s  state=%s reason=%s licenses=%s exit=%s\ncomment: %s\n",
				j.JobID, j.Name, strings.Join(j.JobState, ","), j.StateReason, j.Licenses, exitText(j.ExitCode), j.Comment)
			if j.AdminComment != "" {
				fmt.Fprintf(w, "admin comment: %s\n", j.AdminComment)
			}
		})
		return nil
	}
	if !slurmrest.IsInvalidJob(err) {
		return err
	}
	d, err := c.DBJob(ctx, id)
	if err != nil {
		return err
	}
	v := map[string]any{"source": "slurmdbd", "job": d, "comment": jobs.ParseComment(d.Comment.Job)}
	out.emit(v, func(w io.Writer) {
		fmt.Fprintf(w, "job %d  %s  state=%s exit=%s\ncomment: %s\nadmin comment: %s\n", d.JobID, d.Name,
			strings.Join(d.State.Current, ","), exitText(d.ExitCode), d.Comment.Job, d.Comment.Administrator)
	})
	return nil
}

var doneStates = map[string]bool{"COMPLETED": true, "CANCELLED": true, "FAILED": true, "TIMEOUT": true,
	"NODE_FAIL": true, "PREEMPTED": true, "BOOT_FAIL": true, "DEADLINE": true, "OUT_OF_MEMORY": true}

func queue(ctx context.Context, c *slurmrest.Client, out *printer) error {
	js, err := c.Jobs(ctx)
	if err != nil {
		return err
	}
	n, nerr := c.Node(ctx, nodeName)
	type row struct {
		ID       int64    `json:"job_id"`
		Name     string   `json:"name"`
		State    []string `json:"state"`
		Reason   string   `json:"reason"`
		Licenses string   `json:"licenses"`
		Hold     bool     `json:"hold"`
	}
	var rs []row
	for _, j := range js {
		done := false
		for _, s := range j.JobState {
			if doneStates[s] {
				done = true
			}
		}
		if done {
			continue
		}
		rs = append(rs, row{ID: j.JobID, Name: j.Name, State: j.JobState, Reason: j.StateReason,
			Licenses: j.Licenses, Hold: j.Hold})
	}
	v := map[string]any{"jobs": rs}
	if nerr == nil {
		v["node"] = map[string]any{"name": n.Name, "state": n.State, "reason": n.Reason}
	} else {
		v["node_error"] = nerr.Error()
	}
	out.emit(v, func(w io.Writer) {
		fmt.Fprintf(w, "%-8s %-22s %-12s %-10s %s\n", "JOBID", "NAME", "STATE", "LICENSES", "REASON")
		for _, r := range rs {
			reason := r.Reason
			if r.Hold {
				reason += " (held)"
			}
			fmt.Fprintf(w, "%-8d %-22s %-12s %-10s %s\n", r.ID, r.Name, strings.Join(r.State, ","), r.Licenses, reason)
		}
		if nerr == nil {
			fmt.Fprintf(w, "node %s: %s", n.Name, strings.Join(n.State, ","))
			if n.Reason != "" {
				fmt.Fprintf(w, " -- %s", n.Reason)
			}
			fmt.Fprintln(w)
		}
	})
	return nil
}

func cancelJob(ctx context.Context, c *slurmrest.Client, a *buscli.Args, out *printer) error {
	id, err := jobID(a)
	if err != nil {
		return err
	}
	if err := c.Cancel(ctx, id); err != nil {
		return err
	}
	out.emit(map[string]any{"ok": true, "job_id": id}, func(w io.Writer) { fmt.Fprintf(w, "cancelled %d\n", id) })
	return nil
}

func node(ctx context.Context, c *slurmrest.Client, out *printer) error {
	n, err := c.Node(ctx, nodeName)
	if err != nil {
		return err
	}
	out.emit(n, func(w io.Writer) {
		fmt.Fprintf(w, "node %s: %s cpus=%d/%d memory=%d MiB", n.Name, strings.Join(n.State, ","), n.AllocCPUs, n.CPUs, n.RealMemory)
		if n.Reason != "" {
			fmt.Fprintf(w, " reason=%q", n.Reason)
		}
		fmt.Fprintln(w)
	})
	return nil
}

func submit(ctx context.Context, c *slurmrest.Client, a *buscli.Args, out *printer) error {
	if len(a.Pos) != 1 {
		return fmt.Errorf("submit takes one kind: engine, trainer, probe, bench or action")
	}
	spec := jobs.Spec{
		Kind: a.Pos[0], Backend: a.Vals["backend"], FTJob: a.Vals["ft-job"], Action: a.Vals["action"],
		Model: a.Vals["model"], Run: a.Vals["run"], Task: a.Vals["task"],
		MTP: a.Bools["mtp"], GPU: a.Bools["gpu"], Hold: a.Bools["hold"],
		Requester: a.Vals["requester"], Git: a.Vals["git"],
	}
	if spec.Requester == "" {
		spec.Requester = "devai-jobs"
	}
	if v := a.Vals["load"]; v != "" {
		spec.Load = json.RawMessage(v)
	}
	if v := a.Vals["params"]; v != "" {
		spec.Params = json.RawMessage(v)
	}
	for name, dst := range map[string]*int{"ctx": &spec.Ctx, "time-limit-min": &spec.TimeLimit} {
		if v := a.Vals[name]; v != "" {
			n, err := strconv.Atoi(v)
			if err != nil || n < 0 {
				return fmt.Errorf("bad --%s %q", name, v)
			}
			*dst = n
		}
	}
	b, err := jobs.Build(spec)
	if err != nil {
		return err
	}
	id, err := c.Submit(ctx, b.Desc)
	if err != nil {
		return err
	}
	out.emit(map[string]any{"ok": true, "job_id": id, "kind": spec.Kind, "name": b.Desc.Name},
		func(w io.Writer) { fmt.Fprintln(w, id) })
	return nil
}
