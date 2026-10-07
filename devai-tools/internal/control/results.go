package control

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
)

// Result is results/<jobid>/result.json (C4).
type Result struct {
	V         int            `json:"v"`
	Job       int64          `json:"job"`
	Kind      string         `json:"kind"`
	Status    string         `json:"status"` // ok, failed, truncated, cancelled
	StartedAt string         `json:"started_at"`
	EndedAt   string         `json:"ended_at"`
	Summary   string         `json:"summary"`
	Outputs   map[string]any `json:"outputs"`
}

// writeResult is devai-control's part of C4's writer rule:
//   - an action-kind run: devai-control writes result.json at the run's end,
//     status ok, failed, or cancelled after a stop, outputs {exit_code};
//   - a bench or probe run: only when the run left none (the fallback),
//     status failed, or cancelled after a stop; never overwriting one.
func (a *actions) writeResult(r run) {
	status := "failed"
	switch {
	case r.Stopped:
		status = "cancelled"
	case r.Kind == "action" && r.ExitCode != nil && *r.ExitCode == 0:
		status = "ok"
	}
	var code any
	summary := ""
	if r.ExitCode != nil {
		code = *r.ExitCode
		summary = fmt.Sprintf("%s exited %d", r.Action, *r.ExitCode)
	} else {
		summary = fmt.Sprintf("%s lost: devai-control restarted and could not collect its exit status", r.Action)
	}
	if r.Stopped {
		summary += " after a stop"
	}
	res := Result{V: 1, Job: r.Job, Kind: r.Kind, Status: status, StartedAt: r.Since, EndedAt: r.Ended,
		Summary: summary, Outputs: map[string]any{"exit_code": code}}
	b, err := json.MarshalIndent(res, "", " ")
	if err != nil {
		return
	}
	b = append(b, '\n')
	path := filepath.Join(a.resultsDir(r.Job), "result.json")
	if r.Kind == "action" {
		tmp := path + ".tmp"
		if err := os.WriteFile(tmp, b, 0o664); err != nil {
			a.c.log.Printf("actions: job %d: result.json: %v", r.Job, err)
			return
		}
		if err := os.Rename(tmp, path); err != nil {
			a.c.log.Printf("actions: job %d: result.json: %v", r.Job, err)
		}
		return
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o664)
	if errors.Is(err, fs.ErrExist) {
		return // the run wrote its own: never overwritten
	}
	if err != nil {
		a.c.log.Printf("actions: job %d: result.json: %v", r.Job, err)
		return
	}
	if _, err := f.Write(b); err != nil {
		a.c.log.Printf("actions: job %d: result.json: %v", r.Job, err)
	}
	f.Close()
}
