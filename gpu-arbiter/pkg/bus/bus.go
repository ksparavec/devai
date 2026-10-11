// Package bus is devai's NATS bus contract (docs/plans/platform-rebuild.md,
// contract C1; docs/slurm.md Sec. 4): subjects, the request/reply envelope,
// verbs, states and error codes, plus a small client (client.go).
//
// Commands are request/reply on devai.<container>.<service>.<verb>; events
// are published by devai-control on devai.events.<container>.<service>, on
// every state change and every 30 s as a heartbeat. Messages are JSON, at
// most 64 KiB.
//
// The router, devai-control, devai-bus and devai-jobs all share this
// package (I2), so the wire format has one definition.
package bus

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"
	"unicode/utf8"
)

// Version is the envelope version every request and reply carries.
const Version = 1

// MaxMessageBytes is the largest message on the bus (nats-server's
// max_payload in deploy/slurm/nats-server.conf.in).
const MaxMessageBytes = 64 * 1024

// MaxLogTailBytes bounds an error's log tail so a reply always fits.
const MaxLogTailBytes = 48 * 1024

// StaleAfter: a subscriber treats a state older than this as unknown (C1).
const StaleAfter = 90 * time.Second

// Error codes (C1).
const (
	CodeBadRequest   = "bad_request"
	CodeUnknownModel = "unknown_model"
	CodeBusy         = "busy"
	CodeNotOwner     = "not_owner"
	CodeLoadFailed   = "load_failed"
	CodeTimeout      = "timeout"
	CodeUnsupported  = "unsupported"
	CodeNoGPU        = "no_gpu"
)

// Verbs (C1).
const (
	VerbState   = "state"
	VerbStart   = "start"
	VerbStop    = "stop"
	VerbRestart = "restart"
	VerbReload  = "reload"
	VerbLoad    = "load"
	VerbUnload  = "unload"
	VerbSleep   = "sleep"
	VerbLease   = "lease"
	VerbTrain   = "train"
	VerbCancel  = "cancel"
	VerbRun     = "run"
)

// Service statuses (C1, "State").
const (
	StatusIdle      = "idle"
	StatusLoading   = "loading"
	StatusServing   = "serving"
	StatusSleeping  = "sleeping"
	StatusWaking    = "waking"
	StatusUnloading = "unloading"
	StatusFailed    = "failed"
	StatusStopped   = "stopped"
)

// Run statuses of the actions and trainer services (C1).
const (
	RunRunning  = "running"
	RunExited   = "exited"
	RunLost     = "lost"
	RunStopping = "stopping"
	RunEnded    = "ended"
)

// Containers and the services whose name decides their kind (C1).
const (
	ContainerEngines     = "engines"
	ContainerLayaTrainer = "laya-trainer"
	ContainerOperator    = "operator"

	ServiceActions = "actions"
	ServiceTrainer = "trainer"
)

// GuardEventSubject is where the prolog guard publishes its verdict (I7).
const GuardEventSubject = "devai.events.slurm.guard"

// Unload targets.
const (
	TargetActive  = "active"
	TargetSleeper = "sleeper"
)

var (
	nameRE   = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{0,62}$`)
	reasonRE = regexp.MustCompile(`^[a-z_]{1,32}$`)
)

// ValidName reports whether s may be a container or service name: one
// subject token, lower-case letters, digits and dashes.
func ValidName(s string) bool { return nameRE.MatchString(s) }

// ValidReason reports whether s is a valid cancel reason (C1: one short word).
func ValidReason(s string) bool { return reasonRE.MatchString(s) }

// CommandSubject is devai.<container>.<service>.<verb>.
func CommandSubject(container, service, verb string) string {
	return "devai." + container + "." + service + "." + verb
}

// ServiceSubject is devai.<container>.<service>, the prefix of every command
// subject of that service; devai-bus hold takes it as its argument.
func ServiceSubject(container, service string) string {
	return "devai." + container + "." + service
}

// EventSubject is devai.events.<container>.<service>.
func EventSubject(container, service string) string {
	return "devai.events." + container + "." + service
}

// ParseCommandSubject splits devai.<container>.<service>.<verb>.
func ParseCommandSubject(subject string) (container, service, verb string, err error) {
	p := strings.Split(subject, ".")
	if len(p) != 4 || p[0] != "devai" || p[1] == "events" {
		return "", "", "", fmt.Errorf("not a command subject: %q (want devai.<container>.<service>.<verb>)", subject)
	}
	if !ValidName(p[1]) || !ValidName(p[2]) || !ValidName(p[3]) {
		return "", "", "", fmt.Errorf("bad token in subject %q", subject)
	}
	return p[1], p[2], p[3], nil
}

// ParseServiceSubject splits devai.<container>.<service>.
func ParseServiceSubject(subject string) (container, service string, err error) {
	p := strings.Split(subject, ".")
	if len(p) != 3 || p[0] != "devai" || p[1] == "events" {
		return "", "", fmt.Errorf("not a service subject: %q (want devai.<container>.<service>)", subject)
	}
	if !ValidName(p[1]) || !ValidName(p[2]) {
		return "", "", fmt.Errorf("bad token in subject %q", subject)
	}
	return p[1], p[2], nil
}

// Request is a command: {"v":1, "job":<Slurm job id, 0 for none>, ...} with
// the verb's own fields (C1, "Verbs"). Fields a verb does not use stay
// unset.
type Request struct {
	V   int   `json:"v"`
	Job int64 `json:"job"`

	// state on actions and trainer is per job; Job names it.

	// load
	Launch   json.RawMessage `json:"launch,omitempty"`
	TimeoutS *int            `json:"timeout_s,omitempty"`

	// unload
	Target string `json:"target,omitempty"`
	Force  bool   `json:"force,omitempty"`

	// stop (actions)
	Signal string `json:"signal,omitempty"`
	GraceS *int   `json:"grace_s,omitempty"`
	RunJob *int64 `json:"run_job,omitempty"`

	// run (actions)
	Action string          `json:"action,omitempty"`
	Params json.RawMessage `json:"params,omitempty"`
	Kind   string          `json:"kind,omitempty"`

	// train / cancel (trainer)
	FTJob  string `json:"ft_job,omitempty"`
	Reason string `json:"reason,omitempty"`
}

// wireRequest detects a missing v or job, which Request's plain ints hide.
type wireRequest struct {
	V   *int   `json:"v"`
	Job *int64 `json:"job"`

	Launch   json.RawMessage `json:"launch,omitempty"`
	TimeoutS *int            `json:"timeout_s,omitempty"`
	Target   string          `json:"target,omitempty"`
	Force    bool            `json:"force,omitempty"`
	Signal   string          `json:"signal,omitempty"`
	GraceS   *int            `json:"grace_s,omitempty"`
	RunJob   *int64          `json:"run_job,omitempty"`
	Action   string          `json:"action,omitempty"`
	Params   json.RawMessage `json:"params,omitempty"`
	Kind     string          `json:"kind,omitempty"`
	FTJob    string          `json:"ft_job,omitempty"`
	Reason   string          `json:"reason,omitempty"`
}

// DecodeRequest parses a request strictly: unknown fields, a missing or
// wrong v, and a missing or negative job are errors (answered bad_request),
// so a misspelt field never silently falls back to a default.
func DecodeRequest(data []byte) (Request, error) {
	var w wireRequest
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&w); err != nil {
		return Request{}, fmt.Errorf("bad request body: %w", err)
	}
	if dec.More() {
		return Request{}, errors.New("bad request body: trailing data")
	}
	if w.V == nil || *w.V != Version {
		return Request{}, fmt.Errorf("bad request: v must be %d", Version)
	}
	if w.Job == nil || *w.Job < 0 {
		return Request{}, errors.New("bad request: job must be a Slurm job id, 0 for none")
	}
	return Request{
		V: *w.V, Job: *w.Job,
		Launch: w.Launch, TimeoutS: w.TimeoutS,
		Target: w.Target, Force: w.Force,
		Signal: w.Signal, GraceS: w.GraceS, RunJob: w.RunJob,
		Action: w.Action, Params: w.Params, Kind: w.Kind,
		FTJob: w.FTJob, Reason: w.Reason,
	}, nil
}

// Error is a reply's or a state's error object (C1).
type Error struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	LogTail string `json:"log_tail"`
}

func (e *Error) Error() string {
	if e == nil {
		return "<nil>"
	}
	return e.Code + ": " + e.Message
}

// Reply is {"v":1, "ok":true, "state":{...}} or
// {"v":1, "ok":false, "error":{"code","message","log_tail"}}.
type Reply struct {
	V     int    `json:"v"`
	OK    bool   `json:"ok"`
	State *State `json:"state,omitempty"`
	Error *Error `json:"error,omitempty"`
}

// OKReply wraps a state.
func OKReply(s State) Reply { return Reply{V: Version, OK: true, State: &s} }

// ErrReply builds a refusal or a failure.
func ErrReply(code, message, logTail string) Reply {
	return Reply{V: Version, OK: false, Error: &Error{Code: code, Message: message, LogTail: TrimLogTail(logTail)}}
}

// Sleeper is a sleeping model (C1, "State").
type Sleeper struct {
	Job     int64           `json:"job"`
	Launch  json.RawMessage `json:"launch"`
	Since   string          `json:"since"`
	UsedMiB int             `json:"used_mib"`
}

// Run is one run of the actions or the trainer service. The service keeps
// it until 10 minutes after its end.
type Run struct {
	Job      int64  `json:"job"`
	Status   string `json:"status"` // running, exited, lost (actions); running, stopping, ended (trainer)
	Action   string `json:"action,omitempty"`
	Kind     string `json:"kind,omitempty"`
	ExitCode *int   `json:"exit_code,omitempty"`
	FTJob    string `json:"ft_job,omitempty"`
	Outcome  string `json:"outcome,omitempty"`
	Since    string `json:"since"`
	Ended    string `json:"ended,omitempty"`
	Error    *Error `json:"error,omitempty"`
}

// State is a service's state (C1): {"container", "service", "status",
// "job", "launch", "since", "build", "sleeper", "error"}. On actions and
// trainer, `state` with a job returns that job's run in Run (null when the
// service keeps none for it), and every event carries Runs, all the runs
// the service still keeps.
type State struct {
	Container string          `json:"container"`
	Service   string          `json:"service"`
	Status    string          `json:"status"`
	Job       *int64          `json:"job"`
	Launch    json.RawMessage `json:"launch"`
	Since     string          `json:"since"`
	Build     string          `json:"build"`
	Sleeper   *Sleeper        `json:"sleeper"`
	Error     *Error          `json:"error"`

	Run  *Run  `json:"run,omitempty"`
	Runs []Run `json:"runs,omitempty"`
}

// FindRun returns the run of job in s.Runs.
func (s *State) FindRun(job int64) (Run, bool) {
	for _, r := range s.Runs {
		if r.Job == job {
			return r, true
		}
	}
	return Run{}, false
}

// JobIs reports whether the state names job as its owner.
func (s *State) JobIs(job int64) bool { return s.Job != nil && *s.Job == job }

// Timestamp formats t the way every bus time is written.
func Timestamp(t time.Time) string { return t.UTC().Format(time.RFC3339Nano) }

// TrimLogTail keeps the END of a log, which is where a failure shows, so a
// reply stays under MaxMessageBytes. The bound is on the JSON encoding, not
// the raw bytes: a control character encodes to six bytes, and <, > and &
// are escaped too.
func TrimLogTail(s string) string {
	if encodedLen(s) <= MaxLogTailBytes {
		return s
	}
	const marker = "[...]\n"
	budget := MaxLogTailBytes - len(marker)
	// Walk back from the end, rune by rune, while the encoding still fits.
	n, i := 0, len(s)
	for i > 0 {
		_, size := utf8.DecodeLastRuneInString(s[:i])
		l := encodedLen(s[i-size : i])
		if n+l > budget {
			break
		}
		n += l
		i -= size
	}
	cut := s[i:]
	if j := strings.IndexByte(cut, '\n'); j >= 0 && j < len(cut)-1 {
		cut = cut[j+1:]
	}
	return marker + cut
}

// encodedLen is the length of s as a JSON string, without its quotes.
func encodedLen(s string) int {
	b, _ := json.Marshal(s)
	return len(b) - 2
}

// Marshal encodes v and refuses a message larger than the bus allows.
func Marshal(v any) ([]byte, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	if len(b) > MaxMessageBytes {
		return nil, fmt.Errorf("message of %d bytes exceeds the bus limit of %d", len(b), MaxMessageBytes)
	}
	return b, nil
}
