package slurmrest

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

// APIVersion is the data parser devai speaks (Debian's 24.11.5 ships
// v0.0.40 to v0.0.42).
const APIVersion = "v0.0.42"

// Environment variables (C7).
const (
	EnvURL     = "SLURM_REST_URL"
	EnvKeyFile = "SLURM_JWT_KEY_FILE"
	EnvUser    = "SLURM_USER"
)

// DefaultUser is the one Slurm user every devai job runs as (I6).
const DefaultUser = "devai"

// Paths, exactly as the v0.0.42 spec spells them (some end in "/").
const (
	pathSubmit = "/slurm/" + APIVersion + "/job/submit"
	pathJob    = "/slurm/" + APIVersion + "/job/"  // + id
	pathJobs   = "/slurm/" + APIVersion + "/jobs/" //
	pathNode   = "/slurm/" + APIVersion + "/node/" // + name
	pathDBJob  = "/slurmdb/" + APIVersion + "/job/"
	pathDBJobs = "/slurmdb/" + APIVersion + "/jobs/"
)

// Client talks to slurmrestd as one Slurm user, signing a fresh token for
// every call with the shared HS256 key.
type Client struct {
	BaseURL string
	User    string
	Key     []byte
	HTTP    *http.Client
	Now     func() time.Time
}

// New returns a client; the key is the content of jwt_hs256.key.
func New(baseURL, user string, key []byte) *Client {
	return &Client{
		BaseURL: strings.TrimRight(baseURL, "/"),
		User:    user,
		Key:     key,
		HTTP:    &http.Client{Timeout: 30 * time.Second},
		Now:     time.Now,
	}
}

// FromEnv reads SLURM_REST_URL, SLURM_JWT_KEY_FILE and SLURM_USER (default
// devai).
func FromEnv() (*Client, error) {
	base := os.Getenv(EnvURL)
	keyFile := os.Getenv(EnvKeyFile)
	if base == "" || keyFile == "" {
		return nil, fmt.Errorf("slurm REST configuration incomplete: %s and %s must be set", EnvURL, EnvKeyFile)
	}
	key, err := os.ReadFile(keyFile)
	if err != nil {
		return nil, fmt.Errorf("read the JWT key: %w", err)
	}
	if len(key) == 0 {
		return nil, fmt.Errorf("the JWT key %s is empty", keyFile)
	}
	user := os.Getenv(EnvUser)
	if user == "" {
		user = DefaultUser
	}
	return New(base, user, key), nil
}

// Error is a failed call: a non-2xx status or a non-empty `errors` list.
type Error struct {
	Method, Path string
	Status       int
	Errors       []APIError
}

func (e *Error) Error() string {
	var parts []string
	for _, a := range e.Errors {
		s := a.Error
		if a.Description != "" {
			s += " (" + a.Description + ")"
		}
		parts = append(parts, s)
	}
	msg := strings.Join(parts, "; ")
	if msg == "" {
		msg = http.StatusText(e.Status)
	}
	return fmt.Sprintf("slurmrestd %s %s: HTTP %d: %s", e.Method, e.Path, e.Status, msg)
}

// IsInvalidJob reports whether err says the job id is unknown to slurmctld
// (a job that ended long ago is only in slurmdbd).
func IsInvalidJob(err error) bool {
	var e *Error
	if !errors.As(err, &e) {
		return false
	}
	for _, a := range e.Errors {
		if a.ErrorNumber == 2017 || strings.Contains(strings.ToLower(a.Error+a.Description), "invalid job id") {
			return true
		}
	}
	return false
}

func (c *Client) do(ctx context.Context, method, path string, query url.Values, body, out any) error {
	token, err := SignJWT(c.Key, c.User, c.Now(), TokenTTL)
	if err != nil {
		return err
	}
	u := c.BaseURL + path
	if len(query) > 0 {
		u += "?" + query.Encode()
	}
	var rd io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, u, rd)
	if err != nil {
		return err
	}
	req.Header.Set("X-SLURM-USER-NAME", c.User)
	req.Header.Set("X-SLURM-USER-TOKEN", token)
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return fmt.Errorf("slurmrestd %s %s: %w", method, path, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 64<<20))
	if err != nil {
		return fmt.Errorf("slurmrestd %s %s: read: %w", method, path, err)
	}
	var env response
	_ = json.Unmarshal(data, &env)
	if resp.StatusCode/100 != 2 || len(env.Errors) > 0 {
		return &Error{Method: method, Path: path, Status: resp.StatusCode, Errors: env.Errors}
	}
	if out != nil {
		if err := json.Unmarshal(data, out); err != nil {
			return fmt.Errorf("slurmrestd %s %s: decode: %w", method, path, err)
		}
	}
	return nil
}

func jobPath(id int64) string { return pathJob + strconv.FormatInt(id, 10) }

// Submit sends one batch job (POST /slurm/v0.0.42/job/submit, the
// description under "job") and returns its id.
func (c *Client) Submit(ctx context.Context, d JobDesc) (int64, error) {
	if d.Environment == nil {
		d.Environment = []string{}
	}
	var out struct {
		JobID int64 `json:"job_id"`
	}
	if err := c.do(ctx, http.MethodPost, pathSubmit, nil, map[string]any{"job": d}, &out); err != nil {
		return 0, err
	}
	if out.JobID <= 0 {
		return 0, errors.New("slurmrestd: submit returned no job id")
	}
	return out.JobID, nil
}

// Job reads one job from slurmctld.
func (c *Client) Job(ctx context.Context, id int64) (JobInfo, error) {
	var out struct {
		Jobs []JobInfo `json:"jobs"`
	}
	if err := c.do(ctx, http.MethodGet, jobPath(id), nil, nil, &out); err != nil {
		return JobInfo{}, err
	}
	for _, j := range out.Jobs {
		if j.JobID == id {
			return j, nil
		}
	}
	return JobInfo{}, &Error{Method: http.MethodGet, Path: jobPath(id), Status: http.StatusNotFound,
		Errors: []APIError{{Error: "Invalid job id specified", ErrorNumber: 2017}}}
}

// Jobs lists every job slurmctld still knows (the queue and recent ends).
func (c *Client) Jobs(ctx context.Context) ([]JobInfo, error) {
	var out struct {
		Jobs []JobInfo `json:"jobs"`
	}
	if err := c.do(ctx, http.MethodGet, pathJobs, nil, nil, &out); err != nil {
		return nil, err
	}
	return out.Jobs, nil
}

// Signal sends a signal (e.g. "SIGUSR1"); batchOnly signals the batch
// script alone (flags=BATCH_JOB, `scancel --signal=<s> --batch`).
func (c *Client) Signal(ctx context.Context, id int64, signal string, batchOnly bool) error {
	q := url.Values{"signal": {signal}}
	if batchOnly {
		q.Set("flags", "BATCH_JOB")
	}
	return c.do(ctx, http.MethodDelete, jobPath(id), q, nil, nil)
}

// Cancel cancels a job (`scancel`).
func (c *Client) Cancel(ctx context.Context, id int64) error {
	return c.do(ctx, http.MethodDelete, jobPath(id), nil, nil, nil)
}

// Hold holds a pending job; Release releases it (POST
// /slurm/v0.0.42/job/{id} with "hold"). devai has AdminLevel=Operator (I6).
func (c *Client) Hold(ctx context.Context, id int64) error {
	return c.do(ctx, http.MethodPost, jobPath(id), nil, map[string]any{"hold": true}, nil)
}

// Release releases a held job.
func (c *Client) Release(ctx context.Context, id int64) error {
	return c.do(ctx, http.MethodPost, jobPath(id), nil, map[string]any{"hold": false}, nil)
}

// Node reads one node.
func (c *Client) Node(ctx context.Context, name string) (NodeInfo, error) {
	var out struct {
		Nodes []NodeInfo `json:"nodes"`
	}
	p := pathNode + url.PathEscape(name)
	if err := c.do(ctx, http.MethodGet, p, nil, nil, &out); err != nil {
		return NodeInfo{}, err
	}
	for _, n := range out.Nodes {
		if n.Name == name {
			return n, nil
		}
	}
	return NodeInfo{}, fmt.Errorf("slurmrestd: node %q not found", name)
}

// ResumeNode sets a node's state to RESUME.
func (c *Client) ResumeNode(ctx context.Context, name string) error {
	return c.do(ctx, http.MethodPost, pathNode+url.PathEscape(name), nil,
		map[string]any{"state": []string{"RESUME"}}, nil)
}

// DBQuery filters slurmdbd's job history. Times are what sacct accepts
// (e.g. a UNIX time or "now-1days").
type DBQuery struct {
	Users     string
	StartTime string
	EndTime   string
	State     string
	JobName   string
}

func (q DBQuery) values() url.Values {
	v := url.Values{}
	set := func(k, s string) {
		if s != "" {
			v.Set(k, s)
		}
	}
	set("users", q.Users)
	set("start_time", q.StartTime)
	set("end_time", q.EndTime)
	set("state", q.State)
	set("job_name", q.JobName)
	return v
}

// DBJobs lists jobs from slurmdbd.
func (c *Client) DBJobs(ctx context.Context, q DBQuery) ([]DBJob, error) {
	var out struct {
		Jobs []DBJob `json:"jobs"`
	}
	if err := c.do(ctx, http.MethodGet, pathDBJobs, q.values(), nil, &out); err != nil {
		return nil, err
	}
	return out.Jobs, nil
}

// DBJob reads one job's record from slurmdbd.
func (c *Client) DBJob(ctx context.Context, id int64) (DBJob, error) {
	var out struct {
		Jobs []DBJob `json:"jobs"`
	}
	p := pathDBJob + strconv.FormatInt(id, 10)
	if err := c.do(ctx, http.MethodGet, p, nil, nil, &out); err != nil {
		return DBJob{}, err
	}
	for _, j := range out.Jobs {
		if j.JobID == id {
			return j, nil
		}
	}
	return DBJob{}, fmt.Errorf("slurmdbd: no record of job %d", id)
}
