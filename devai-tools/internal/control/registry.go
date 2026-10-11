package control

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path"
	"regexp"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// Registry is the operator's action registry, deploy/operator/actions.yaml
// (C8). devai-control (kind actions) and the web app both read it and check
// every parameter against it; nothing runs through a shell.
type Registry struct {
	Version    int          `yaml:"version"`
	Actions    []Action     `yaml:"actions"`
	NotOffered []NotOffered `yaml:"not_offered"`
	Pending    []string     `yaml:"pending"`

	byID map[string]*Action
}

// Action is one registry entry.
type Action struct {
	ID           string       `yaml:"id"`
	Group        string       `yaml:"group"`
	Title        string       `yaml:"title"`
	Kind         string       `yaml:"kind"` // job | direct | host
	Script       string       `yaml:"script"`
	Args         []string     `yaml:"args"`
	Params       []Param      `yaml:"params"`
	JobKind      string       `yaml:"job_kind"` // action | probe | bench
	TimeLimitMin *int         `yaml:"time_limit_min"`
	GPU          bool         `yaml:"gpu"`
	Comment      *CommentKeys `yaml:"comment"`
	Confirm      bool         `yaml:"confirm"`
	Covers       []string     `yaml:"covers"`
	Help         string       `yaml:"help"`
}

// Param is one declared parameter.
type Param struct {
	Name     string   `yaml:"name"`
	Type     string   `yaml:"type"` // string | int | bool | enum
	Pattern  string   `yaml:"pattern"`
	Values   []string `yaml:"values"`
	Required bool     `yaml:"required"`
	To       ParamTo  `yaml:"to"`
	Help     string   `yaml:"help"`

	re *regexp.Regexp
}

// ParamTo maps a parameter to a flag or to an environment variable.
type ParamTo struct {
	Flag string `yaml:"flag"`
	Env  string `yaml:"env"`
}

// CommentKeys fills C3's base keys for a run that takes a C5 hold: backend
// is a literal, model/ctx/mtp name parameters.
type CommentKeys struct {
	Backend string `yaml:"backend"`
	Model   string `yaml:"model"`
	Ctx     string `yaml:"ctx"`
	MTP     string `yaml:"mtp"`
}

// NotOffered is a Makefile target with no action, and why.
type NotOffered struct {
	Target string `yaml:"target"`
	Reason string `yaml:"reason"`
}

var (
	actionIDRE  = regexp.MustCompile(`^[a-z0-9-]+$`)
	paramNameRE = regexp.MustCompile(`^[a-z][a-z0-9_]{0,63}$`)
	flagRE      = regexp.MustCompile(`^--?[A-Za-z0-9][A-Za-z0-9-]*$`)
	envNameRE   = regexp.MustCompile(`^[A-Z_][A-Z0-9_]*$`)
	groups      = map[string]bool{"stack": true, "build": true, "models": true, "probe": true, "bench": true,
		"test": true, "backup": true, "logs": true, "lab": true, "host": true, "library": true, "maintenance": true}
)

// LoadRegistry reads and validates a registry file.
func LoadRegistry(file string) (*Registry, error) {
	data, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}
	return ParseRegistry(data)
}

// ParseRegistry decodes strictly and validates C8's rules: job_kind is
// required on every job entry and is action, probe or bench; gpu and
// comment only with job_kind action; a gpu entry needs time_limit_min; a
// comment names declared parameters.
func ParseRegistry(data []byte) (*Registry, error) {
	var r Registry
	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)
	if err := dec.Decode(&r); err != nil {
		return nil, fmt.Errorf("registry: %w", err)
	}
	if r.Version != 1 {
		return nil, fmt.Errorf("registry: version %d, want 1", r.Version)
	}
	r.byID = map[string]*Action{}
	var errs []error
	for i := range r.Actions {
		a := &r.Actions[i]
		if err := a.validate(); err != nil {
			errs = append(errs, fmt.Errorf("action %q: %w", a.ID, err))
			continue
		}
		if _, dup := r.byID[a.ID]; dup {
			errs = append(errs, fmt.Errorf("action %q: duplicate id", a.ID))
		}
		r.byID[a.ID] = a
	}
	if err := errors.Join(errs...); err != nil {
		return nil, fmt.Errorf("registry: %w", err)
	}
	return &r, nil
}

// Lookup returns an action by id.
func (r *Registry) Lookup(id string) (*Action, bool) {
	a, ok := r.byID[id]
	return a, ok
}

func (a *Action) validate() error {
	if !actionIDRE.MatchString(a.ID) {
		return errors.New("id must match [a-z0-9-]+")
	}
	if !groups[a.Group] {
		return fmt.Errorf("group %q is not one of the registry's groups", a.Group)
	}
	switch a.Kind {
	case "job", "direct", "host":
	default:
		return fmt.Errorf("kind %q: want job, direct or host", a.Kind)
	}
	if a.Script == "" || path.IsAbs(a.Script) || strings.Contains(a.Script, "..") || path.Clean(a.Script) != a.Script {
		return fmt.Errorf("script %q must be a clean repository-relative path", a.Script)
	}
	declared := map[string]bool{}
	for i := range a.Params {
		p := &a.Params[i]
		if err := p.validate(); err != nil {
			return fmt.Errorf("param %q: %w", p.Name, err)
		}
		if declared[p.Name] {
			return fmt.Errorf("param %q declared twice", p.Name)
		}
		declared[p.Name] = true
	}
	if a.Kind != "job" {
		if a.JobKind != "" || a.TimeLimitMin != nil || a.GPU || a.Comment != nil {
			return errors.New("job_kind, time_limit_min, gpu and comment belong to job entries only")
		}
		return nil
	}
	switch a.JobKind {
	case "action", "probe", "bench":
	case "":
		return errors.New("job_kind is required on a job entry (action, probe or bench)")
	default:
		return fmt.Errorf("job_kind %q: want action, probe or bench", a.JobKind)
	}
	if a.TimeLimitMin != nil && *a.TimeLimitMin <= 0 {
		return errors.New("time_limit_min must be positive")
	}
	if a.GPU && a.JobKind != "action" {
		return errors.New("gpu is allowed only with job_kind action")
	}
	if a.GPU && a.TimeLimitMin == nil {
		return errors.New("a gpu action needs time_limit_min")
	}
	if a.Comment != nil {
		if a.JobKind != "action" {
			return errors.New("comment is allowed only with job_kind action")
		}
		if a.Comment.Backend == "" || a.Comment.Model == "" {
			return errors.New("comment needs backend and model")
		}
		for _, n := range []string{a.Comment.Model, a.Comment.Ctx, a.Comment.MTP} {
			if n != "" && !declared[n] {
				return fmt.Errorf("comment names undeclared parameter %q", n)
			}
		}
	}
	return nil
}

func (p *Param) validate() error {
	if !paramNameRE.MatchString(p.Name) {
		return errors.New("name must match [a-z][a-z0-9_]*")
	}
	switch p.Type {
	case "string":
		if p.Pattern == "" {
			return errors.New("a string parameter needs a pattern")
		}
	case "int", "bool":
	case "enum":
		if len(p.Values) == 0 {
			return errors.New("an enum parameter needs values")
		}
	default:
		return fmt.Errorf("type %q: want string, int, bool or enum", p.Type)
	}
	if p.Pattern != "" {
		re, err := regexp.Compile(p.Pattern)
		if err != nil {
			return fmt.Errorf("pattern: %w", err)
		}
		p.re = re
	}
	switch {
	case p.To.Flag != "" && p.To.Env != "":
		return errors.New("to: a flag or an env, not both")
	case p.To.Flag != "":
		if !flagRE.MatchString(p.To.Flag) {
			return fmt.Errorf("to.flag %q is not a flag", p.To.Flag)
		}
	case p.To.Env != "":
		if !envNameRE.MatchString(p.To.Env) {
			return fmt.Errorf("to.env %q is not a variable name", p.To.Env)
		}
	default:
		return errors.New("to: needs flag or env")
	}
	return nil
}

// Mapped is a run's checked parameters: arguments and environment.
type Mapped struct {
	Args []string
	Env  map[string]string
}

// MapParams checks params (a JSON object) against the entry and maps them:
// an unknown parameter, a missing required one, a wrong type or a value
// that breaks its pattern or enum refuses the run.
func (a *Action) MapParams(raw json.RawMessage) (Mapped, error) {
	m := Mapped{Env: map[string]string{}}
	given := map[string]json.RawMessage{}
	if len(raw) > 0 && string(raw) != "null" {
		dec := json.NewDecoder(bytes.NewReader(raw))
		dec.UseNumber()
		if err := dec.Decode(&given); err != nil {
			return m, fmt.Errorf("params must be a JSON object: %w", err)
		}
	}
	known := map[string]bool{}
	for _, p := range a.Params {
		known[p.Name] = true
	}
	for k := range given {
		if !known[k] {
			return m, fmt.Errorf("unknown parameter %q", k)
		}
	}
	for _, p := range a.Params {
		v, ok := given[p.Name]
		if !ok || string(v) == "null" {
			if p.Required {
				return m, fmt.Errorf("parameter %q is required", p.Name)
			}
			continue
		}
		var text string
		var flagOnly, present bool
		switch p.Type {
		case "string", "enum":
			if err := json.Unmarshal(v, &text); err != nil {
				return m, fmt.Errorf("parameter %q must be a string", p.Name)
			}
			if p.Type == "enum" {
				found := false
				for _, x := range p.Values {
					if x == text {
						found = true
					}
				}
				if !found {
					return m, fmt.Errorf("parameter %q: %q is not one of %v", p.Name, text, p.Values)
				}
			}
			if p.re != nil && !p.re.MatchString(text) {
				return m, fmt.Errorf("parameter %q: %q does not match %s", p.Name, text, p.Pattern)
			}
		case "int":
			var n json.Number
			if err := json.Unmarshal(v, &n); err != nil {
				return m, fmt.Errorf("parameter %q must be an integer", p.Name)
			}
			i, err := strconv.ParseInt(n.String(), 10, 64)
			if err != nil {
				return m, fmt.Errorf("parameter %q must be an integer", p.Name)
			}
			text = strconv.FormatInt(i, 10)
			if p.re != nil && !p.re.MatchString(text) {
				return m, fmt.Errorf("parameter %q: %s does not match %s", p.Name, text, p.Pattern)
			}
		case "bool":
			var b bool
			if err := json.Unmarshal(v, &b); err != nil {
				return m, fmt.Errorf("parameter %q must be true or false", p.Name)
			}
			flagOnly, present = true, b
			text = strconv.FormatBool(b)
		}
		if p.To.Env != "" {
			m.Env[p.To.Env] = text
			continue
		}
		if flagOnly {
			if present {
				m.Args = append(m.Args, p.To.Flag)
			}
			continue
		}
		m.Args = append(m.Args, p.To.Flag, text)
	}
	return m, nil
}
