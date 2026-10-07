package control

import (
	"bytes"
	"context"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Supervisor is supervisord's XML-RPC interface (supervisord.org/api.html)
// over its unix socket: the calls devai-control needs.
type Supervisor struct {
	http *http.Client
	url  string
}

// NewSupervisor talks to supervisord on a unix socket.
func NewSupervisor(socket string) *Supervisor {
	tr := &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			var d net.Dialer
			return d.DialContext(ctx, "unix", socket)
		},
	}
	// stopProcess with wait=true returns only once the program has stopped
	// (stopwaitsecs, then SIGKILL): the timeout leaves room for that.
	return &Supervisor{http: &http.Client{Transport: tr, Timeout: 120 * time.Second}, url: "http://supervisord/RPC2"}
}

// Fault is an XML-RPC fault, e.g. BAD_NAME (10), NOT_RUNNING (70),
// ALREADY_STARTED (60), SPAWN_ERROR (50), ALREADY_ADDED (90),
// STILL_RUNNING (91).
type Fault struct {
	Code   int
	String string
}

func (f *Fault) Error() string { return fmt.Sprintf("supervisord fault %d: %s", f.Code, f.String) }

// Fault codes devai-control acts on.
const (
	FaultBadName        = 10
	FaultAbnormal       = 40
	FaultSpawnError     = 50
	FaultAlreadyStarted = 60
	FaultNotRunning     = 70
	FaultAlreadyAdded   = 90
	FaultStillRunning   = 91
)

// IsFault reports whether err is a supervisord fault with code.
func IsFault(err error, code int) bool {
	var f *Fault
	return errors.As(err, &f) && f.Code == code
}

// ProcessInfo is getProcessInfo's struct.
type ProcessInfo struct {
	Name       string
	Group      string
	StateName  string // STOPPED STARTING RUNNING BACKOFF STOPPING EXITED FATAL UNKNOWN
	PID        int
	ExitStatus int
	Start      int64
	Stop       int64
	SpawnErr   string
	Stdout     string
}

// Running reports whether the program is starting or running.
func (p ProcessInfo) Running() bool { return p.StateName == "RUNNING" || p.StateName == "STARTING" }

// --- XML-RPC encoding -------------------------------------------------------

func xmlEscape(s string) string {
	var b bytes.Buffer
	_ = xml.EscapeText(&b, []byte(s))
	return b.String()
}

func encodeValue(v any) string {
	switch x := v.(type) {
	case string:
		return "<value><string>" + xmlEscape(x) + "</string></value>"
	case bool:
		if x {
			return "<value><boolean>1</boolean></value>"
		}
		return "<value><boolean>0</boolean></value>"
	case int:
		return "<value><int>" + strconv.Itoa(x) + "</int></value>"
	}
	panic(fmt.Sprintf("xmlrpc: cannot encode %T", v))
}

func encodeCall(method string, params ...any) []byte {
	var b strings.Builder
	b.WriteString(`<?xml version="1.0"?><methodCall><methodName>` + xmlEscape(method) + `</methodName><params>`)
	for _, p := range params {
		b.WriteString("<param>" + encodeValue(p) + "</param>")
	}
	b.WriteString("</params></methodCall>")
	return []byte(b.String())
}

// decodeValue turns a <value> element into Go: struct -> map[string]any,
// array -> []any, int/i4 -> int, boolean -> bool, string -> string,
// double -> float64.
func decodeValue(dec *xml.Decoder, start xml.StartElement) (any, error) {
	// start is <value>; read its single typed child or bare text.
	var text strings.Builder
	for {
		tok, err := dec.Token()
		if err != nil {
			return nil, err
		}
		switch t := tok.(type) {
		case xml.CharData:
			text.Write(t)
		case xml.EndElement:
			return text.String(), nil // <value>text</value> is a string
		case xml.StartElement:
			v, err := decodeTyped(dec, t)
			if err != nil {
				return nil, err
			}
			// consume up to </value>
			if err := skipTo(dec, start.Name.Local); err != nil {
				return nil, err
			}
			return v, nil
		}
	}
}

func skipTo(dec *xml.Decoder, name string) error {
	depth := 0
	for {
		tok, err := dec.Token()
		if err != nil {
			return err
		}
		switch t := tok.(type) {
		case xml.StartElement:
			depth++
		case xml.EndElement:
			if depth == 0 && t.Name.Local == name {
				return nil
			}
			depth--
		}
	}
}

func readText(dec *xml.Decoder, name string) (string, error) {
	var b strings.Builder
	for {
		tok, err := dec.Token()
		if err != nil {
			return "", err
		}
		switch t := tok.(type) {
		case xml.CharData:
			b.Write(t)
		case xml.EndElement:
			if t.Name.Local == name {
				return b.String(), nil
			}
		}
	}
}

func decodeTyped(dec *xml.Decoder, t xml.StartElement) (any, error) {
	switch t.Name.Local {
	case "string":
		return readText(dec, "string")
	case "int", "i4", "i8":
		s, err := readText(dec, t.Name.Local)
		if err != nil {
			return nil, err
		}
		n, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64)
		return int(n), err
	case "boolean":
		s, err := readText(dec, "boolean")
		return strings.TrimSpace(s) == "1", err
	case "double":
		s, err := readText(dec, "double")
		if err != nil {
			return nil, err
		}
		return strconv.ParseFloat(strings.TrimSpace(s), 64)
	case "nil":
		return nil, skipTo(dec, "nil")
	case "array":
		var out []any
		for {
			tok, err := dec.Token()
			if err != nil {
				return nil, err
			}
			switch x := tok.(type) {
			case xml.StartElement:
				if x.Name.Local == "value" {
					v, err := decodeValue(dec, x)
					if err != nil {
						return nil, err
					}
					out = append(out, v)
				}
			case xml.EndElement:
				if x.Name.Local == "array" {
					return out, nil
				}
			}
		}
	case "struct":
		out := map[string]any{}
		var name string
		for {
			tok, err := dec.Token()
			if err != nil {
				return nil, err
			}
			switch x := tok.(type) {
			case xml.StartElement:
				switch x.Name.Local {
				case "name":
					if name, err = readText(dec, "name"); err != nil {
						return nil, err
					}
				case "value":
					v, err := decodeValue(dec, x)
					if err != nil {
						return nil, err
					}
					out[name] = v
				}
			case xml.EndElement:
				if x.Name.Local == "struct" {
					return out, nil
				}
			}
		}
	}
	return nil, fmt.Errorf("xmlrpc: unsupported type <%s>", t.Name.Local)
}

// decodeResponse returns the result value or a *Fault.
func decodeResponse(r io.Reader) (any, error) {
	dec := xml.NewDecoder(r)
	fault := false
	for {
		tok, err := dec.Token()
		if err != nil {
			return nil, fmt.Errorf("xmlrpc: %w", err)
		}
		t, ok := tok.(xml.StartElement)
		if !ok {
			continue
		}
		switch t.Name.Local {
		case "fault":
			fault = true
		case "value":
			v, err := decodeValue(dec, t)
			if err != nil {
				return nil, fmt.Errorf("xmlrpc: %w", err)
			}
			if fault {
				m, _ := v.(map[string]any)
				code, _ := m["faultCode"].(int)
				s, _ := m["faultString"].(string)
				return nil, &Fault{Code: code, String: s}
			}
			return v, nil
		}
	}
}

func (s *Supervisor) call(ctx context.Context, method string, params ...any) (any, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.url, bytes.NewReader(encodeCall(method, params...)))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "text/xml")
	resp, err := s.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("supervisord %s: %w", method, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("supervisord %s: HTTP %d", method, resp.StatusCode)
	}
	return decodeResponse(resp.Body)
}

func asInt(v any) int {
	n, _ := v.(int)
	return n
}

func asString(v any) string {
	s, _ := v.(string)
	return s
}

// GetProcessInfo is supervisor.getProcessInfo.
func (s *Supervisor) GetProcessInfo(ctx context.Context, name string) (ProcessInfo, error) {
	v, err := s.call(ctx, "supervisor.getProcessInfo", name)
	if err != nil {
		return ProcessInfo{}, err
	}
	m, _ := v.(map[string]any)
	return ProcessInfo{
		Name: asString(m["name"]), Group: asString(m["group"]), StateName: asString(m["statename"]),
		PID: asInt(m["pid"]), ExitStatus: asInt(m["exitstatus"]), Start: int64(asInt(m["start"])),
		Stop: int64(asInt(m["stop"])), SpawnErr: asString(m["spawnerr"]), Stdout: asString(m["stdout_logfile"]),
	}, nil
}

func (s *Supervisor) boolCall(ctx context.Context, method string, params ...any) error {
	v, err := s.call(ctx, method, params...)
	if err != nil {
		return err
	}
	if b, ok := v.(bool); ok && !b {
		return fmt.Errorf("supervisord %s returned false", method)
	}
	return nil
}

// StartProcess is supervisor.startProcess(name, wait).
func (s *Supervisor) StartProcess(ctx context.Context, name string, wait bool) error {
	return s.boolCall(ctx, "supervisor.startProcess", name, wait)
}

// StopProcess is supervisor.stopProcess(name, wait).
func (s *Supervisor) StopProcess(ctx context.Context, name string, wait bool) error {
	return s.boolCall(ctx, "supervisor.stopProcess", name, wait)
}

// SignalProcess is supervisor.signalProcess(name, signal).
func (s *Supervisor) SignalProcess(ctx context.Context, name, signal string) error {
	return s.boolCall(ctx, "supervisor.signalProcess", name, signal)
}

// ReloadConfig is supervisor.reloadConfig: the added, changed and removed
// process groups.
func (s *Supervisor) ReloadConfig(ctx context.Context) (added, changed, removed []string, err error) {
	v, err := s.call(ctx, "supervisor.reloadConfig")
	if err != nil {
		return nil, nil, nil, err
	}
	outer, _ := v.([]any)
	if len(outer) == 0 {
		return nil, nil, nil, nil
	}
	lists, _ := outer[0].([]any)
	get := func(i int) []string {
		if i >= len(lists) {
			return nil
		}
		raw, _ := lists[i].([]any)
		var out []string
		for _, x := range raw {
			out = append(out, asString(x))
		}
		return out
	}
	return get(0), get(1), get(2), nil
}

// AddProcessGroup is supervisor.addProcessGroup.
func (s *Supervisor) AddProcessGroup(ctx context.Context, name string) error {
	return s.boolCall(ctx, "supervisor.addProcessGroup", name)
}

// RemoveProcessGroup is supervisor.removeProcessGroup.
func (s *Supervisor) RemoveProcessGroup(ctx context.Context, name string) error {
	return s.boolCall(ctx, "supervisor.removeProcessGroup", name)
}

// Update applies the program files on disk to supervisord the way
// `supervisorctl update` does: reloadConfig, then remove and re-add every
// changed group, add every new one. It returns the groups it touched.
func (s *Supervisor) Update(ctx context.Context) ([]string, error) {
	added, changed, removed, err := s.ReloadConfig(ctx)
	if err != nil {
		return nil, err
	}
	var touched []string
	for _, g := range removed {
		_ = s.StopProcess(ctx, g, true)
		if err := s.RemoveProcessGroup(ctx, g); err != nil && !IsFault(err, FaultBadName) {
			return touched, err
		}
		touched = append(touched, g)
	}
	for _, g := range changed {
		_ = s.StopProcess(ctx, g, true)
		if err := s.RemoveProcessGroup(ctx, g); err != nil && !IsFault(err, FaultBadName) {
			return touched, err
		}
		if err := s.AddProcessGroup(ctx, g); err != nil && !IsFault(err, FaultAlreadyAdded) {
			return touched, err
		}
		touched = append(touched, g)
	}
	for _, g := range added {
		if err := s.AddProcessGroup(ctx, g); err != nil && !IsFault(err, FaultAlreadyAdded) {
			return touched, err
		}
		touched = append(touched, g)
	}
	return touched, nil
}
