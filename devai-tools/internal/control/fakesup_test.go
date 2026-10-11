package control

import (
	"bufio"
	"context"
	"encoding/xml"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeSupervisord serves supervisord's XML-RPC calls on a unix socket. Its
// "processes" are simulated: a fake engine program (command `... sleep N &&
// exec python3 <script> <MiB>`) prints `ready <MiB>` into its stdout log
// after N seconds and then holds MiB of the fake card until stopped.
type fakeSupervisord struct {
	t       *testing.T
	socket  string
	progDir string
	mem     *fakeMemory

	mu    sync.Mutex
	progs map[string]*fakeProg
	calls []string
	// stopDelay makes stopProcess take this long (an engine slow to stop).
	stopDelay time.Duration
}

type fakeProg struct {
	name    string
	command string
	logfile string
	state   string // STOPPED RUNNING EXITED
	pid     int
	exit    int
	held    int
	gen     int
}

type fakeMemory struct {
	mu    sync.Mutex
	used  int
	delay time.Duration // a slow nvidia-smi
}

func (m *fakeMemory) Present() bool { return true }
func (m *fakeMemory) UsedMiB(context.Context) (int, error) {
	m.mu.Lock()
	d := m.delay
	m.mu.Unlock()
	time.Sleep(d)
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.used, nil
}
func (m *fakeMemory) add(n int) {
	m.mu.Lock()
	m.used += n
	m.mu.Unlock()
}
func (m *fakeMemory) get() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.used
}

func newFakeSupervisord(t *testing.T, progDir string, mem *fakeMemory) *fakeSupervisord {
	t.Helper()
	dir, err := os.MkdirTemp("", "fsup")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	f := &fakeSupervisord{t: t, socket: filepath.Join(dir, "s.sock"), progDir: progDir, mem: mem, progs: map[string]*fakeProg{}}
	l, err := net.Listen("unix", f.socket)
	if err != nil {
		t.Fatal(err)
	}
	srv := &http.Server{Handler: f}
	go srv.Serve(l)
	t.Cleanup(func() { srv.Close() })
	return f
}

// addProgram registers a static program (kind program).
func (f *fakeSupervisord) addProgram(name, state string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.progs[name] = &fakeProg{name: name, state: state}
}

func (f *fakeSupervisord) prog(name string) *fakeProg {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.progs[name]
}

func (f *fakeSupervisord) state(name string) string {
	f.mu.Lock()
	defer f.mu.Unlock()
	if p := f.progs[name]; p != nil {
		return p.state
	}
	return ""
}

// crash makes a running program exit on its own.
func (f *fakeSupervisord) crash(name string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if p := f.progs[name]; p != nil && p.state == "RUNNING" {
		f.stopLocked(p, "EXITED", 1)
	}
}

func (f *fakeSupervisord) stopLocked(p *fakeProg, state string, exit int) {
	p.gen++
	if p.held > 0 {
		f.mem.add(-p.held)
		p.held = 0
	}
	p.state, p.exit = state, exit
}

var (
	sleepRE = regexp.MustCompile(`sleep ([0-9.]+) &&`)
	mibRE   = regexp.MustCompile(`([0-9]+)"?\s*$`)
)

func (f *fakeSupervisord) startLocked(p *fakeProg) {
	p.gen++
	gen := p.gen
	p.state, p.pid = "RUNNING", 10000+gen
	delay := 0.0
	if m := sleepRE.FindStringSubmatch(p.command); m != nil {
		delay, _ = strconv.ParseFloat(m[1], 64)
	}
	mib := 0
	if m := mibRE.FindStringSubmatch(p.command); m != nil {
		mib, _ = strconv.Atoi(m[1])
	}
	if p.logfile == "" {
		return
	}
	go func() {
		time.Sleep(time.Duration(delay * float64(time.Second)))
		f.mu.Lock()
		defer f.mu.Unlock()
		if p.gen != gen || p.state != "RUNNING" {
			return
		}
		p.held = mib
		f.mem.add(mib)
		lf, err := os.OpenFile(p.logfile, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o644)
		if err == nil {
			fmt.Fprintf(lf, "ready %d\n", mib)
			lf.Close()
		}
	}()
}

// readConfs parses [program:x] command= and stdout_logfile= from progDir.
func (f *fakeSupervisord) readConfs() map[string][2]string {
	out := map[string][2]string{}
	files, _ := filepath.Glob(filepath.Join(f.progDir, "*.conf"))
	for _, file := range files {
		fh, err := os.Open(file)
		if err != nil {
			continue
		}
		var name, cmd, logf string
		sc := bufio.NewScanner(fh)
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			switch {
			case strings.HasPrefix(line, "[program:"):
				name = strings.TrimSuffix(strings.TrimPrefix(line, "[program:"), "]")
			case strings.HasPrefix(line, "command="):
				cmd = strings.TrimPrefix(line, "command=")
			case strings.HasPrefix(line, "stdout_logfile="):
				logf = strings.TrimPrefix(line, "stdout_logfile=")
			}
		}
		fh.Close()
		if name != "" {
			out[name] = [2]string{cmd, logf}
		}
	}
	return out
}

type xmlCall struct {
	Method string `xml:"methodName"`
	Params []struct {
		Inner []byte `xml:",innerxml"`
	} `xml:"params>param"`
}

func (f *fakeSupervisord) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var call xmlCall
	if err := xml.NewDecoder(r.Body).Decode(&call); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	var args []any
	for _, p := range call.Params {
		dec := xml.NewDecoder(strings.NewReader(string(p.Inner)))
		for {
			tok, err := dec.Token()
			if err != nil {
				break
			}
			if st, ok := tok.(xml.StartElement); ok && st.Name.Local == "value" {
				v, err := decodeValue(dec, st)
				if err == nil {
					args = append(args, v)
				}
				break
			}
		}
	}
	f.mu.Lock()
	f.calls = append(f.calls, call.Method)
	delay := f.stopDelay
	f.mu.Unlock()
	if call.Method == "supervisor.stopProcess" && delay > 0 {
		time.Sleep(delay)
	}
	w.Header().Set("Content-Type", "text/xml")
	fmt.Fprint(w, f.dispatch(call.Method, args))
}

func fault(code int, s string) string {
	return fmt.Sprintf(`<?xml version="1.0"?><methodResponse><fault><value><struct>`+
		`<member><name>faultCode</name><value><int>%d</int></value></member>`+
		`<member><name>faultString</name><value><string>%s</string></value></member>`+
		`</struct></value></fault></methodResponse>`, code, s)
}

func ok(value string) string {
	return `<?xml version="1.0"?><methodResponse><params><param>` + value + `</param></params></methodResponse>`
}

func xstr(s string) string { return "<value><string>" + xmlEscape(s) + "</string></value>" }
func xint(n int) string    { return fmt.Sprintf("<value><int>%d</int></value>", n) }
func xtrue() string        { return "<value><boolean>1</boolean></value>" }
func xarray(vals ...string) string {
	return "<value><array><data>" + strings.Join(vals, "") + "</data></array></value>"
}

func (f *fakeSupervisord) dispatch(method string, args []any) string {
	f.mu.Lock()
	defer f.mu.Unlock()
	name := ""
	if len(args) > 0 {
		name, _ = args[0].(string)
	}
	p := f.progs[name]
	switch method {
	case "supervisor.getProcessInfo":
		if p == nil {
			return fault(FaultBadName, "BAD_NAME: "+name)
		}
		members := []string{
			"<member><name>name</name>" + xstr(p.name) + "</member>",
			"<member><name>group</name>" + xstr(p.name) + "</member>",
			"<member><name>statename</name>" + xstr(p.state) + "</member>",
			"<member><name>pid</name>" + xint(p.pid) + "</member>",
			"<member><name>exitstatus</name>" + xint(p.exit) + "</member>",
			"<member><name>stdout_logfile</name>" + xstr(p.logfile) + "</member>",
		}
		return ok("<value><struct>" + strings.Join(members, "") + "</struct></value>")
	case "supervisor.startProcess":
		if p == nil {
			return fault(FaultBadName, "BAD_NAME: "+name)
		}
		if p.state == "RUNNING" {
			return fault(FaultAlreadyStarted, "ALREADY_STARTED: "+name)
		}
		f.startLocked(p)
		return ok(xtrue())
	case "supervisor.stopProcess":
		if p == nil {
			return fault(FaultBadName, "BAD_NAME: "+name)
		}
		if p.state != "RUNNING" {
			return fault(FaultNotRunning, "NOT_RUNNING: "+name)
		}
		f.stopLocked(p, "STOPPED", 0)
		return ok(xtrue())
	case "supervisor.signalProcess":
		return ok(xtrue())
	case "supervisor.reloadConfig":
		confs := f.readConfs()
		var added, changed []string
		for n, c := range confs {
			q := f.progs[n]
			switch {
			case q == nil:
				added = append(added, xstr(n))
			case q.command != c[0] || q.logfile != c[1]:
				changed = append(changed, xstr(n))
			}
		}
		return ok(xarray(xarray(xarray(added...), xarray(changed...), xarray())))
	case "supervisor.addProcessGroup":
		c, found := f.readConfs()[name]
		if !found {
			return fault(FaultBadName, "BAD_NAME: "+name)
		}
		if p != nil {
			return fault(FaultAlreadyAdded, "ALREADY_ADDED: "+name)
		}
		f.progs[name] = &fakeProg{name: name, command: c[0], logfile: c[1], state: "STOPPED"}
		return ok(xtrue())
	case "supervisor.removeProcessGroup":
		if p == nil {
			return fault(FaultBadName, "BAD_NAME: "+name)
		}
		if p.state == "RUNNING" {
			return fault(FaultStillRunning, "STILL_RUNNING: "+name)
		}
		delete(f.progs, name)
		return ok(xtrue())
	}
	return fault(1, "UNKNOWN_METHOD")
}
