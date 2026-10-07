// Package bustest starts an embedded nats-server for tests FROM THE RENDERED
// PRODUCTION PERMISSIONS: deploy/slurm/nats-server.conf.in with every
// @PASS:<user>@ replaced by a random password, never an open configuration
// (platform-rebuild Phase 5, step 5). A test that passes here therefore also
// exercises C1's users, subjects and allow_responses.
//
// It imports nats-server, so only tests import it; nothing the router or a
// devai-tools binary links pulls it in.
package bustest

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"testing"
	"time"

	"github.com/nats-io/nats-server/v2/server"

	"github.com/sparavec/gpu-arbiter/pkg/bus"
)

// EnvTemplate overrides where the template is read from.
const EnvTemplate = "DEVAI_NATS_CONF_IN"

var passRE = regexp.MustCompile(`@PASS:([a-z0-9-]+)@`)

// TemplatePath is deploy/slurm/nats-server.conf.in, found from this source
// file's own location (gpu-arbiter/pkg/bus/bustest -> the repository root).
func TemplatePath() string {
	if p := os.Getenv(EnvTemplate); p != "" {
		return p
	}
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return filepath.Join("deploy", "slurm", "nats-server.conf.in")
	}
	root := filepath.Join(filepath.Dir(file), "..", "..", "..", "..")
	return filepath.Join(root, "deploy", "slurm", "nats-server.conf.in")
}

// Server is a running embedded nats-server.
type Server struct {
	t        testing.TB
	Port     int
	URL      string
	Dir      string
	ConfPath string
	Users    []string

	srv *server.Server
}

// Start renders the template into a temporary directory, writes one
// <user>.pass per user, and starts nats-server on a free loopback port.
func Start(t testing.TB) *Server {
	t.Helper()
	tmpl, err := os.ReadFile(TemplatePath())
	if err != nil {
		t.Fatalf("bustest: read %s: %v", TemplatePath(), err)
	}
	dir := t.TempDir()
	pass := map[string]string{}
	rendered := passRE.ReplaceAllStringFunc(string(tmpl), func(m string) string {
		user := passRE.FindStringSubmatch(m)[1]
		if _, ok := pass[user]; !ok {
			b := make([]byte, 16)
			if _, err := rand.Read(b); err != nil {
				t.Fatalf("bustest: random: %v", err)
			}
			pass[user] = hex.EncodeToString(b)
		}
		return pass[user]
	})
	conf := filepath.Join(dir, "nats-server.conf")
	if err := os.WriteFile(conf, []byte(rendered), 0o600); err != nil {
		t.Fatal(err)
	}
	users := make([]string, 0, len(pass))
	for u, p := range pass {
		if err := os.WriteFile(filepath.Join(dir, u+".pass"), []byte(p+"\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		users = append(users, u)
	}
	sort.Strings(users)
	s := &Server{t: t, Port: freePort(t), Dir: dir, ConfPath: conf, Users: users}
	s.URL = fmt.Sprintf("nats://127.0.0.1:%d", s.Port)
	s.start()
	t.Cleanup(s.Shutdown)
	return s
}

func freePort(t testing.TB) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("bustest: free port: %v", err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

func (s *Server) start() {
	s.t.Helper()
	opts, err := server.ProcessConfigFile(s.ConfPath)
	if err != nil {
		s.t.Fatalf("bustest: the rendered configuration does not load: %v", err)
	}
	opts.Host = "127.0.0.1"
	opts.Port = s.Port
	opts.NoLog = true
	opts.NoSigs = true
	srv, err := server.NewServer(opts)
	if err != nil {
		s.t.Fatalf("bustest: new server: %v", err)
	}
	go srv.Start()
	if !srv.ReadyForConnections(10 * time.Second) {
		s.t.Fatalf("bustest: nats-server not ready on port %d", s.Port)
	}
	s.srv = srv
}

// Shutdown stops the server; a second call does nothing.
func (s *Server) Shutdown() {
	if s.srv == nil {
		return
	}
	s.srv.Shutdown()
	s.srv.WaitForShutdown()
	s.srv = nil
}

// Restart starts the server again on the same port, with the same
// configuration and passwords.
func (s *Server) Restart() {
	s.t.Helper()
	s.Shutdown()
	deadline := time.Now().Add(5 * time.Second)
	for {
		l, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", s.Port))
		if err == nil {
			l.Close()
			break
		}
		if time.Now().After(deadline) {
			s.t.Fatalf("bustest: port %d still busy: %v", s.Port, err)
		}
		time.Sleep(20 * time.Millisecond)
	}
	s.start()
}

// Running reports whether the server is up.
func (s *Server) Running() bool { return s.srv != nil }

// Config is the bus configuration of one user of the template.
func (s *Server) Config(user string) bus.Config {
	return bus.Config{URL: s.URL, User: user, PasswordFile: filepath.Join(s.Dir, user+".pass")}
}
