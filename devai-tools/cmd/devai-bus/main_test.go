package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/nats-io/nats.go"

	"github.com/sparavec/gpu-arbiter/pkg/bus"
	"github.com/sparavec/gpu-arbiter/pkg/bus/bustest"
)

var binary string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "devai-bus-test")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	binary = filepath.Join(dir, "devai-bus")
	if out, err := exec.Command("go", "build", "-o", binary, ".").CombinedOutput(); err != nil {
		fmt.Fprintf(os.Stderr, "build devai-bus: %v\n%s", err, out)
		os.Exit(1)
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

func busEnv(srv *bustest.Server, user string) []string {
	c := srv.Config(user)
	return append(os.Environ(), bus.EnvURL+"="+c.URL, bus.EnvUser+"="+c.User, bus.EnvPasswordFile+"="+c.PasswordFile)
}

// devai-bus request is refused (exit 1) without --job, and for a body that
// carries job or v (C3).
func TestRequestRefusals(t *testing.T) {
	srv := bustest.Start(t)
	cases := [][]string{
		{"request", "devai.engines.fake.state", `{}`},
		{"request", "devai.engines.fake.state", `{"job":5}`, "--job", "5"},
		{"request", "devai.engines.fake.state", `{"v":1}`, "--job", "5"},
		{"request", "devai.engines.fake.state", `{}`, "--job", "-1"},
		{"request", "devai.engines.fake", `{}`, "--job", "5"},
	}
	for _, args := range cases {
		cmd := exec.Command(binary, args...)
		cmd.Env = busEnv(srv, "jobs")
		out, err := cmd.CombinedOutput()
		var ee *exec.ExitError
		if err == nil || !asExit(err, &ee) || ee.ExitCode() != 1 {
			t.Errorf("%v: %v (%s), want exit 1", args, err, out)
		}
	}
}

func asExit(err error, target **exec.ExitError) bool {
	e, ok := err.(*exec.ExitError)
	if ok {
		*target = e
	}
	return ok
}

// A hold started with a dead --parent exits at once.
func TestHoldWithADeadParentExitsAtOnce(t *testing.T) {
	srv := bustest.Start(t)
	cmd := exec.Command(binary, "hold", "devai.engines.fake", "--job", "5", "--parent", "999999")
	cmd.Env = busEnv(srv, "jobs")
	start := time.Now()
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Errorf("exit 0, want a failure: %s", out)
	}
	if d := time.Since(start); d > time.Second {
		t.Errorf("took %s", d)
	}
}

// A hold whose parent shell is SIGKILLed exits within 1 s (PR_SET_PDEATHSIG),
// and it ignores TERM and INT meanwhile.
func TestHoldDiesWithItsShell(t *testing.T) {
	srv := bustest.Start(t)
	ctl, err := bus.Connect(srv.Config("control-engines"), "test")
	if err != nil {
		t.Fatal(err)
	}
	defer ctl.Close()
	job := int64(5)
	_, _ = ctl.Subscribe("devai.engines.fake.state", func(m *nats.Msg) {
		b, _ := bus.Marshal(bus.OKReply(bus.State{Container: "engines", Service: "fake", Status: bus.StatusServing, Job: &job}))
		_ = m.Respond(b)
	})
	_ = ctl.Flush()

	sh := exec.Command("/bin/sh", "-c", binary+` hold devai.engines.fake --job 5 --parent $$ & echo $!; wait`)
	sh.Env = busEnv(srv, "jobs")
	stdout, _ := sh.StdoutPipe()
	if err := sh.Start(); err != nil {
		t.Fatal(err)
	}
	line, err := bufio.NewReader(stdout).ReadString('\n')
	if err != nil {
		t.Fatal(err)
	}
	hold, _ := strconv.Atoi(strings.TrimSpace(line))
	t.Cleanup(func() { _ = syscall.Kill(hold, syscall.SIGKILL) })
	time.Sleep(time.Second)
	if !alive(hold) {
		t.Fatal("the hold is not running")
	}
	_ = syscall.Kill(hold, syscall.SIGTERM)
	_ = syscall.Kill(hold, syscall.SIGINT)
	time.Sleep(300 * time.Millisecond)
	if !alive(hold) {
		t.Fatal("the hold died of TERM or INT; it must ignore both")
	}
	_ = sh.Process.Kill()
	_, _ = sh.Process.Wait()
	killed := time.Now()
	for alive(hold) {
		if time.Since(killed) > time.Second {
			t.Fatalf("the hold outlived its shell by more than 1 s")
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// alive: the process exists and is not a zombie.
func alive(pid int) bool {
	b, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return false
	}
	s := string(b)
	i := strings.LastIndexByte(s, ')')
	return i >= 0 && i+2 < len(s) && s[i+2] != 'Z'
}

func TestUnknownCommandAndHelp(t *testing.T) {
	if run("frobnicate", nil) == 0 {
		t.Error("an unknown command succeeded")
	}
	if run("help", nil) != 0 {
		t.Error("help failed")
	}
}
