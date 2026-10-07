package buscli

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/nats-io/nats.go"

	"github.com/sparavec/gpu-arbiter/pkg/bus"
)

// AllEvents is every event subject.
const AllEvents = "devai.events.>"

// LogLine is one line of the bus log (C4: bus/<YYYY-MM-DD>.jsonl).
type LogLine struct {
	Received string          `json:"received"`
	Subject  string          `json:"subject"`
	Event    json.RawMessage `json:"event"`
}

// LogOptions is `devai-bus log --dir D`.
type LogOptions struct {
	Bus    bus.Config
	Dir    string
	Now    func() time.Time
	Stderr io.Writer
	// Ready is closed once the subscription is in place (tests).
	Ready chan struct{}
}

// busLog appends each line to the file of its receive date (UTC).
type busLog struct {
	mu   sync.Mutex
	dir  string
	day  string
	file *os.File
}

func (l *busLog) write(line LogLine, t time.Time) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	day := t.UTC().Format("2006-01-02")
	if l.file == nil || day != l.day {
		if l.file != nil {
			_ = l.file.Close()
		}
		f, err := os.OpenFile(filepath.Join(l.dir, day+".jsonl"), os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o644)
		if err != nil {
			return err
		}
		l.file, l.day = f, day
	}
	b, err := json.Marshal(line)
	if err != nil {
		return err
	}
	_, err = l.file.Write(append(b, '\n'))
	return err
}

func (l *busLog) close() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.file != nil {
		_ = l.file.Close()
	}
}

// Log writes every bus event, one JSON line with its receive time, until ctx
// ends (the bus log, a supervisord program in devai-slurm; Phase 5 step 7).
func Log(ctx context.Context, o LogOptions) int {
	if o.Stderr == nil {
		o.Stderr = io.Discard
	}
	if o.Now == nil {
		o.Now = time.Now
	}
	if err := os.MkdirAll(o.Dir, 0o755); err != nil {
		fmt.Fprintf(o.Stderr, "devai-bus log: %v\n", err)
		return ExitFailed
	}
	l := &busLog{dir: o.Dir}
	defer l.close()
	opts, err := o.Bus.Options("devai-bus log")
	if err != nil {
		fmt.Fprintf(o.Stderr, "devai-bus log: %v\n", err)
		return ExitFailed
	}
	nc, err := nats.Connect(o.Bus.URL, append(opts, nats.RetryOnFailedConnect(true))...)
	if err != nil {
		fmt.Fprintf(o.Stderr, "devai-bus log: %v\n", err)
		return ExitUnreachable
	}
	defer nc.Close()
	_, err = nc.Subscribe(AllEvents, func(m *nats.Msg) {
		t := o.Now()
		ev := json.RawMessage(m.Data)
		if !json.Valid(m.Data) {
			q, _ := json.Marshal(string(m.Data))
			ev = q
		}
		if err := l.write(LogLine{Received: bus.Timestamp(t), Subject: m.Subject, Event: ev}, t); err != nil {
			fmt.Fprintf(o.Stderr, "devai-bus log: %v\n", err)
		}
	})
	if err != nil {
		fmt.Fprintf(o.Stderr, "devai-bus log: subscribe: %v\n", err)
		return ExitUnreachable
	}
	if o.Ready != nil {
		_ = nc.Flush()
		close(o.Ready)
	}
	<-ctx.Done()
	return ExitOK
}

// WatchOptions is `devai-bus watch [subject]`: events as JSON lines on
// stdout, for people.
type WatchOptions struct {
	Bus     bus.Config
	Subject string
	Stdout  io.Writer
	Stderr  io.Writer
}

// Watch prints events until ctx ends.
func Watch(ctx context.Context, o WatchOptions) int {
	if o.Subject == "" {
		o.Subject = AllEvents
	}
	nc, err := bus.Connect(o.Bus, "devai-bus watch")
	if err != nil {
		fmt.Fprintf(o.Stderr, "devai-bus watch: %v\n", err)
		return ExitUnreachable
	}
	defer nc.Close()
	var mu sync.Mutex
	_, err = nc.Subscribe(o.Subject, func(m *nats.Msg) {
		ev := json.RawMessage(m.Data)
		if !json.Valid(m.Data) {
			q, _ := json.Marshal(string(m.Data))
			ev = q
		}
		b, _ := json.Marshal(LogLine{Received: bus.Timestamp(time.Now()), Subject: m.Subject, Event: ev})
		mu.Lock()
		fmt.Fprintln(o.Stdout, string(b))
		mu.Unlock()
	})
	if err != nil {
		fmt.Fprintf(o.Stderr, "devai-bus watch: %v\n", err)
		return ExitUnreachable
	}
	<-ctx.Done()
	return ExitOK
}
