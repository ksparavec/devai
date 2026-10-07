package bus

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/nats-io/nats.go"
)

// Environment variables every bus client reads (C7).
const (
	EnvURL          = "DEVAI_BUS_URL"
	EnvUser         = "DEVAI_BUS_USER"
	EnvPasswordFile = "DEVAI_BUS_PASSWORD_FILE"
)

// ReconnectWait is the pause between reconnect attempts (C3: nats.go
// MaxReconnects(-1) with ReconnectWait 1 s).
const ReconnectWait = time.Second

// Config names a bus user: the server URL, the user and the 0600 file that
// holds its password (C1, C7).
type Config struct {
	URL          string `yaml:"url"`
	User         string `yaml:"user"`
	PasswordFile string `yaml:"password_file"`
}

// ConfigFromEnv reads DEVAI_BUS_URL, DEVAI_BUS_USER and
// DEVAI_BUS_PASSWORD_FILE.
func ConfigFromEnv() (Config, error) {
	c := Config{
		URL:          os.Getenv(EnvURL),
		User:         os.Getenv(EnvUser),
		PasswordFile: os.Getenv(EnvPasswordFile),
	}
	return c, c.Validate()
}

// Validate refuses an incomplete configuration.
func (c Config) Validate() error {
	var missing []string
	if c.URL == "" {
		missing = append(missing, EnvURL)
	}
	if c.User == "" {
		missing = append(missing, EnvUser)
	}
	if c.PasswordFile == "" {
		missing = append(missing, EnvPasswordFile)
	}
	if len(missing) > 0 {
		return fmt.Errorf("bus configuration incomplete: %s not set", strings.Join(missing, ", "))
	}
	if !ValidName(c.User) {
		return fmt.Errorf("bus user %q is not a valid name", c.User)
	}
	return nil
}

// InboxPrefix is the user's own inbox prefix, _INBOX.<user>: each user may
// subscribe only under its own (C1), so one user cannot read another's
// replies.
func (c Config) InboxPrefix() string { return "_INBOX." + c.User }

// Password reads the password file; surrounding whitespace is dropped.
func (c Config) Password() (string, error) {
	b, err := os.ReadFile(c.PasswordFile)
	if err != nil {
		return "", fmt.Errorf("read bus password file: %w", err)
	}
	p := strings.TrimSpace(string(b))
	if p == "" {
		return "", fmt.Errorf("bus password file %s is empty", c.PasswordFile)
	}
	return p, nil
}

// Options returns the nats.go options devai's clients share: the user's
// credentials and inbox prefix, and reconnecting without limit, one second
// apart, with the subscriptions restored on reconnect (nats.go's default).
func (c Config) Options(name string) ([]nats.Option, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	pass, err := c.Password()
	if err != nil {
		return nil, err
	}
	return []nats.Option{
		nats.Name(name),
		nats.UserInfo(c.User, pass),
		nats.CustomInboxPrefix(c.InboxPrefix()),
		nats.MaxReconnects(-1),
		nats.ReconnectWait(ReconnectWait),
		nats.ReconnectJitter(0, 0),
		nats.Timeout(5 * time.Second),
	}, nil
}

// Connect connects as the configured user. Extra options (handlers) are
// applied after the shared ones.
func Connect(c Config, name string, extra ...nats.Option) (*nats.Conn, error) {
	opts, err := c.Options(name)
	if err != nil {
		return nil, err
	}
	nc, err := nats.Connect(c.URL, append(opts, extra...)...)
	if err != nil {
		return nil, fmt.Errorf("connect to the bus at %s as %s: %w", c.URL, c.User, err)
	}
	return nc, nil
}

// DecodeReply parses a reply and checks its envelope.
func DecodeReply(data []byte) (Reply, error) {
	var r Reply
	if err := json.Unmarshal(data, &r); err != nil {
		return Reply{}, fmt.Errorf("bad reply: %w", err)
	}
	if r.V != Version {
		return Reply{}, fmt.Errorf("bad reply: v=%d, want %d", r.V, Version)
	}
	if !r.OK && r.Error == nil {
		return Reply{}, errors.New("bad reply: ok=false without an error")
	}
	return r, nil
}

// DecodeState parses an event (a bare State).
func DecodeState(data []byte) (State, error) {
	var s State
	if err := json.Unmarshal(data, &s); err != nil {
		return State{}, fmt.Errorf("bad state: %w", err)
	}
	return s, nil
}

// Do sends req to subject and waits for the reply until ctx ends.
func Do(ctx context.Context, nc *nats.Conn, subject string, req Request) (Reply, error) {
	data, err := Marshal(req)
	if err != nil {
		return Reply{}, err
	}
	msg, err := nc.RequestWithContext(ctx, subject, data)
	if err != nil {
		return Reply{}, err
	}
	return DecodeReply(msg.Data)
}

// Publish encodes v and publishes it.
func Publish(nc *nats.Conn, subject string, v any) error {
	data, err := Marshal(v)
	if err != nil {
		return err
	}
	return nc.Publish(subject, data)
}
