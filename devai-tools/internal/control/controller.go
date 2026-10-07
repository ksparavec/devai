package control

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/nats-io/nats.go"

	"github.com/sparavec/gpu-arbiter/pkg/bus"
)

// service is one entry of control.yaml's services, by kind.
type service interface {
	// handle answers one command; ok=false means the verb takes no reply
	// (lease).
	handle(ctx context.Context, verb string, req bus.Request) (r bus.Reply, ok bool)
	// event is the state published on the service's event subject.
	event() bus.State
	// tick runs the lease clock, the liveness check and the run expiry.
	tick(ctx context.Context, now time.Time)
	// persist returns what the state file keeps; adopt takes it back.
	persist() json.RawMessage
	adopt(ctx context.Context, raw json.RawMessage)
}

// Options are the controller's injectable parts (tests).
type Options struct {
	Memory Memory
	Logger *log.Logger
	Now    func() time.Time
	// NatsOptions are added to the bus connection's (tests).
	NatsOptions []nats.Option
}

// Controller is devai-control.
type Controller struct {
	cfg *Config
	sup *Supervisor
	mem Memory
	log *log.Logger
	now func() time.Time

	natsOpts []nats.Option
	nc       *nats.Conn
	ncMu     sync.RWMutex

	names    []string
	services map[string]service

	saveMu sync.Mutex
	// runCtx is Run's context. Once it ends, the controller is dead: it
	// writes no state file and publishes nothing, as a killed process would
	// not (the tests "kill" a controller by ending it).
	runCtx context.Context
}

func (c *Controller) alive() bool { return c.runCtx != nil && c.runCtx.Err() == nil }

// New builds the controller and its services.
func New(cfg *Config, o Options) (*Controller, error) {
	c := &Controller{cfg: cfg, mem: o.Memory, log: o.Logger, now: o.Now, natsOpts: o.NatsOptions,
		services: map[string]service{}}
	if c.mem == nil {
		c.mem = NewMemory(cfg.GPU)
	}
	if c.log == nil {
		c.log = log.New(os.Stderr, "devai-control: ", log.LstdFlags)
	}
	if c.now == nil {
		c.now = time.Now
	}
	if cfg.Supervisor.Socket != "" {
		c.sup = NewSupervisor(cfg.Supervisor.Socket)
	}
	for name, sc := range cfg.Services {
		var s service
		var err error
		switch sc.Kind {
		case KindFake:
			s = newFakeEngine(c, name, sc)
		case KindProgram:
			s = newProgram(c, name, sc)
		case KindActions:
			s, err = newActions(c, name, sc)
		}
		if err != nil {
			return nil, fmt.Errorf("service %s: %w", name, err)
		}
		c.services[name] = s
		c.names = append(c.names, name)
	}
	sort.Strings(c.names)
	return c, nil
}

func (c *Controller) conn() *nats.Conn {
	c.ncMu.RLock()
	defer c.ncMu.RUnlock()
	return c.nc
}

// publish sends a service's state on devai.events.<container>.<service>.
func (c *Controller) publish(name string) {
	if !c.alive() {
		return
	}
	nc := c.conn()
	s, ok := c.services[name]
	if nc == nil || !ok {
		return
	}
	st := s.event()
	if err := bus.Publish(nc, bus.EventSubject(c.cfg.Container, name), st); err != nil {
		c.log.Printf("publish %s: %v", name, err)
	}
}

// publishState sends an explicit state (a run's event).
func (c *Controller) publishState(name string, st bus.State) {
	if !c.alive() {
		return
	}
	nc := c.conn()
	if nc == nil {
		return
	}
	if err := bus.Publish(nc, bus.EventSubject(c.cfg.Container, name), st); err != nil {
		c.log.Printf("publish %s: %v", name, err)
	}
}

// stateFile is control.json: what each service recorded (C1, "Adoption
// after a restart").
type stateFile struct {
	V        int                        `json:"v"`
	Saved    string                     `json:"saved"`
	Services map[string]json.RawMessage `json:"services"`
}

// save writes the state file atomically (temporary name, rename).
func (c *Controller) save() {
	if !c.alive() {
		return
	}
	c.saveMu.Lock()
	defer c.saveMu.Unlock()
	sf := stateFile{V: 1, Saved: bus.Timestamp(c.now()), Services: map[string]json.RawMessage{}}
	for _, n := range c.names {
		sf.Services[n] = c.services[n].persist()
	}
	b, err := json.MarshalIndent(sf, "", " ")
	if err != nil {
		c.log.Printf("state file: %v", err)
		return
	}
	dir := filepath.Dir(c.cfg.StateFile)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		c.log.Printf("state file: %v", err)
		return
	}
	tmp, err := os.CreateTemp(dir, ".control-*.json")
	if err != nil {
		c.log.Printf("state file: %v", err)
		return
	}
	_, werr := tmp.Write(b)
	cerr := tmp.Close()
	if werr != nil || cerr != nil {
		_ = os.Remove(tmp.Name())
		c.log.Printf("state file: write: %v %v", werr, cerr)
		return
	}
	if err := os.Rename(tmp.Name(), c.cfg.StateFile); err != nil {
		_ = os.Remove(tmp.Name())
		c.log.Printf("state file: %v", err)
	}
}

func (c *Controller) loadState() stateFile {
	var sf stateFile
	b, err := os.ReadFile(c.cfg.StateFile)
	if err != nil {
		if !os.IsNotExist(err) {
			c.log.Printf("state file: %v", err)
		}
		return sf
	}
	if err := json.Unmarshal(b, &sf); err != nil {
		c.log.Printf("state file %s is unreadable, ignored: %v", c.cfg.StateFile, err)
		return stateFile{}
	}
	return sf
}

// Run serves until ctx ends: adopts what the state file records, answers
// commands, publishes every state change and a heartbeat every
// heartbeat_s, and runs each service's clock every poll_s.
func (c *Controller) Run(ctx context.Context) error {
	c.runCtx = ctx
	sf := c.loadState()
	for _, n := range c.names {
		if raw, ok := sf.Services[n]; ok {
			c.services[n].adopt(ctx, raw)
		}
	}
	c.save()

	opts, err := c.cfg.Bus.Options("devai-control " + c.cfg.Container)
	if err != nil {
		return err
	}
	// devai-control reconnects faster than the clients' 1 s (C3 sets that
	// for devai-bus): the sooner its subscription is back after a bus
	// outage, the fewer leases and commands go unanswered.
	opts = append(opts, nats.RetryOnFailedConnect(true), nats.ReconnectWait(250*time.Millisecond))
	opts = append(opts, nats.ConnectHandler(func(*nats.Conn) {
		c.log.Printf("connected to the bus at %s", c.cfg.Bus.URL)
	}))
	opts = append(opts, nats.ReconnectHandler(func(*nats.Conn) {
		c.log.Printf("reconnected to the bus")
		go c.publishAll()
	}))
	opts = append(opts, nats.DisconnectErrHandler(func(_ *nats.Conn, err error) {
		c.log.Printf("disconnected from the bus: %v", err)
	}))
	opts = append(opts, c.natsOpts...)
	nc, err := nats.Connect(c.cfg.Bus.URL, opts...)
	if err != nil {
		return fmt.Errorf("bus: %w", err)
	}
	defer nc.Close()
	c.ncMu.Lock()
	c.nc = nc
	c.ncMu.Unlock()

	subject := "devai." + c.cfg.Container + ".>"
	if _, err := nc.Subscribe(subject, func(m *nats.Msg) { c.onMsg(ctx, m) }); err != nil {
		return fmt.Errorf("subscribe %s: %w", subject, err)
	}
	c.log.Printf("serving %s for services %v", subject, c.names)
	c.publishAll()

	hb := time.NewTicker(c.cfg.Heartbeat())
	defer hb.Stop()
	poll := time.NewTicker(c.cfg.Poll())
	defer poll.Stop()
	for {
		select {
		case <-ctx.Done():
			return nil
		case <-hb.C:
			c.publishAll()
		case <-poll.C:
			now := c.now()
			for _, n := range c.names {
				c.services[n].tick(ctx, now)
			}
		}
	}
}

func (c *Controller) publishAll() {
	for _, n := range c.names {
		c.publish(n)
	}
}

func (c *Controller) onMsg(ctx context.Context, m *nats.Msg) {
	respond := func(r bus.Reply) {
		if m.Reply == "" {
			return
		}
		data, err := bus.Marshal(r)
		if err != nil {
			data, _ = bus.Marshal(bus.ErrReply(bus.CodeBadRequest, "reply too large: "+err.Error(), ""))
		}
		if err := m.Respond(data); err != nil {
			c.log.Printf("reply to %s: %v", m.Subject, err)
		}
	}
	container, name, verb, err := bus.ParseCommandSubject(m.Subject)
	if err != nil || container != c.cfg.Container {
		respond(bus.ErrReply(bus.CodeBadRequest, fmt.Sprintf("not a command for this container: %s", m.Subject), ""))
		return
	}
	s, ok := c.services[name]
	if !ok {
		respond(bus.ErrReply(bus.CodeBadRequest, fmt.Sprintf("no service %q in container %s", name, container), ""))
		return
	}
	req, err := bus.DecodeRequest(m.Data)
	if err != nil {
		respond(bus.ErrReply(bus.CodeBadRequest, err.Error(), ""))
		return
	}
	go func() {
		r, ok := s.handle(ctx, verb, req)
		// A controller that is gone answers nothing: a reply lost with a
		// killed devai-control is what the requester's event and state
		// handling covers (C1, "Adoption after a restart").
		if ok && c.alive() {
			respond(r)
		}
	}()
}

// unknownVerb is the refusal for a verb a kind does not take.
func unknownVerb(kind, verb string) bus.Reply {
	return bus.ErrReply(bus.CodeBadRequest, fmt.Sprintf("verb %q does not apply to a %s service", verb, kind), "")
}
