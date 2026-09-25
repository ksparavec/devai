package main

// Job runners: backends that hold the GPU for a JOB rather than for
// requests. The laya trainer (docs/plans/laya-trainer.md) is the first: a
// request to its port starts a fine-tuning job that runs for minutes to
// hours after the request has returned.
//
// Everything else in the router assumes the opposite -- a backend is busy
// exactly while requests are in flight -- so three things differ for a
// backend whose config has JobRunner set:
//
//   - It is model-agnostic. The request's `model` names a job setting (the
//     base checkpoint), never a launch setting, so a different name never
//     recreates the container; the allowlist comes from the laya catalog.
//   - An empty-body POST is accepted on its port: OpenAI's cancel call has
//     no body, and the router otherwise parses every POST body as JSON.
//   - The busy hold. While its /health says "busy", no other backend may
//     evict it: stopOtherBackends refuses with a gpuHeldError (503 +
//     Retry-After) until started_at + MaxHold, after which it evicts anyway.
//     The deadline is computed here from the runner's own `started_at`, so
//     a restarted router (boot adoption) keeps the original clock.

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"syscall"
	"time"

	"gopkg.in/yaml.v3"
)

// jobHoldRetryAfter is the Retry-After (seconds) sent while a job holds
// the GPU. Jobs run for minutes to hours; 30 s keeps a polling client
// cheap without making it wait long once the job ends.
const jobHoldRetryAfter = 30

// jobRunnerHealth is a job runner's GET /health body.
type jobRunnerHealth struct {
	Status    string `json:"status"`
	Job       string `json:"job"`
	Phase     string `json:"phase"`
	StartedAt int64  `json:"started_at"`
}

// gpuHeldError refuses a switch while a job runner holds the GPU.
type gpuHeldError struct {
	holder   string
	job      string // "" when the runner does not answer /health
	phase    string
	deadline time.Time // zero: no cap
}

func (e *gpuHeldError) Error() string {
	until := "until the job ends"
	if !e.deadline.IsZero() {
		until = fmt.Sprintf("until the job ends, at the latest %s (LAYA_MAX_HOLD_S)",
			e.deadline.UTC().Format(time.RFC3339))
	}
	if e.job == "" {
		return fmt.Sprintf("the GPU is held by %s: its container runs but it does not answer "+
			"/health, so it is treated as busy %s; retry later", e.holder, until)
	}
	return fmt.Sprintf("the GPU is held by %s: training job %s (phase %s) holds it %s; retry later",
		e.holder, e.job, e.phase, until)
}

// probeJobRunner reads a job runner's /health once.
func (a *arbiter) probeJobRunner(bs *backendState) (*jobRunnerHealth, error) {
	resp, err := a.healthClient.Get(bs.config.BackendURL.String() + bs.config.HealthPath)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("health %s", resp.Status)
	}
	var h jobRunnerHealth
	if err := json.Unmarshal(body, &h); err != nil {
		return nil, fmt.Errorf("health body: %w", err)
	}
	return &h, nil
}

// probeRefused says nothing is listening: the connection was refused. That
// is the compose `sleep infinity` placeholder, or a controller still
// starting -- neither runs a job. Anything else that fails -- a timeout, a
// dropped connection, an error status, and a DNS failure too (one resolver
// hiccup must not evict a job) -- is "silent", and podman decides.
func probeRefused(err error) bool {
	return errors.Is(err, syscall.ECONNREFUSED)
}

// probeJobRunnerRetrying retries a silent runner like backendVanished
// retries an engine: one slow answer from a loaded process is not silence.
func (a *arbiter) probeJobRunnerRetrying(bs *backendState) (*jobRunnerHealth, error) {
	var err error
	for attempt := 1; attempt <= healthProbeAttempts; attempt++ {
		var h *jobRunnerHealth
		if h, err = a.probeJobRunner(bs); err == nil || probeRefused(err) {
			return h, err
		}
		if attempt < healthProbeAttempts {
			time.Sleep(500 * time.Millisecond)
		}
	}
	return nil, err
}

// containerGone reports a libpod state in which a container holds no GPU.
// "" is what containerState reads for a container libpod does not know
// (a 404 body decodes to an empty status): removed, so gone.
func containerGone(status string) bool {
	switch status {
	case "", "exited", "stopped", "dead", "created", "configured":
		return true
	}
	return false
}

// clearHold forgets everything gpuHeldBy remembered about a job runner.
// Called whenever the runner is stopped, found gone or launched afresh, so
// a later probe never judges a new container by an old job.
func (bs *backendState) clearHold() {
	bs.lastBusy = nil
	bs.busySeenAt = time.Time{}
	bs.silentSince = time.Time{}
	bs.heldVerdict = nil
	bs.holdCheckedAt = time.Time{}
}

// cachedHold returns a hold verdict reached less than a.holdCacheTTL ago,
// so a burst of refused requests costs one probe, not one each -- the probe
// runs under a.mu. Only holds are cached: a "not held" verdict leads to an
// eviction, which must be decided fresh (after draining the runner).
func (a *arbiter) cachedHold(bs *backendState) *gpuHeldError {
	h := bs.heldVerdict
	if h == nil || time.Since(bs.holdCheckedAt) >= a.holdCacheTTL {
		return nil
	}
	if !h.deadline.IsZero() && time.Now().After(h.deadline) {
		return nil
	}
	return h
}

// gpuHeldBy reports whether job runner `bs` holds the GPU against other
// backends. Called with a.mu held; probes /health (with retries) and, when
// it is silent, asks podman.
//
//   - busy                        -> held until started_at + MaxHold
//   - ok, or nothing listening    -> not held
//   - silent, container gone      -> not held (and the runner's state reset)
//   - silent, container running   -> held: as its last busy answer if it
//     had one, else from the moment it went silent, + MaxHold. A controller
//     packaging on every core can miss probes; one that hangs is bounded by
//     the cap (0 = no cap: then only a stop or cache-down ends it).
func (a *arbiter) gpuHeldBy(bs *backendState) *gpuHeldError {
	if held := a.cachedHold(bs); held != nil {
		return held
	}
	h, err := a.probeJobRunnerRetrying(bs)
	now := time.Now()
	var started time.Time
	switch {
	case err == nil && h.Status == "busy":
		if bs.lastBusy == nil || bs.lastBusy.Job != h.Job {
			bs.busySeenAt = now
		}
		bs.lastBusy, bs.silentSince = h, time.Time{}
		started = bs.busySeenAt
	case err == nil, probeRefused(err):
		bs.clearHold()
		return nil
	default:
		if status, _, ok := a.containerState(bs.config.ContainerName); ok && containerGone(status) {
			log.Printf("%s: /health silent and its container is %s; it holds nothing",
				bs.config.Name, status)
			bs.clearHold()
			bs.running, bs.containerLaunched = false, false
			bs.currentModel, bs.currentContext, bs.currentSpec = "", 0, nil
			return nil
		}
		if bs.silentSince.IsZero() {
			bs.silentSince = now
		}
		if bs.lastBusy != nil {
			h, started = bs.lastBusy, bs.busySeenAt
		} else {
			h, started = &jobRunnerHealth{}, bs.silentSince
		}
		log.Printf("warning: %s: /health silent (%v) while its container runs; holding the GPU "+
			"(last seen: job %q)", bs.config.Name, err, h.Job)
	}
	held := &gpuHeldError{holder: bs.config.Name, job: h.Job, phase: h.Phase}
	if bs.config.MaxHold > 0 {
		if h.StartedAt > 0 {
			started = time.Unix(h.StartedAt, 0)
		}
		held.deadline = started.Add(bs.config.MaxHold)
		if now.After(held.deadline) {
			log.Printf("%s: held the GPU since %s (job %q), past its %s cap; evicting",
				bs.config.Name, started.UTC().Format(time.RFC3339), h.Job, bs.config.MaxHold)
			bs.clearHold()
			return nil
		}
	}
	bs.heldVerdict, bs.holdCheckedAt = held, now
	return held
}

// jobRunnerPresent is boot adoption's test for a job runner: adopt unless
// nothing listens or podman says the container is gone. A busy runner that
// happens to be silent at boot must still be adopted -- otherwise nothing
// stops the next switch from launching an engine onto its GPU.
func (a *arbiter) jobRunnerPresent(bs *backendState) bool {
	_, err := a.probeJobRunnerRetrying(bs)
	if err == nil {
		return true
	}
	if probeRefused(err) {
		return false
	}
	status, _, ok := a.containerState(bs.config.ContainerName)
	return !ok || !containerGone(status)
}

// --- the laya trainer ---------------------------------------------------------

// layaTrainerEntrypoint starts the trainer's controller. The model is a job
// setting, so it is not part of the launch.
func layaTrainerEntrypoint(_ string, _ launchConfig) []string {
	return []string{"python", "-m", "laya_trainer.controller",
		"--store", "/laya", "--catalog", "/etc/devai/laya-models.yaml", "--port", "11434"}
}

// layaTrainerBackend is the laya trainer's backend entry, from the
// environment compose hands the router.
func layaTrainerBackend(network string) backendConfig {
	u, err := url.Parse(env("LAYA_TRAINER_URL", "http://laya-trainer:11434"))
	if err != nil {
		log.Fatalf("invalid LAYA_TRAINER_URL: %v", err)
	}
	image := env("LAYA_TRAINER_IMAGE", "localhost/devai-laya-trainer:latest")
	return backendConfig{
		Name:          "laya-trainer",
		ListenPort:    envInt("LAYA_TRAINER_PORT", 11438),
		BackendURL:    u,
		ContainerName: env("LAYA_TRAINER_CONTAINER", "devai-laya-trainer"),
		Image:         image,
		// The whole laya store, read-write: the trainer imports inbox/ into
		// datasets/ and writes runs/; it keeps base/ read-only itself.
		ModelsDir:  env("LAYA_STORE_DIR", "/var/cache/devai/laya"),
		MountDest:  "/laya",
		MountRW:    true,
		Network:    network,
		HealthPath: "/health",
		Entrypoint: layaTrainerEntrypoint,
		EnvVars: map[string]string{
			// Recorded in every artifact manifest.
			"LAYA_TRAINER_IMAGE": image,
			"LAYA_JOB_TIMEOUT_S": env("LAYA_JOB_TIMEOUT_S", "0"),
		},
		JobRunner: true,
		// 0 disables the cap. 900 s: about 6x the jobs measured in plan
		// Phase 4 (123-140 s), headroom for larger datasets (owner decision).
		MaxHold: time.Duration(envIntAllowZero("LAYA_MAX_HOLD_S", 900)) * time.Second,
	}
}

// loadLayaModelNames reads the base checkpoint names from deploy/laya-
// models.yaml: the trainer's model allowlist. A missing or unreadable
// catalog registers no names, so every job request is refused with 404
// rather than launching the trainer for a model it cannot have.
func loadLayaModelNames(path string) []string {
	data, err := os.ReadFile(path)
	if err != nil {
		log.Printf("laya catalog: %s not readable (%v) -- the trainer accepts no models", path, err)
		return nil
	}
	var doc struct {
		Models []struct {
			Name string `yaml:"name"`
		} `yaml:"models"`
	}
	if err := yaml.Unmarshal(data, &doc); err != nil {
		log.Printf("warning: laya catalog %s parse failed: %v -- the trainer accepts no models", path, err)
		return nil
	}
	var names []string
	for _, m := range doc.Models {
		if m.Name != "" && isSafeModelName(m.Name) {
			names = append(names, m.Name)
		}
	}
	return names
}
