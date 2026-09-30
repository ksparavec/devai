# Slurm gatekeeper

_Slurm, in containers, becomes the only thing that starts GPU work in devai; the router becomes its client; every job leaves a central record with its results and GPU accounting._

## Status

Draft. Fourth layout proposed 2026-10-01 (D15-D16, on top of D7-D14); nothing of it is built or tested yet. An earlier spike (2026-09-29) tested two layouts since discarded and is obsolete (see Context).

## Dependencies

- [Plan: minimal-external-images](./minimal-external-images.md) -- builds `devai-engines` (Ollama, vLLM 0.28 + HyperQwen, SGLang 0.5.16; CUDA 13.1; everything compiled here; no Slurm), switches today's router to it, renames `vllm-devai` to `vllm`, and re-probes the fleet. This plan adds Slurm around it. Its decisions M9-M15 apply here too: Open WebUI and MCP dropped, AMD/ROCm and the sops/age scaffold to the attic, one digest-pinned `debian:trixie-slim` for every image, one lab image (`devai-lab`, on `devai-base`), and the GPU optional -- one host check decides at launch, and a host without an NVIDIA GPU runs only Ollama (CPU) and the laya trainer.

## Enables / Unblocks

- A guarantee that no engine starts on a GPU another process still holds (the 2026-09-28 failure, below).
- Queueing, suspend/resume, cancel and queue-clearing for GPU work, done by a scheduler instead of by hand.
- One place that says what ran yesterday: every job's state, exit, times, GPU memory, utilisation and energy, plus its results.
- Several local GPUs: one engine per GPU falls out of Slurm's GPU allocation (not in scope here, see Out of scope).

## Out of scope

- Multi-host clustering. Slurm could do it later; devai does not use it.
- Shielding the GPU from the host user or from software outside devai. Application containers stop using the GPU by convention (D1), not by a security boundary. A production setup would add one (driver device-file mode plus a service account); this plan does not.
- A separate project/repository (D5).
- Changing what the router does to requests (rewrites, reasoning policy, tool stripping, SSE keepalive). Only its launch layer changes.

## Operator decisions (2026-09-29)

- **D1 -- Workload control.** Both queueing and preemption:
  - queue a job for later;
  - suspend the running job, run a quick job (another backend and/or model), then resume the suspended job;
  - remove one job from the queue, or clear the queue;
  - when the interrupting job needs the same backend and model, the interruption must be fast (no engine restart).
  - devai application containers always go through the router and never use the GPU directly.
- **D2 -- Slurm runs in containers**, so devai stays independent of the host's Linux distribution. The containers get the privileges they need.
- **D3 -- Job results live in `/var/cache/devai/jobs`** (its own volume, per the mount-point convention in CLAUDE.md).
- **D4 -- Use a published image if one exists.** *(Superseded by D7 on 2026-09-30.)* Docker Hub has none that is maintained (2026-09-29: the `slurm/` namespace holds two repositories last updated in 2017 and 2018; the rest are personal images). SchedMD, Slurm's maintainer, publishes official images on GHCR (`ghcr.io/slinkyproject/{slurmctld,slurmd,slurmdbd,slurmrestd}:26.05-ubuntu26.04`, Slurm 26.05.4, recipe github.com/SlinkyProject/containers). Its `slurmd` was built without the `gpu_nvml` plugin.
- **D5 -- Not a separate project.** One consumer, contracts that change together with the router and the workloads, no product without devai; a second repository would make every change a cross-repo release (the aiagent experience). It stays extractable: own directory, own tests, config under `deploy/slurm/`, a written job/result format.
- **D6 -- No containers inside containers.** Ruled out after spike round 1, which ran engines as podman containers nested in the slurmd container. (It also meant, until D10, that an engine ran as the job's own process; D10 replaced that with `podman exec` into a sibling container, which is still not nesting; since D15 no job starts a process in another container at all.)

## Operator decisions (2026-09-30, second round)

The first architecture (one Slurm node per backend image) was judged far too complicated; a second one (Slurm and every backend in one image) mixed Slurm with the backends.

- **D7 -- Fewer moving parts, above all fewer images.** Follow the image-reduction plan: `debian:trixie` is the only upstream image, everything else is built here. For this plan: Slurm's daemons come from devai's own Slurm build and MariaDB from Debian, which **supersedes D4**.
- **D8 -- One version per backend.** One vLLM (0.28.0 + HyperQwen), one SGLang, one Ollama, one laya trainer; a model that does not run on its backend's version is dropped. One vLLM backend, `vllm` on 11435; `vllm-devai` and 11437 are retired.
- **D9 -- The backends live in one Debian trixie image, `devai-engines`, with no Slurm in it.** The laya trainer keeps its own image, also without Slurm.
- **D10 -- All of Slurm -- slurmctld, slurmdbd, slurmrestd, MariaDB and slurmd -- in one image, `devai-slurm`, run as one container.** A job starts its process inside the backend's container with `podman exec`. Chosen over putting slurmd into the backends. *(Its `podman exec` part is superseded by D15 on 2026-10-01; all of Slurm in one container stands.)*
- **D12 -- Slurm from Debian trixie's own packages** (2026-09-30): `slurmctld`, `slurmd`, `slurmdbd`, `slurmrestd`, `slurm-wlm-jwt-plugin`, `slurm-wlm-mysql-plugin`, version 24.11.5; no source build. Slurm no longer accounts GPUs itself (D10), so it needs no NVML build, and nothing in the design needs 26.05. The current build is `24.11.5-4+deb13u1` from `trixie-security`, so the image keeps that suite enabled.
- **D11 -- CUDA 13.1 only, everything compiled here** (the engines and their own CUDA kernels; torch, FlashInfer, Triton stay pinned, hash-locked PyPI packages). SGLang 0.5.16 is compiled here, with `sglang-kernel` built for sm120. (Recorded in the image-reduction session and relayed from there.)
- **D13 -- A suspended bench task is re-run, not continued** (2026-09-30). inspect's per-sample clock would keep running through a suspend and score the samples in flight as wrong, so a bench job is not frozen: suspending it cancels the running task and submits the same task again, held; resuming releases it, and the task starts again from its first sample. The cancelled attempt stays in the history with its log and is not scored. Tasks already finished are kept (one job per model and task).
- **D14 -- The laya trainer's jobs become Slurm trainer jobs** (2026-09-30, agreed by the owner for devai and aiagent). Its fine-tuning API toward aiagent on :11438 does not change.

## Operator decisions (2026-10-01, third round)

- **D15 -- Engines run all the time; jobs only load and unload models.** Each engine container keeps its services running under its own supervisord; nothing in `devai-slurm` starts or stops an engine. A Slurm job tells an engine to load a model onto the GPU and, when it ends, to unload it; the engine never loads or drops a model on its own. vLLM and SGLang serve one model per process, with its settings fixed at start (vLLM 0.28 source, SGLang 0.5.16 image), so for them the part that always runs is a devai control service that starts and stops the model process on the job's command (option A). The fast path: a vLLM or SGLang model that a pending job will need again is put to sleep instead of unloaded (vLLM `/sleep` level 1, SGLang `/release_memory_occupation`), at most one at a time, and woken by the next load of the same model. The operator can start, stop, restart and reload any engine service. Supersedes D10's `podman exec`.
- **D16 -- A message bus, NATS, inside `devai-slurm`** (Debian's `nats-server` 2.10.27, under devai-slurm's supervisord). Every devai container runs `devai-control`, which takes commands from the bus -- start, stop, restart, reload a service; load, unload, sleep a model; train; run an operator action -- and publishes its state. Chosen over supervisord's built-in remote interface, which has no events and needs every client to know every container's address and password.

## Open questions

1. ~~Where does an engine run?~~ In its own container, always running under supervisord; jobs load and unload models through the bus (D15, D16).
2. ~~Does the exec path hold up?~~ Gone: jobs no longer start processes in other containers (D15).
3. ~~Slurm from Debian's packages or built from source?~~ Debian's packages, 24.11.5 (D12).
4. ~~inspect's per-sample clock during a suspend?~~ A suspended bench task is re-run from its first sample (D13).
5. ~~A `devai-workload` container?~~ No: bench and test runs are operator actions, run as jobs in `devai-operator` ([Plan: devai-operator](./devai-operator.md)).
6. Does slurmd run in an unprivileged container with `proctrack/linuxproc`, now that job scripts only send bus commands? -- Phase 1; the measured privileged setup is the fallback.
7. The sleep fast path: how long sleep and wake take, how much VRAM a sleeping vLLM or SGLang process keeps, and whether sleep support costs speed or memory while serving. -- Phase 1 measures it.

## Context

On 2026-09-28 the re-bench lost a day. Two bench runs overlapped (a watcher read a truncated `podman ps`); the router flipped the GPU between Ollama and vLLM and relaunched stock vLLM every ~10 s; vllm-devai then refused to start three times ("Free memory on device cuda:0 (3.25/23.43 GiB) on startup is less than desired GPU memory utilization (0.96, 22.49 GiB)"), which tripped the router's launch circuit breaker, and the breaker refused that model for the rest of the day -- every later bench task on it was 503.

Reading the router showed why nothing stopped this (`gpu-arbiter/main.go`, `ensureBackendRunning` / `stopOtherBackends`):

- it never looks at the GPU: no free-VRAM check, no process list, between "stop the others" and "launch";
- it stops only backends its own flags call running, so anything it lost track of, and anything it did not start, is invisible;
- a failed stop is a warning and the launch goes ahead; Ollama's unload is asynchronous and followed by a fixed 2 s sleep;
- at 0.96 utilisation any holder above ~0.94 GiB makes vLLM refuse to start;
- the breaker counts such a refusal against the model, and only a router restart or another model resets it;
- the router has no GPU device at all; the device is handed to five compose services, every router-created engine, every lab and bench container and the probers.

Separately, results are scattered: markdown, JSON, often outside the tree, with no single place that says what ran and how it ended.

Writing a process manager for this in the router was the first idea; Slurm already does queueing, priorities, preemption, time limits, signals, process-tree cleanup, accounting and GPU allocation, and is maintained.

An earlier spike (2026-09-29, commits 2e68b45 to e194309) tested two layouts that were then discarded: engines as containers nested in the Slurm node, and one Slurm node per backend image with the engine as the job's own process. It is obsolete. The facts it established that do not depend on the layout are the ones docs/slurm.md marks as measured: Slurm under rootless podman with a cgroup nesting step, slurmrestd accepting a JWT the client signs itself, the queue commands, a second job waiting on a license, a suspended job keeping its resources, the guard killing an outside GPU holder, and MariaDB 11.8's snapshot-isolation setting.

## Approach

The full proposed architecture, with a diagram of every component and interaction, is [docs/slurm.md](../slurm.md). In short:

- **`devai-slurm`, one image and one container:** Slurm 24.11.5, MariaDB, NATS and supervisor, all from Debian's packages (D12, D16). A single-node cluster with one license, `engine:1`, which every engine, trainer and probe job takes, so one holds the GPU at a time -- with or without a GPU (M15). It has `--pid=host` and gets the GPU (for NVML only) when the host has one; it holds no podman socket, and may not need `--privileged` any more (Open question 6).
- **Engines run all the time and carry no Slurm (D15):** `devai-engines` (the image-reduction plan) and `devai-laya-trainer` run supervisord and `devai-control`. Operator actions -- bench, pulls, builds -- are devai's scripts, run as jobs whose commands `devai-control` in `devai-operator` carries out ([Plan: devai-operator](./devai-operator.md)).
- **Jobs talk to the bus:** an engine job sends `load`, renews a lease while the model serves, and sends `unload` -- or `sleep`, when a pending job needs the model -- when it ends; a lease that runs out unloads the model if the job vanished. Kinds: engine (router only), trainer, probe (license); operator action (no license; a bench keeps its engine with a hold through the router).
- **The router** stays a separate, unprivileged container, gives up its podman socket, becomes a Slurm client (REST + self-signed JWT) and follows engine state on the bus; requests still go straight to the engines. It blocks the engines' control routes.
- **The GPU guard** is the prolog of every GPU job: idle card, or ask devai's engines to unload and kill any other holder, or drain the node and hold the job.
- **History:** slurmdbd, `/var/cache/devai/jobs/results/<jobid>/` and the bus log, joined by `devai-jobs`.

Code we expect to delete: `scripts/bench/clean_slate.py`, the deadline handling in `scripts/bench-sync.py`, the router's podman launch layer and job-runner hold.

---

## Phase 1 -- devai-slurm, the bus and one engine

First, the unknowns, on a stand-in: Debian's Slurm 24.11.5 in an unprivileged container with `proctrack/linuxproc` (Open question 6), and vLLM sleep and wake on the 27B -- times, the VRAM a sleeping process keeps, host RAM (Open question 7). Then `deploy/slurm/`: the Dockerfile (Debian's Slurm, MariaDB, NATS and supervisor packages), the Slurm, NATS and supervisord configuration, the guard, the sampler and the bus log; `devai-control` and `devai-bus` in `devai-tools` (Go, `nats.go`); supervisord and `devai-control` in `deploy/Dockerfile.engines` and `deploy/Dockerfile.laya-trainer` (the image-reduction plan's Phase 4, step 4c, leaves them to this plan); the image on the one pinned `debian:trixie-slim` variable (its M13); `make build-slurm`, `make slurm-init` (keys, the MariaDB password and the bus passwords as plain 0600 files; no sops, which goes to the attic, M12), the `/var/cache/devai/jobs` volume, backups. Exit: an engine job loads and unloads a model through the bus, a lease that runs out unloads it, a sleeping model wakes, the job's GPU summary appears in `sacct`, a stray GPU holder is killed or drains the node, the operator restarts an engine service over the bus, and on a host without a GPU the same stack runs Ollama and trainer jobs with the guard and the sampler switched off.

## Phase 2 -- Router as Slurm client

The router launches and stops engines through slurmrestd and follows their state on the bus, gives up its podman socket, takes the GPU holder from Slurm, and blocks the engines' control routes. A start refused on a busy card no longer counts against the model. Exit: the 2026-09-28 "free memory" failure cannot be reproduced.

## Phase 3 -- Workloads, trainer, probes as jobs

Bench (the 30-minute limit as a Slurm time limit), probes (the probers load through the bus instead of starting engine containers) and the laya trainer as Slurm jobs; operator actions as jobs in `devai-operator` (its Phase 3); holds; queue, suspend/resume (D13), cancel and clear-queue commands (D1); the sleep fast path for an interrupt with another model. aiagent's fine-tuning API on :11438 stays. Exit: a bench and an interactive session run side by side without contention; interrupting works with the same and with another model.

## Phase 4 -- History

The result format and `devai-jobs` (no MCP tool: the image-reduction plan drops MCP, M10). The bench leaderboard records each task's job id. Exit: "what happened yesterday" is one command.

## Phase 5 -- Cleanup

The router's podman launch code and job-runner hold, the probers' container code, `clean_slate.py`, the bench deadline code; the GPU device in the lab; router.md, backends.md, bench-results.md, CLAUDE.md. Exit: no devai application container has a GPU device; `make test-router` and `make test-python` pass.

---

## Combined risk register

| Risk | Phase | Mitigation |
| ---- | ----- | ---------- |
| a job vanishes without unloading (its script killed, `devai-slurm` restarted) | 1 | the epilog's `unload`; the 60 s lease; the guard's idle check |
| the bus is down longer than a lease and engines unload under running jobs | 1 | supervisord restarts nats-server at once; the jobs end, the router resubmits |
| a sleeping model's VRAM leaves too little for the next load | 1 | the load's memory fraction leaves it free; else `devai-control` unloads the sleeper and loads again |
| sleep support (`--enable-sleep-mode`, `--enable-memory-saver`) costs speed or memory while serving | 1 | measured in Phase 1; if it does, start with it only the model a pending job needs |
| slurmd needs `--privileged` after all | 1 | the measured privileged setup with the cgroup step |
| submitting a job can run any registered operator action, destructive ones included | 1-2 | the JWT key only in the router (read-only) and a 0600 host file; `devai-slurm` and NATS on `devai-net` only; bus users limited to their subjects |
| GPU accounting by sampling is coarser than Slurm's own | 1 | one GPU job at a time, so the whole card is the job's; 5 s samples plus NVML's energy counter |
| Slurm reports no GPU energy on NVIDIA | 1 | devai reads NVML's energy counter at job start and end |
| a job held after a prolog failure starts only after Slurm's requeue delay | 2 | the router cancels and resubmits |
| an interrupt with another model costs a cold start (about 95 s for Qwen3.8-27B) | 3 | the sleep fast path for vLLM and SGLang; the same-model path needs nothing |

## Migration / rollback story

Until Phase 2 ships, the router launches as today and Slurm runs beside it. Phase 2 keeps the current launch path behind a switch until the Slurm path has run a full re-bench; rollback is flipping the switch back.

## Estimated effort

| Phase | Wall-clock |
| ----- | ---------- |
| 1 | 2-3 days |
| 2 | 3-5 days |
| 3 | 3-5 days |
| 4 | 1-2 days |
| 5 | 1 day |

Estimates, not measurements.

## References

- Slurm documentation: slurm.conf (`Licenses`, `CompleteWait`, `KillWait`), cgroup.conf (`IgnoreSystemd`), accounting, slurmrestd, rest_api (JWT) -- slurm.schedmd.com
- [docs/slurm.md](../slurm.md) (the architecture), docs/router.md, docs/bench-results.md "Run protocol", gpu-arbiter/main.go (`ensureBackendRunning`, `stopOtherBackends`)
