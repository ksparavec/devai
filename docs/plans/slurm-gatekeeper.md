# Slurm gatekeeper

_Slurm, in containers, becomes the only thing that starts GPU work in devai; the router becomes its client; every job leaves a central record with its results and GPU accounting._

## Status

In Progress -- Phase 0 (spike) 2026-09-29: S1-S5 answered (see "Spike results"); one item open (a real engine inside a job). Decisions D1-D5 below are the operator's.

## Dependencies

None. The plan replaces parts of the router's launch layer (docs/router.md) and of the bench protocol of 2026-09-28 (docs/bench-results.md "Run protocol"); neither is a prerequisite.

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
- **D4 -- Use a published image if one exists.** Docker Hub has none that is maintained (2026-09-29: the `slurm/` namespace holds two repositories last updated in 2017 and 2018; the rest are personal images). SchedMD, Slurm's maintainer, publishes official images on GHCR (`ghcr.io/slinkyproject/{slurmctld,slurmd,slurmdbd,slurmrestd}:26.05-ubuntu26.04`, Slurm 26.05.4, recipe github.com/SlinkyProject/containers). Its `slurmd` is built without the `gpu_nvml` plugin, which GPU accounting needs (see S3).
- **D5 -- Not a separate project.** One consumer, contracts that change together with the router and the workloads, no product without devai; a second repository would make every change a cross-repo release (the aiagent experience). It stays extractable: own directory, own tests, config under `deploy/slurm/`, a written job/result format.

## Open questions

1. ~~Where does an engine run inside a job, so that cancelling the job frees its VRAM and Slurm can account its GPU use?~~ Answered by S2/S3: a nested container inside the job, under `tini -s`.
2. ~~Can `slurmd` get the cgroup control it needs under rootless podman, or does it need a rootful container?~~ Answered by S1: rootless works.
3. ~~`gpu_nvml`: rebuild SchedMD's `slurmd` image from its own recipe with the NVML headers added, or something else?~~ Answered by S3: rebuild the full Slurm package set with the NVML headers; a plugin alone is not enough.
4. Does a real engine (vLLM on vllm-devai, Ollama) serve from inside a job with its model store and engine-cache volumes, and does the router reach it? -- needs the GPU free; last spike item.

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

## Approach

Slurm (controller, node daemon, REST daemon, accounting daemon + MariaDB) runs as containers next to the router. Three kinds of job:

- **Engine job** -- holds one GPU (`gres/gpu:1`) and runs one engine (backend, model, context, MTP). Only the router submits and cancels these, through slurmrestd. No time limit unless one is given (keep-warm).
- **Workload job** -- a bench, probe or test client. Holds no GPU; declares the engine it needs and talks to the router like any client. At most one active workload per GPU (a Slurm license or GRES), so two benches can no longer contend.
- **Trainer job** -- the laya trainer; holds the GPU itself.

Suspend has to be built from parts, because suspending a process (SIGSTOP) does not free its VRAM: suspend = suspend the workload job, and only if the interrupting job needs a different engine, stop the engine job (its epilog verifies the GPU is empty); resume = start the engine the suspended workload needs, then resume the workload. With the same engine, only the workload is suspended and resumed, which is the fast path D1 asks for.

Every GPU job ends with an epilog that removes the job's containers, kills anything left on the GPU, and polls until VRAM is back at idle; if it cannot, the node is set to DRAIN with the reason, so nothing else starts on a dirty GPU. This is the guarantee the router lacks today.

History: slurmdbd keeps every job (state, exit code, times, TRES including GPU memory and utilisation via `gpu_nvml`, the job script, and a JSON comment carrying model, backend, context and git commit). GPU energy is not available from Slurm on NVIDIA (S3) and is recorded by devai. Each job writes `result.json`, its logs and artifacts to `/var/cache/devai/jobs/<jobid>/`. A `devai-jobs` CLI (devai-tools) joins the two: `devai-jobs --since yesterday`.

Code we expect to delete once this works: `scripts/bench/clean_slate.py`, the deadline/SIGINT/SIGKILL handling in `scripts/bench-sync.py`, the router's `stopOtherBackends` / `containerRecreate` bookkeeping and `backendVanished` heuristics, and the job-runner hold in `gpu-arbiter/job_runner.go`.

---

## Phase 0 -- Spike

### Goal

Answer the questions that decide the design, with the official images, before writing anything permanent.

### Questions

- **S1** -- Do slurmctld and slurmd run under rootless podman (privileged as needed) with cgroup v2 (`IgnoreSystemd=yes`, no systemd in the container), see the GPU as `gres/gpu:1`, and run `srun --gres=gpu:1 nvidia-smi`?
- **S2** -- Where does the engine run? (a) as a nested container inside the slurmd container (image store shared read-only via podman `additionalimagestores`; the engine's processes stay in the job's cgroup), or (b) created by the host's podman over its socket, outside the job's cgroup, with the epilog as the only cleanup. Must hold: `scancel` leaves VRAM empty.
- **S3** -- GPU accounting: with `gpu_nvml` built in, do `sacct` / `sstat` show `gres/gpumem` and `gres/gpuutil` for the job? NVML reports host PIDs, so the slurmd container probably needs `--pid=host`. Does `acct_gather_energy/gpu` record energy?
- **S4** -- slurmrestd reachable from the router container with JWT auth; job start latency from submit to running.
- **S5** -- `scontrol suspend` / `resume` on a workload job, `scontrol requeue` on an engine job, `scancel` of a pending job and of the whole queue.

### Deliverables

Findings written into this plan (a "Spike results" section), each answer with the command that showed it. Throwaway config lives in the session scratch directory, not in the tree.

### Exit criteria

S1-S5 answered; Open questions 1-3 closed; Phase 1 deliverables fixed.

### Spike results (2026-09-29)

All on this host (RTX PRO 4000 Blackwell, rootless podman 5, cgroup v2), Slurm 26.05.4, five spike containers on `devai-net` (`spike-slurmctld`, `-slurmd`, `-slurmrestd`, `-slurmdbd`, `-mariadb`). The router's own vllm-devai engine held 21.8 GiB of the card throughout; every GPU test below ran beside it in the remaining ~2.6 GiB.

- **S1 -- yes, rootless.** slurmctld runs unprivileged; slurmd runs `--privileged --cgroupns=private`, with the GPU through CDI. Three things were needed:
  - slurmd failed at first ("Controller memory is not enabled") because the container's processes sat in its cgroup root; an entrypoint hook moves them to a child cgroup first and enables the delegated controllers (the docker-in-docker recipe). After that, jobs run in their own cgroup (`/system.slice/slurmstepd.scope/<job>/step_0/...`) with `CUDA_VISIBLE_DEVICES=0`.
  - `cpuset` is not delegated to user sessions here (systemd `DelegateControllers=cpu memory pids`), so Slurm cannot pin cores. Not needed.
  - slurmctld and slurmdbd drop to the `slurm` user and cannot create their run directories under podman; a two-line entrypoint wrapper creates them.
- **S2 -- the engine runs as a nested container inside the job, (a).** Podman inside the slurmd container, with the host's image store mounted read-only as an `additionalimagestores` entry (no image copy; a 1 GiB-VRAM PyTorch holder started 3 s after submit). Needed:
  - the host's `/usr/bin/nvidia-cdi-hook` mounted into the slurmd container (the CDI spec calls it; it depends on glibc only);
  - `podman run --cgroups=disabled` so the engine's processes stay in the job's cgroup (checked in `/proc/<pid>/cgroup` of the GPU process);
  - `--init --init-path /usr/bin/tini`. Without it the engine is PID 1, ignores SIGTERM, and `scancel` took 30.1 s (Slurm's default `KillWait`, then SIGKILL). With it, VRAM was released **0.2 s** after `scancel`;
  - an epilog that removes the job's containers by label: a SIGKILLed conmon left podman's record ("Up 44 seconds") behind, and the next launch under that name would collide.
  - Network: `--network host` inside the job is the slurmd container's network, so the router reaches an engine at `<slurmd container>:<port>` (HTTP 200 in 4 ms from another `devai-net` container).
- **S3 -- GPU memory and utilisation per job, yes; energy, no.**
  - The official `slurmd` lacks `gpu_nvml`, and adding the plugin file is not enough: NVML autodetection is compiled into Slurm's core ("configured to autodetect nvml functionality, but we weren't able to find that lib when Slurm was configured"). The whole package set has to be built with the NVML headers: SchedMD's own `debuild` step on `ubuntu:26.04` plus `libnvidia-ml-dev` (Ubuntu multiverse, 12.4 headers), source `slurm-26-05-4-1` (sha256 `0e522d39324b7b7da5e8096c678c4af00500ca4c3fe2e6da7e4f8d01f7082ec7`), installed over the official image's packages.
  - GPU TRES need slurmdbd ("slurmdbd is required to run with TRES gres/gpu"), so the accounting database is part of the base. MariaDB 11.8 enables `innodb_snapshot_isolation` by default, which slurmdbd warns will make it fatal on a write conflict: run MariaDB with `--innodb-snapshot-isolation=OFF`.
  - slurmd needs `--pid=host`: NVML reports host PIDs, and Slurm matches them against the job's processes.
  - Slurm sums GPU use over the task's process tree by parent PID. podman's conmon double-forks and is re-parented away, so the engine was not counted (`gres/gpumem=0` while the debug log showed "pid 301001 has GPUUtil=100 and MemMB=1354"). Running the job under `tini -s` (a child subreaper) re-parents conmon to the task: **`gres/gpumem=1354M`, `gres/gpuutil=100`**, live in `sstat` and afterwards in `sacct`.
  - Energy: Slurm 26.05's NVML plugin does not read energy at all (`gpu_p_energy_read` returns success with no data), so `acct_gather_energy/gpu` reports 0 on NVIDIA. GPU energy has to come from devai: NVML's cumulative energy counter read at job start and end, written to the job record.
  - `AccountingStoreFlags=job_comment` is needed for `--comment` to reach `sacct` (it came back empty without it).
- **S4 -- yes.** slurmrestd (official image, run as `nobody`, `-a rest_auth/jwt`) on `devai-net`, token from `scontrol token`. Through REST: submit to RUNNING 0.9 / 1.4 / 2.9 s, DELETE to CANCELLED 15-19 ms (three runs).
- **S5 -- all work; suspend keeps the GPU.** Queueing, `scancel` of one job, `scontrol hold` / `release`, `scontrol requeue` of a running job, and `scancel --user` of everything behaved as documented. A suspended job's output stopped (4 -> 4 lines) and continued on resume. But a suspended job **keeps its GPU allocation**: a queued GPU job stayed `PENDING (Resources)` throughout. So D1's "suspend, run a quick job on another engine, resume" is: suspend the workload job, stop the engine job, run the quick job, restart the engine, resume. With the same engine it is the workload suspend alone.

Also found:

- Recreating the slurmd container gives it a new IP and slurmctld keeps the old one (the node went NOT_RESPONDING and a batch job was requeued). Phase 1 needs stable addresses (fixed IPs on `devai-net`, or restart ordering).
- A changed `slurm.conf` drains the node ("appears to have a different slurm.conf"). Phase 1 should use configless mode (slurmd fetches its config from slurmctld), so there is one copy.
- `slurmd -C` reports 8 CPUs where the host has 24; set `CPUs=` explicitly.
- Not tested yet: a real inference engine (vLLM, Ollama) inside a job, with its model store and engine-cache volumes. It needs the GPU, which the router's warm engine held during the spike.

Open questions 1 and 2 are answered: nested containers in the job, rootless. Open question 3: rebuild the full Slurm package set with NVML (not only the plugin).

---

## Phase 1 -- Base

Slurm images (official where possible; `slurmd` rebuilt with `gpu_nvml` per Open question 3), `deploy/slurm/` config (slurm.conf, gres.conf, cgroup.conf, epilog, JWT and slurm keys generated on first run), compose services, the `/var/cache/devai/jobs` volume, accounting database, backup of both in `devai-backup`. Exit: a test job's GPU memory, utilisation and energy appear in `sacct`, and the epilog drains the node when a stray GPU process survives.

## Phase 2 -- Router as Slurm client

The router launches and stops engines through slurmrestd and takes engine state from Slurm, not from its own flags. Refusals come from Slurm's recorded job outcome, so a start that failed on a busy GPU no longer counts against the model. Exit: the 2026-09-28 "free memory" failure cannot be reproduced (a stray GPU holder drains the node; no engine starts).

## Phase 3 -- Workloads as jobs

Bench (the 30-minute per-task limit as a Slurm time limit), probes and the laya trainer as Slurm jobs; aiagent's fine-tuning API on :11438 stays as it is. Queue, suspend/resume, cancel and clear-queue commands (D1). Exit: a bench and an interactive session run side by side without contention; suspend/run/resume works with both a different and the same engine.

## Phase 4 -- History

The result format, `devai-jobs`, optionally an MCP tool in devai-model-status. The bench leaderboard stays and each entry records its job id. Exit: "what happened yesterday" is one command.

## Phase 5 -- Cleanup

GPU device removed from devai application containers (lab, bench, probers); the code listed under Approach deleted; docs (router.md, backends.md, bench-results.md, CLAUDE.md). Exit: no devai application container has a GPU device; `make test-router` and `make test-python` pass.

---

## Combined risk register

| Risk | Phase | Mitigation |
| ---- | ----- | ---------- |
| slurmd cannot manage cgroups under rootless podman | 0 | resolved by S1 (cgroup nesting hook) |
| engine escapes the job's process tree (conmon double-fork) and is not accounted | 0 | resolved by S3 (`tini -s` subreaper); the epilog still removes leftovers by label |
| `gpu_nvml` absent from the official image | 1 | build the full Slurm package set from SchedMD's recipe with the NVML headers (S3) |
| the nested setup depends on host paths (`nvidia-cdi-hook`, CDI spec, image store) | 1 | check them at start-up and fail with the path named |
| Slurm reports no GPU energy on NVIDIA | 1 | devai reads NVML's energy counter at job start and end |
| Slurm down means nothing can launch | 2 | router reports it plainly (503 naming Slurm); compose restarts the daemons |
| suspend with a different engine costs a cold start (25-125 s measured for Qwen3.8-27B on vllm-devai) | 3 | the same-engine fast path; operators choose |

## Migration / rollback story

Until Phase 2 ships, the router launches as today and Slurm runs beside it. Phase 2 keeps the current launch path behind a switch until the Slurm path has run a full re-bench; rollback is flipping the switch back.

## Estimated effort

| Phase | Wall-clock |
| ----- | ---------- |
| 0 | 1-2 days |
| 1 | 2-3 days |
| 2 | 3-5 days |
| 3 | 3-5 days |
| 4 | 1-2 days |
| 5 | 1 day |

Estimates, not measurements.

## References

- Slurm documentation: slurm.conf, gres.conf, cgroup.conf (`IgnoreSystemd`), preempt, accounting, slurmrestd, rest_api (JWT) -- slurm.schedmd.com
- SchedMD container images: github.com/SlinkyProject/containers (Apache-2.0 recipe; Slurm itself GPL-2.0)
- An unofficial fork that adds `gpu_nvml` to that recipe: github.com/baldwinSPC/slinky-containers (not used; shows the change needed)
- docs/router.md, docs/bench-results.md "Run protocol", gpu-arbiter/main.go (`ensureBackendRunning`, `stopOtherBackends`)
