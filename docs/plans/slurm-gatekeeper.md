# Slurm gatekeeper

_Slurm, in containers, becomes the only thing that starts GPU work in devai; the router becomes its client; every job leaves a central record with its results and GPU accounting._

## Status

In Progress -- Phase 0 (spike) done 2026-09-29. **Design revised 2026-10-01 (D7-D9):** one engine image and one engine container instead of one Slurm node per backend image. Decisions D1-D9 below are the operator's.

## Dependencies

- (placeholder) the image-reduction plan (a parallel session's proposal of 2026-10-01, to be written up as `docs/plans/minimal-external-images.md`): only `debian:trixie` pulled from upstream, everything else built here. This plan's `devai-engines` image is that proposal's GPU part (its items 7-8 and decision D).

Otherwise none. The plan replaces parts of the router's launch layer (docs/router.md) and of the bench protocol of 2026-09-28 (docs/bench-results.md "Run protocol"); neither is a prerequisite.

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
- **D4 -- Use a published image if one exists.** *(Superseded by D7 on 2026-10-01.)* Docker Hub has none that is maintained (2026-09-29: the `slurm/` namespace holds two repositories last updated in 2017 and 2018; the rest are personal images). SchedMD, Slurm's maintainer, publishes official images on GHCR (`ghcr.io/slinkyproject/{slurmctld,slurmd,slurmdbd,slurmrestd}:26.05-ubuntu26.04`, Slurm 26.05.4, recipe github.com/SlinkyProject/containers). Its `slurmd` is built without the `gpu_nvml` plugin, which GPU accounting needs (see S3).
- **D5 -- Not a separate project.** One consumer, contracts that change together with the router and the workloads, no product without devai; a second repository would make every change a cross-repo release (the aiagent experience). It stays extractable: own directory, own tests, config under `deploy/slurm/`, a written job/result format.
- **D6 -- No containers inside containers.** Ruled out after spike round 1, which ran engines as podman containers nested in the slurmd container. Engines run as the job's own process.

## Operator decisions (2026-10-01)

The first architecture (one Slurm node per backend image, 2026-09-30) was judged far too complicated.

- **D7 -- Fewer moving parts, above all fewer images.** Adopt the parallel image-reduction proposal: `debian:trixie` is the only upstream image; everything else is built here. For this plan: Slurm's daemons come from devai's own Slurm build and MariaDB from Debian, which **supersedes D4**.
- **D8 -- One version per backend.** One vLLM (0.28.0 + HyperQwen, today's vllm-devai), one SGLang, one Ollama, one laya trainer. A model that does not run on its backend's version is dropped.
- **D9 -- All backends in one Debian trixie image,** `devai-engines`, run as one container that is also the whole Slurm cluster (one node plus the control plane).

## Open questions

1. ~~Where does an engine run, so that cancelling the job frees its VRAM and Slurm can account its GPU use?~~ As the job's own process (S2, D6).
2. ~~Does `slurmd` need a rootful container?~~ No; rootless with a cgroup nesting step (S1).
3. ~~How does `gpu_nvml` get into the node?~~ devai builds the whole Slurm package set with the NVML headers from SchedMD's recipe (S3).
4. ~~Does a real engine serve from inside a job?~~ Yes, vLLM and Ollama (S2, S6).
5. ~~One Slurm build per distribution?~~ One build: everything is Debian trixie (D9).
6. Does SGLang 0.5.16 from PyPI run on trixie with this GPU? Its upstream image carries extras beyond the pip dependencies. Keep SGLang only if it passes the re-probe? -- recommendation: yes.
7. Does the laya trainer run without system CUDA (its base is `nvidia/cuda` Ubuntu today)?
8. Which Python per venv: vLLM's build uses Debian's 3.13; SGLang, laya and the workload clients may need uv's 3.14.

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

The full proposed architecture, with a diagram of every component and interaction, is [docs/slurm.md](../slurm.md). In short:

- **One image, `devai-engines`,** on `debian:trixie-slim`: devai's Slurm build (all daemons, with `gpu_nvml`), MariaDB and supervisor from Debian, and every backend -- Ollama (our build), vLLM 0.28 + HyperQwen (our build), SGLang (PyPI, hash-locked), the laya trainer -- each engine in its own venv, plus a venv for workload clients (bench, probes).
- **One privileged container** from it (`--privileged --cgroupns=private --pid=host`, GPU via CDI), on `devai-net` only. supervisord runs MariaDB, slurmdbd, slurmctld, slurmrestd and slurmd: a single-node cluster. The node has `Gres=gpu:1`, so Slurm runs one GPU job at a time without a license.
- **Jobs:** engine (router only; `exec`s the backend's server on its fixed port), trainer, probe (GPU); workload (bench and test clients; no GPU; holds its engine through the router).
- **The router stays a separate, unprivileged container:** it handles the lab's traffic and must not run inside a container that can signal every host process. Its launch layer becomes a Slurm client (REST + self-signed JWT); requests still go straight to the engines.
- **The GPU guard** is the prolog of every GPU job: idle card, or kill the holder, or drain the node and hold the job. It also wipes vLLM's compile cache.
- **History:** slurmdbd plus `/var/cache/devai/jobs/results/<jobid>/`, joined by `devai-jobs`.

Code we expect to delete: `scripts/bench/clean_slate.py`, the deadline handling in `scripts/bench-sync.py`, the router's podman launch layer and job-runner hold, five engine compose services and eight engine images.

---

## Phase 0 -- Spike

### Goal

Answer the questions that decide the design before writing anything permanent.

### Questions

- **S1** -- Do the Slurm daemons run under rootless podman with cgroup v2 (`IgnoreSystemd=yes`, no systemd in the container), see the GPU as `gres/gpu:1`, and run a GPU job?
- **S2** -- Where does the engine run, so that `scancel` leaves VRAM empty? Must hold: no container inside a container (D6).
- **S3** -- GPU accounting: do `sstat` / `sacct` show `gres/gpumem` and `gres/gpuutil` for the engine? Energy?
- **S4** -- slurmrestd reachable from `devai-net` with JWT auth, the router's path; submit-to-running latency.
- **S5** -- queueing, suspend/resume, hold/release, requeue, cancel, clearing the queue, and D1's two interruption paths.
- **S6** -- an engine switch between two backends' nodes under the license.
- **S7** -- the 2026-09-28 failure: a process outside Slurm holds VRAM when an engine job is submitted.

### Deliverables

Findings written into this plan, each answer with the command that showed it. Throwaway config lives in the session scratch directory, not in the tree.

### Exit criteria

S1-S7 answered; Open questions closed; Phase 1 deliverables fixed.

### Spike results, round 2 (2026-09-29, GPU free, no nested containers)

Round 2 ran one Slurm node per backend image. D9 replaced that with a single node, so its multi-node findings -- the `gpu0` license, fixed node addresses, configless nodes, per-distribution builds, the race between one node's epilog and another node's prolog -- no longer apply. Everything about a job on a node (cgroup, `exec`, accounting, the guard, REST, timings) does.

Stack: SchedMD's official `slurmctld`, `slurmdbd`, `slurmrestd` images (Ubuntu 26.04, Slurm 26.05.4), MariaDB 11.8, and three nodes -- `vllm-devai` (from `docker.io/devai/vllm-devai`, +~90 MB), `ollama` (from `localhost/devai-ollama`, +82 MB) and a GPU-less work node -- all on `devai-net` at fixed IPs, the engine nodes built with devai's own Slurm packages for Debian trixie (below). `make cache-down` had stopped the router and every engine, and the card was idle (2 MiB). Engine under test: `Qwen3.8-27B-W4A16-devai-AutoRound` @131072 with exactly the arguments and environment of the router's last launch (vLLM's own "non-default args" line, `deploy/recovery-flags.json`, the backend's `EnvVars`); and `qwen3.8:27b-mtp-q4_K_M` on Ollama.

- **S1 -- yes, rootless.** Controller, REST and accounting daemons run unprivileged; nodes run `--privileged --cgroupns=private --pid=host` with the GPU through CDI. Needed:
  - an entrypoint step that moves the container's processes out of its cgroup root and enables the delegated controllers (without it: "Controller memory is not enabled"); jobs then get their own cgroup (`.../slurmstepd.scope/<job>/step_0/user/task_0`) with `CUDA_VISIBLE_DEVICES=0`;
  - run directories created for the `slurm` user in the controller and accounting containers (they drop privileges and cannot create them);
  - fixed IPs: a node recreated with a new IP went NOT_RESPONDING until the controller restarted (round 1); on a fixed IP it re-registered by itself;
  - configless nodes (`SlurmctldParameters=enable_configless`, `slurmd --conf-server`): in round 1 a changed `slurm.conf` drained the node;
  - `cpuset` is not delegated to user sessions here (`DelegateControllers=cpu memory pids`), so no core pinning; `slurmd -C` counts 8 of 24 CPUs, so `CPUs=` is set in the config with `SlurmdParameters=config_overrides`.
- **S2 -- the engine as the job's own process on its backend's node.** The engine job `exec`s `python3 -m vllm...` (or `ollama serve`); `VLLM::EngineCore` (21.4 GiB) and Ollama's `llama-server` (18.6 GiB) sat in the job's cgroup. vLLM: submit -> RUNNING 1.0 s, `/health` 95.5 s after submit (a cold start). A chat request over `devai-net` to `spike-node-vllm-devai:11434` returned 1,200 tokens at 38.6 tok/s (MTP off). `scancel` -> no GPU process **0.30 s**, vLLM shut down gracefully, 4 MiB left. Two things matter:
  - a job carries only the environment its submitter gives it, so a REST-submitted vLLM job failed with `No module named 'vllm'` (the image's `PATH` includes `/opt/vllm/bin`); the node entrypoint now writes its image environment to `/run/devai/node-env.sh` and every engine script sources it, so the router passes only what is job-specific;
  - the node container outlives its jobs, so vLLM's compile caches did too: a restart on the same node reached `/health` in 27-29 s instead of 95 s. That is the cache docs/router.md says must NOT persist (it under-measured activation memory and OOM-killed engines, 2026-09-23). The prolog has to wipe `/root/.cache/vllm` and `/tmp/torchinductor_*` before an engine job (the FlashInfer cache stays, as today).
- **S3 -- GPU memory and utilisation per job, yes; energy, no.**
  - The official `slurmd` lacks `gpu_nvml`, and the plugin file alone is not enough (NVML autodetection is compiled into Slurm's core: "configured to autodetect nvml functionality, but we weren't able to find that lib when Slurm was configured"). devai builds the whole package set with the NVML headers: SchedMD's `debuild` step, source `slurm-26-05-4-1` (sha256 `0e522d39324b7b7da5e8096c678c4af00500ca4c3fe2e6da7e4f8d01f7082ec7`), on `debian:trixie` with `libnvidia-ml-dev` from non-free, for the trixie-based engine images.
  - GPU TRES need slurmdbd ("slurmdbd is required to run with TRES gres/gpu"). MariaDB 11.8 enables `innodb_snapshot_isolation`, which slurmdbd warns will make it fatal on a write conflict: `--innodb-snapshot-isolation=OFF`.
  - Nodes need `--pid=host`: NVML reports host PIDs and Slurm matches them against the job's processes, summed over the task's process tree -- which the engine is part of, because the job `exec`s it.
  - Result: vLLM job `gres/gpumem=21794M`, `gres/gpuutil=99-100` live in `sstat` during generation and in `sacct` afterwards; Ollama job `gres/gpumem=18570M`.
  - Energy: Slurm 26.05's NVML plugin does not read it (`gpu_p_energy_read` returns success with no data), so `acct_gather_energy/gpu` reports 0 on NVIDIA. NVML's own counter works (`nvmlDeviceGetTotalEnergyConsumption` read through `pynvml` in the vLLM node: 14,323,881,481 mJ since driver load); `nvidia-smi` on this driver does not expose it. devai reads it at job start and end, which needs a small NVML reader in every node image (the Ollama image has no Python).
  - `AccountingStoreFlags=job_comment` is needed for comments to reach `sacct`. A JSON comment submitted through REST arrives intact; `#SBATCH --comment` strips the quotes.
- **S4 -- yes.** slurmrestd (run as `nobody`, `-a rest_auth/jwt`, token from `scontrol token`), called from another `devai-net` container. Round 1: submit -> RUNNING 0.9 / 1.4 / 2.9 s, DELETE -> CANCELLED 15-19 ms. Round 2 (the real engine through REST): submit -> RUNNING 1.26 s, `/health` 95.3 s.
- **S5 -- all work; a suspended job keeps its GPU.**
  - A second engine job waits as `PENDING (Licenses)` while one holds `gpu0`; a deferred job (`--begin=now+1hour`) waits as `BeginTime`; `scontrol hold` / `release`, `scontrol top` (move to the front), `scancel` of one job and `scancel --state=PENDING` (clear the queue, running job untouched) behave as documented.
  - Round 1: a suspended GPU job kept its allocation -- a queued GPU job stayed `PENDING (Resources)` throughout.
  - Same-engine interruption: a bench-like workload (20 requests to vLLM) suspended after 3, a one-request job ran on the same engine, the workload resumed -- **2.9 s** in all; no request while suspended (3 -> 3), 20/20 done, 0 retries; `sacct` records the suspended time.
  - Different-engine interruption: workload suspended, vLLM cancelled and Ollama started (3.0 s), a quick job on Ollama including its model load (done at 10.4 s), Ollama cancelled and vLLM resubmitted, healthy at 37.6 s (27 s of it the start that the cache wipe above will lengthen to ~95 s), workload resumed: 20/20 done, 0 retries.
  - A job held after a prolog failure (S7) started 130 s after release: Slurm delays requeued jobs (reason `BeginTime`, also seen in round 1). The router should cancel and resubmit instead; a fresh job started in ~1 s every time.
- **S6 -- engine switch under the license, through REST.** vLLM running, Ollama job pending on `Licenses`; cancel vLLM -> vLLM off the GPU 0.23 s -> Ollama job RUNNING 1.22 s -> Ollama answering 2.79 s. The epilog confirmed vLLM left the card idle and Ollama's prolog confirmed it before starting.
- **S7 -- the 2026-09-28 failure cannot recur.** A container outside Slurm allocated 4 GiB; an engine job submitted through REST was refused: the prolog waited 30 s ("gpu NOT idle after 30s: 4328 MiB used; holders: 723061, python3, 4318 MiB"), Slurm drained the node ("Prolog error") and held the job, and no vLLM process ever reached the GPU. After the holder was removed, the node resumed and the job released, it started and served.
  - Race to design out: the previous engine's epilog and the next engine's prolog ran in the same second (13:27:28) on two nodes. Both saw an idle card here, but an epilog that demands a globally idle card can see the next engine already allocating and drain its own node. So the prolog carries the global check and the epilog checks the finished job's own leftovers only.

Round 1 (earlier the same day, the router's engine holding 21.8 GiB) ran engines as podman containers nested inside the slurmd container. It worked -- VRAM released 0.2 s after `scancel` with `tini` as the container's init, GPU accounting after a `tini -s` subreaper -- but needed the host's `nvidia-cdi-hook` and image store inside the node, and the operator ruled out containers inside containers (D6). Its S1, S4, S5 and S3 findings on the Slurm side carried over and are folded in above.

Open questions 1-4 are answered.

---

## Phase 1 -- The engines image and container

`deploy/engines/`: the Dockerfile (devai's Slurm packages, MariaDB, supervisor, the four backends, the workload venv), Slurm and supervisord config, entrypoint and guard scripts; `make build-engines`, `make engines-init` (keys, MariaDB password), the `/var/cache/devai/jobs` volume, backups. Then re-probe every vLLM row on 0.28 and every SGLang row on the image's SGLang, and drop what fails (D8). Exit: a test job's GPU memory and utilisation appear in `sacct`, a stray GPU holder is killed or drains the node, and the re-probed fleet is known.

## Phase 2 -- Router as Slurm client

The router launches and stops engines through slurmrestd, proxies to `devai-engines:<port>`, and takes the GPU holder from Slurm. Refusals come from Slurm's recorded outcome, so a start refused on a busy card no longer counts against the model. Port 11437 is retired (one vLLM on 11435). Exit: the 2026-09-28 "free memory" failure cannot be reproduced.

## Phase 3 -- Workloads, trainer, probes as jobs

Bench (the 30-minute limit as a Slurm time limit), probes and the laya trainer as Slurm jobs; holds; queue, suspend/resume, cancel and clear-queue commands (D1). aiagent's fine-tuning API on :11438 stays. Exit: a bench and an interactive session run side by side without contention; interrupting works with the same and with another engine.

## Phase 4 -- History

The result format, `devai-jobs`, optionally an MCP tool. The bench leaderboard records each task's job id. Exit: "what happened yesterday" is one command.

## Phase 5 -- Cleanup

Remove the five engine compose services, the eight engine images and port 11437; the GPU device from the lab; the code listed under Approach; update router.md, backends.md, bench-results.md, CLAUDE.md. Exit: no devai application container has a GPU device; `make test-router` and `make test-python` pass.

---

## Combined risk register

| Risk | Phase | Mitigation |
| ---- | ----- | ---------- |
| SGLang from PyPI does not run on trixie | 1 | the re-probe decides; SGLang is dropped if it fails (Open question 6) |
| the laya trainer needs system CUDA | 1 | install NVIDIA's CUDA runtime packages for debian13 in the image (Open question 7) |
| models lost to the single vLLM / SGLang version | 1 | accepted by D8; the re-probe lists them |
| vLLM compile caches outlive the job in the long-lived container (the 2026-09-23 OOMs) | 1 | the prolog wipes them before every engine job |
| Slurm reports no GPU energy on NVIDIA | 1 | devai reads NVML's energy counter at job start and end |
| one container is one failure domain: restarting it kills running jobs | 1 | the router stays up and resubmits on demand; supervisord restarts a single failed daemon |
| a job held after a prolog failure starts only after Slurm's requeue delay | 2 | the router cancels and resubmits |
| suspend with another engine costs a cold start (about 95 s for Qwen3.8-27B) | 3 | the same-engine fast path; operators choose |

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
