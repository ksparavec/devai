# Slurm gatekeeper

_Slurm, in containers, becomes the only thing that starts GPU work in devai; the router becomes its client; every job leaves a central record with its results and GPU accounting._

## Status

In Progress -- Phase 0 (spike) done 2026-09-29. **Design revised 2026-09-30 (D7-D11):** Slurm in its own image and container, no Slurm in the backends; jobs start their process in the backend's container with `podman exec` (not yet tested, S8). Decisions D1-D11 below are the operator's.

## Dependencies

- [Plan: minimal-external-images](./minimal-external-images.md) -- builds `devai-engines` (Ollama, vLLM 0.28 + HyperQwen, SGLang 0.5.16; CUDA 13.1; everything compiled here; no Slurm), switches today's router to it, renames `vllm-devai` to `vllm`, and re-probes the fleet. This plan adds Slurm around it. Its decisions M9-M13 apply here too: Open WebUI and MCP dropped, AMD/ROCm and the sops/age scaffold to the attic, one digest-pinned `debian:trixie-slim` for every image.

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
- **D4 -- Use a published image if one exists.** *(Superseded by D7 on 2026-09-30.)* Docker Hub has none that is maintained (2026-09-29: the `slurm/` namespace holds two repositories last updated in 2017 and 2018; the rest are personal images). SchedMD, Slurm's maintainer, publishes official images on GHCR (`ghcr.io/slinkyproject/{slurmctld,slurmd,slurmdbd,slurmrestd}:26.05-ubuntu26.04`, Slurm 26.05.4, recipe github.com/SlinkyProject/containers). Its `slurmd` is built without the `gpu_nvml` plugin, which GPU accounting needs (see S3).
- **D5 -- Not a separate project.** One consumer, contracts that change together with the router and the workloads, no product without devai; a second repository would make every change a cross-repo release (the aiagent experience). It stays extractable: own directory, own tests, config under `deploy/slurm/`, a written job/result format.
- **D6 -- No containers inside containers.** Ruled out after spike round 1, which ran engines as podman containers nested in the slurmd container. (It also meant, until D10, that an engine ran as the job's own process; D10 replaced that with `podman exec` into a sibling container, which is still not nesting.)

## Operator decisions (2026-09-30, second round)

The first architecture (one Slurm node per backend image) was judged far too complicated; a second one (Slurm and every backend in one image) mixed Slurm with the backends.

- **D7 -- Fewer moving parts, above all fewer images.** Follow the image-reduction plan: `debian:trixie` is the only upstream image, everything else is built here. For this plan: Slurm's daemons come from devai's own Slurm build and MariaDB from Debian, which **supersedes D4**.
- **D8 -- One version per backend.** One vLLM (0.28.0 + HyperQwen), one SGLang, one Ollama, one laya trainer; a model that does not run on its backend's version is dropped. One vLLM backend, `vllm` on 11435; `vllm-devai` and 11437 are retired.
- **D9 -- The backends live in one Debian trixie image, `devai-engines`, with no Slurm in it.** The laya trainer keeps its own image, also without Slurm.
- **D10 -- All of Slurm -- slurmctld, slurmdbd, slurmrestd, MariaDB and slurmd -- in one image, `devai-slurm`, run as one container.** A job starts its process inside the backend's container with `podman exec`. Chosen over putting slurmd into the backends (the trade-off: Slurm then owns only the `podman exec` client, not the engine).
- **D11 -- CUDA 13.1 only, everything compiled here** (the engines and their own CUDA kernels; torch, FlashInfer, Triton stay pinned, hash-locked PyPI packages). SGLang 0.5.16 is compiled here, with `sglang-kernel` built for sm120. (Recorded in the image-reduction session and relayed from there.)

## Open questions

1. ~~Where does an engine run?~~ In the backend's own container, started by the job with `podman exec` (D9, D10).
2. ~~Does `slurmd` need a rootful container?~~ No; rootless with a cgroup nesting step (S1).
3. ~~How does `gpu_nvml` get into the build?~~ devai builds the whole Slurm package set with the NVML headers from SchedMD's recipe (S3).
4. ~~Does a real engine serve under Slurm?~~ Yes, as a job's own process (S2, S6); through `podman exec` it is untested (S8).
5. ~~One Slurm build per distribution?~~ One build, Debian trixie.
6. Does the exec path hold up (S8): exit status, long runs, start-up overhead, the trap and kill helpers, suspend through `STOP` / `CONT`, the GPU sampler?
7. inspect's per-sample clock keeps running while a bench is suspended: suspend between samples, or accept it?
8. `devai-workload` is a new permanent container (from the lab image, so no new image). Accept it, or run workloads somewhere else?

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

- **`devai-slurm`, one image and one container:** devai's Slurm build (all daemons, with `gpu_nvml`) plus MariaDB, supervisor and podman from Debian. A single-node cluster whose node has `Gres=gpu:1`, so Slurm runs one GPU job at a time. It is privileged, has `--pid=host` and the GPU for NVML, and holds the host's podman socket.
- **Backends carry no Slurm:** `devai-engines` (the image-reduction plan) and `devai-laya-trainer` are permanent, idle containers; `devai-workload` (the lab image) runs bench and test clients.
- **Jobs:** each job script runs `podman exec <container> devai-run <jobid> <command>` and stops it with `devai-kill <jobid> TERM`, then `KILL` (two small devai scripts in each target image). Kinds: engine (router only), trainer, probe (GPU); workload (no GPU; holds its engine through the router).
- **What Slurm no longer does itself:** it owns only the local `podman exec` client, so cancel, suspend and GPU accounting of the remote process are devai's scripts -- the trap, `devai-kill`, the epilog backstop, and an NVML sampler writing `gpu.json` and the job's `AdminComment`.
- **The router** stays a separate, unprivileged container, gives up its podman socket, and becomes a Slurm client (REST + self-signed JWT); requests still go straight to the engines.
- **The GPU guard** is the prolog of every GPU job: idle card, or kill the holder, or drain the node and hold the job. It also wipes vLLM's compile cache inside `devai-engines`.
- **History:** slurmdbd plus `/var/cache/devai/jobs/results/<jobid>/`, joined by `devai-jobs`.

Code we expect to delete: `scripts/bench/clean_slate.py`, the deadline handling in `scripts/bench-sync.py`, the router's podman launch layer and job-runner hold.

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

Round 2 ran one Slurm node per backend image, with each engine as the job's own process. D9-D10 replaced both: one node, and engines started with `podman exec` in the backend's container. So its multi-node findings (the `gpu0` license, fixed node addresses, configless nodes, per-distribution builds, the cross-node epilog/prolog race) no longer apply, and neither do its process results: `scancel` freeing the GPU in 0.23-0.30 s and Slurm accounting `gres/gpumem` came from the engine being in the job's cgroup and process tree, which it no longer is. What carries over: Slurm under rootless podman, the cgroup nesting step, NVML in the Slurm build, REST with a self-signed JWT, the queue commands, a suspended job keeping its GPU, the guard killing an outside holder, and the timing of the engines themselves. **S8 (to do): the `podman exec` path end to end.**

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

## Phase 1 -- devai-slurm and the exec path

First S8: a job that execs an engine into a stand-in container, is cancelled, times out, is suspended and resumed, and whose GPU use the sampler records. Then `deploy/slurm/`: the Dockerfile (devai's Slurm packages, MariaDB, supervisor, podman), Slurm and supervisord config, the entrypoint, the guard and the sampler; `devai-run` / `devai-kill` (source in `deploy/slurm/`) and their COPY lines in `deploy/Dockerfile.engines` and `deploy/Dockerfile.laya-trainer` (agreed with the image-reduction plan: its Phase 4, step 4c); the image on the one pinned `debian:trixie-slim` variable (its M13); `make build-slurm`, `make slurm-init` (keys and MariaDB password as plain 0600 files; no sops, which goes to the attic, M12), the `/var/cache/devai/jobs` volume, backups. Exit: an engine job starts and stops through `podman exec`, its GPU summary appears in `sacct`, and a stray GPU holder is killed or drains the node.

## Phase 2 -- Router as Slurm client

The router launches and stops engines through slurmrestd, gives up its podman socket, and takes the GPU holder from Slurm. A start refused on a busy card no longer counts against the model. Exit: the 2026-09-28 "free memory" failure cannot be reproduced.

## Phase 3 -- Workloads, trainer, probes as jobs

Bench (the 30-minute limit as a Slurm time limit), probes and the laya trainer as Slurm jobs; `devai-workload`; holds; queue, suspend/resume, cancel and clear-queue commands (D1). aiagent's fine-tuning API on :11438 stays. Exit: a bench and an interactive session run side by side without contention; interrupting works with the same and with another engine.

## Phase 4 -- History

The result format and `devai-jobs` (no MCP tool: the image-reduction plan drops MCP, M10). The bench leaderboard records each task's job id. Exit: "what happened yesterday" is one command.

## Phase 5 -- Cleanup

The router's podman launch code and job-runner hold, `clean_slate.py`, the bench deadline code; the GPU device in the lab; router.md, backends.md, bench-results.md, CLAUDE.md. Exit: no devai application container has a GPU device; `make test-router` and `make test-python` pass.

---

## Combined risk register

| Risk | Phase | Mitigation |
| ---- | ----- | ---------- |
| the exec path loses the engine on cancel (the job script is SIGKILLed before its trap runs) | 1 | `KillWait=30` > the trap's 10 s; the epilog's `devai-kill ... KILL`; the guard's idle check |
| GPU accounting by sampling is coarser than Slurm's own | 1 | one GPU job at a time, so the whole card is the job's; 5 s samples plus NVML's energy counter |
| `devai-slurm` restarts and orphans engines | 1 | the entrypoint kills every recorded process group before slurmd starts |
| the podman socket makes job submission equal to control of the host user | 1-2 | the JWT key only in the router (read-only) and a 0600 host file; `devai-slurm` on `devai-net` only; the router gives up its own socket |
| vLLM compile caches outlive the job in the long-lived `devai-engines` (the 2026-09-23 OOMs) | 1 | the prolog wipes them before every vLLM job |
| Slurm reports no GPU energy on NVIDIA | 1 | devai reads NVML's energy counter at job start and end |
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
