# Slurm gatekeeper

_Slurm, in containers, becomes the only thing that starts GPU work in devai; the router becomes its client; every job leaves a central record with its results and GPU accounting._

## Status

Draft. Third layout proposed 2026-09-30 (D7-D12); nothing of it is built or tested yet. An earlier spike (2026-09-29) tested two layouts since discarded and is obsolete (see Context).

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
- **D6 -- No containers inside containers.** Ruled out after spike round 1, which ran engines as podman containers nested in the slurmd container. (It also meant, until D10, that an engine ran as the job's own process; D10 replaced that with `podman exec` into a sibling container, which is still not nesting.)

## Operator decisions (2026-09-30, second round)

The first architecture (one Slurm node per backend image) was judged far too complicated; a second one (Slurm and every backend in one image) mixed Slurm with the backends.

- **D7 -- Fewer moving parts, above all fewer images.** Follow the image-reduction plan: `debian:trixie` is the only upstream image, everything else is built here. For this plan: Slurm's daemons come from devai's own Slurm build and MariaDB from Debian, which **supersedes D4**.
- **D8 -- One version per backend.** One vLLM (0.28.0 + HyperQwen), one SGLang, one Ollama, one laya trainer; a model that does not run on its backend's version is dropped. One vLLM backend, `vllm` on 11435; `vllm-devai` and 11437 are retired.
- **D9 -- The backends live in one Debian trixie image, `devai-engines`, with no Slurm in it.** The laya trainer keeps its own image, also without Slurm.
- **D10 -- All of Slurm -- slurmctld, slurmdbd, slurmrestd, MariaDB and slurmd -- in one image, `devai-slurm`, run as one container.** A job starts its process inside the backend's container with `podman exec`. Chosen over putting slurmd into the backends (the trade-off: Slurm then owns only the `podman exec` client, not the engine).
- **D12 -- Slurm from Debian trixie's own packages** (2026-09-30): `slurmctld`, `slurmd`, `slurmdbd`, `slurmrestd`, `slurm-wlm-jwt-plugin`, `slurm-wlm-mysql-plugin`, version 24.11.5; no source build. Slurm no longer accounts GPUs itself (D10), so it needs no NVML build, and nothing in the design needs 26.05.
- **D11 -- CUDA 13.1 only, everything compiled here** (the engines and their own CUDA kernels; torch, FlashInfer, Triton stay pinned, hash-locked PyPI packages). SGLang 0.5.16 is compiled here, with `sglang-kernel` built for sm120. (Recorded in the image-reduction session and relayed from there.)

## Open questions

1. ~~Where does an engine run?~~ In the backend's own container, started by the job with `podman exec` (D9, D10).
2. Does the exec path hold up: exit status, long runs, start-up overhead, the trap and kill helpers, suspend through `STOP` / `CONT`, the GPU sampler? -- Phase 1 tests it first.
3. ~~Slurm from Debian's packages or built from source?~~ Debian's packages, 24.11.5 (D12).
4. inspect's per-sample clock keeps running while a bench is suspended: suspend between samples, or accept it?
5. `devai-workload` is a new permanent container (from `devai-lab`, so no new image). Accept it, or run workloads somewhere else?

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

- **`devai-slurm`, one image and one container:** Slurm 24.11.5, MariaDB, supervisor and podman, all from Debian's packages (D12). A single-node cluster with one license, `engine:1`, which every engine, trainer and probe job takes, so one runs at a time -- with or without a GPU (M15). It is privileged, has `--pid=host`, holds the host's podman socket, and gets the GPU (for NVML only) when the host has one.
- **Backends carry no Slurm:** `devai-engines` (the image-reduction plan) and `devai-laya-trainer` are permanent, idle containers; `devai-workload` (the lab image) runs bench and test clients.
- **Jobs:** each job script runs `podman exec <container> devai-run <jobid> <command>` and stops it with `devai-kill <jobid> TERM`, then `KILL` (two small devai scripts in each target image). Kinds: engine (router only), trainer, probe (GPU); workload (no GPU; holds its engine through the router).
- **What Slurm no longer does itself:** it owns only the local `podman exec` client, so cancel, suspend and GPU accounting of the remote process are devai's scripts -- the trap, `devai-kill`, the epilog backstop, and an NVML sampler writing `gpu.json` and the job's `AdminComment`.
- **The router** stays a separate, unprivileged container, gives up its podman socket, and becomes a Slurm client (REST + self-signed JWT); requests still go straight to the engines.
- **The GPU guard** is the prolog of every GPU job: idle card, or kill the holder, or drain the node and hold the job. It also wipes vLLM's compile cache inside `devai-engines`.
- **History:** slurmdbd plus `/var/cache/devai/jobs/results/<jobid>/`, joined by `devai-jobs`.

Code we expect to delete: `scripts/bench/clean_slate.py`, the deadline handling in `scripts/bench-sync.py`, the router's podman launch layer and job-runner hold.

---

## Phase 1 -- devai-slurm and the exec path

First, test the exec path: a job that execs an engine into a stand-in container, is cancelled, times out, is suspended and resumed, and whose GPU use the sampler records. Then `deploy/slurm/`: the Dockerfile (Debian's Slurm, MariaDB, supervisor and podman packages), Slurm and supervisord config, the entrypoint, the guard and the sampler; `devai-run` / `devai-kill` (source in `deploy/slurm/`) and their COPY lines in `deploy/Dockerfile.engines` and `deploy/Dockerfile.laya-trainer` (agreed with the image-reduction plan: its Phase 4, step 4c); the image on the one pinned `debian:trixie-slim` variable (its M13); `make build-slurm`, `make slurm-init` (keys and MariaDB password as plain 0600 files; no sops, which goes to the attic, M12), the `/var/cache/devai/jobs` volume, backups. Exit: an engine job starts and stops through `podman exec`, its GPU summary appears in `sacct`, a stray GPU holder is killed or drains the node, and on a host without a GPU the same stack runs Ollama and trainer jobs with the guard and the sampler switched off.

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
| 1 | 2-3 days |
| 2 | 3-5 days |
| 3 | 3-5 days |
| 4 | 1-2 days |
| 5 | 1 day |

Estimates, not measurements.

## References

- Slurm documentation: slurm.conf (`Licenses`, `CompleteWait`, `KillWait`), cgroup.conf (`IgnoreSystemd`), accounting, slurmrestd, rest_api (JWT) -- slurm.schedmd.com
- [docs/slurm.md](../slurm.md) (the architecture), docs/router.md, docs/bench-results.md "Run protocol", gpu-arbiter/main.go (`ensureBackendRunning`, `stopOtherBackends`)
