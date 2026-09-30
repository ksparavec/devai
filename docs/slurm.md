# Slurm gatekeeper -- architecture

**Status: proposed (2026-09-30), not built.** This is the design the phases of
[the plan](plans/slurm-gatekeeper.md) implement; the plan holds the operator's
decisions (D1-D6), the phase breakdown and the Phase 0 spike. Statements marked
**(measured)** were measured on this host in that spike (plan, "Spike results,
round 2") or in the follow-up checks of 2026-09-30; everything else is
proposal. Once built, this document is the source of truth for Slurm in devai,
as [router.md](router.md) is for the router.

## 1. What changes

Today the router starts engines itself. For every model switch it recreates a
podman container, and it decides what holds the GPU from its own flags. It never
looks at the card, so an engine can start on a GPU another process still holds
-- the failure that cost the re-bench of 2026-09-28 a day (plan, "Context").

Proposed: **Slurm is the only thing that starts GPU work.** The router remains
the only entry point for inference, but its launch layer becomes a Slurm
client: it asks Slurm to run an *engine job* and proxies requests to that
engine as before. Slurm then

- runs one GPU job at a time (a cluster-wide license),
- refuses to start a GPU job on a card someone else holds, after first trying
  to kill the holder,
- kills a job's whole process tree when it is cancelled,
- queues, holds, suspends and resumes work, and
- records every job with its GPU memory and utilisation, in one database.

devai application containers (the lab, its agents, Open WebUI) talk to the
router and never to the GPU (D1). Nothing runs a container inside a container
(D6): an engine is the job's own process, on a Slurm node that *is* the
backend's engine image. Slurm runs in containers (D2), on this single host;
Slurm's multi-host features are not used.

## 2. Architecture at a glance

![Slurm gatekeeper architecture](slurm-architecture.svg)

Source: [`scripts/diagrams/slurm_architecture.py`](../scripts/diagrams/slurm_architecture.py)
(explicit coordinates; stdlib only). Re-render with:

```bash
python3 scripts/diagrams/slurm_architecture.py    # writes docs/slurm-architecture.svg
```

### Interactions

The numbers are the ones on the arrows.

| # | From -> to | What | Protocol, auth | Status |
|---|---|---|---|---|
| 1 | lab agents, Open WebUI, aiagent -> router | inference (OpenAI `/v1/chat/completions`, `/v1/responses`, Anthropic `/v1/messages`, Ollama `/api/*`); fine-tuning jobs on :11438 | HTTP on `devai-net` / `devai-lab-egress` | unchanged |
| 2 | router -> engine | proxied requests to `<node container>:11434` | HTTP on `devai-net` | measured (S2: a job's vLLM served a chat request over `devai-net`) |
| 3 | router -> slurmrestd | submit, cancel, signal, job and node state | REST v0.0.44; JWT the router signs itself | measured (S4; self-signed JWT 2026-09-30) |
| 4 | slurmrestd -> slurmctld | the same, as Slurm RPC | `auth/slurm`, shared `slurm.key` | measured |
| 5 | slurmctld <-> slurmd | launch, signal, kill; configless config; node registration and health | Slurm RPC, `auth/slurm` | measured (S1) |
| 6 | slurmctld -> slurmdbd | accounting records | Slurm RPC | measured |
| 7 | slurmdbd -> MariaDB | SQL | MariaDB user + password | measured |
| 8 | engine job -> GPU | CUDA; the guard and accounting read NVML | device via CDI | measured |
| 9 | workload job -> router | inference requests; hold / release calls | HTTP | requests measured (S5); hold API proposed (Sec. 7.4) |
| 10 | operator (`devai-jobs`) -> slurmrestd, slurmctld, `jobs/` | queue and history (REST); `scontrol suspend` / `resume`; result files | REST + JWT; `podman exec`; files | proposed (`scontrol suspend` measured in S5) |
| 11 | devai-model-status (MCP) -> slurmrestd | job history | REST + JWT | proposed |
| 12 | GPU guard -> stray process | SIGTERM, then SIGKILL; if still busy, drain the node | signals (`--pid=host`) | measured (kill 2026-09-30; drain S7) |
| 13 | host volumes -> nodes | model stores, engine caches, vLLM parser plugins | bind mounts, named volumes | model store and FlashInfer mounts measured |
| 14 | nodes -> `jobs/results` | job logs, `result.json`, `gpu.json` | files | proposed |
| 15 | control plane -> `jobs/` | MariaDB files, slurmctld state | files | proposed |

## 3. Components

### 3.1 Control plane

SchedMD's own images (Slurm's maintainer; D4 -- Docker Hub has no maintained
Slurm image), digest-pinned, with a two-line entrypoint wrapper where a daemon
drops to the `slurm` user and cannot create its run directory under podman
(measured).

| Container | Image | Address | Role | Data |
|---|---|---|---|---|
| `devai-slurmctld` | `ghcr.io/slinkyproject/slurmctld:26.05-ubuntu26.04` | 10.89.0.200:6817 | queue, scheduling, license `gpu0`, priorities; serves the nodes' config (configless) | state in `/var/cache/devai/jobs/slurmctld/` |
| `devai-slurmdbd` | `.../slurmdbd:26.05-ubuntu26.04` | 10.89.0.201:6819 | accounting | -- |
| `devai-slurm-db` | `docker.io/library/mariadb:11.8` | 10.89.0.203:3306 | accounting database | `/var/cache/devai/jobs/mariadb/` |
| `devai-slurmrestd` | `.../slurmrestd:26.05-ubuntu26.04` | 10.89.0.202:6820 | REST API, `rest_auth/jwt`, runs as `nobody` | -- |

All four are unprivileged and on `devai-net` only. MariaDB runs with
`--innodb-snapshot-isolation=OFF` (11.8 enables it by default and slurmdbd
warns it will then die on a write conflict; measured) and
`--innodb-lock-wait-timeout=900`; slurmdbd flags a 1 GiB buffer pool as below
its recommendation (measured), so the pool size is a Phase 1 setting.

Fixed addresses are required: a node recreated with a new IP went
NOT_RESPONDING until slurmctld restarted, while on a fixed IP it re-registered
by itself (measured). The Slurm block of `devai-net` is .200-.229.

### 3.2 Nodes

A node is a backend's engine image with Slurm added. The node container keeps
the **name of today's backend container**, so the router's proxy targets
(`http://devai-<backend>:11434`) do not change; what changes is that the
container runs `slurmd` permanently instead of a `sleep infinity` placeholder
the router recreates.

| Node container | Built from | Base | Engine or role | Address |
|---|---|---|---|---|
| `devai-ollama` | `localhost/devai-ollama` | Debian trixie | `ollama serve` | .210 |
| `devai-vllm` | `vllm/vllm-openai:v0.22.1-x86_64-cu129-ubuntu2404` | Ubuntu 24.04 | vLLM 0.22.1 | .211 |
| `devai-vllm-devai` | `docker.io/devai/vllm-devai` | Debian trixie | vLLM 0.28 + HyperQwen | .212 |
| `devai-sglang` | `lmsysorg/sglang:v0.5.16-cu130` | Ubuntu 24.04 | SGLang 0.5.16 | .213 |
| `devai-laya-trainer` | `localhost/devai-laya-trainer` | Ubuntu 24.04 | laya fine-tuning | .214 |
| `devai-vllm-v0251` | `vllm/vllm-openai:v0.25.1-x86_64-cu129-ubuntu2404` | Ubuntu 24.04 | per-model image override (Nemotron-Nano-9B-v2) | .215 |
| `devai-vllm-gemma` | `vllm/vllm-openai:gemma-x86_64-cu130` | Ubuntu 22.04 | per-model image override (diffusiongemma) | .216 |
| `devai-slurm-work` | `localhost/devai-lab-cpu` | Debian trixie | workload jobs (bench, probe clients, tests); no GPU | .220 |

A node's image is fixed, so each distinct engine image is its own node: the two
`image` overrides in `deploy/recovery-flags.json` become two small extra nodes
(an idle `slurmd` costs next to nothing). Bases were checked 2026-09-30; the
gemma image's Ubuntu 22.04 would need a third Slurm build for one model (Open
point 1).

**Image contract** (`deploy/slurm/Dockerfile.node`, `ARG BASE`):

1. devai's Slurm packages for the base distribution: `slurm-smd`,
   `slurm-smd-slurmd`, `slurm-smd-client` (Sec. 10), plus `tini` and a small
   NVML energy reader;
2. user and group `slurm` with uid/gid 401, as in the control-plane images;
3. `/usr/local/lib/devai/`: `node-entry.sh`, `prolog.sh`, `epilog.sh`,
   `gpu-idle.sh`, `is-gpu-job.sh`;
4. `ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/lib/devai/node-entry.sh"]`.

Measured sizes of the addition: +~90 MB on vllm-devai, +82 MB on Ollama.

**Entrypoint** (`node-entry.sh`), in order, each step needed in the spike:

1. cgroup nesting: move the container's processes out of its cgroup root into
   `/init` and enable the delegated controllers (without it slurmd stops with
   "Controller memory is not enabled");
2. write the image's environment to `/run/devai/node-env.sh`, which every job
   script sources (without it a REST-submitted vLLM job failed with
   `No module named 'vllm'`: a job gets only what its submitter passes);
3. `exec slurmd -D --conf-server devai-slurmctld` (configless: in round 1 a
   changed `slurm.conf` drained the node).

**Runtime flags and mounts:**

| Flag / mount | Why |
|---|---|
| `--privileged --cgroupns=private` | slurmd writes its own cgroup tree (rootless is enough; measured) |
| `--pid=host` | NVML reports host PIDs; Slurm matches them to the job's processes (measured) |
| `--device nvidia.com/gpu=all` (engine nodes) | the GPU, as the engine containers get it today |
| `--ip 10.89.0.2xx` | stable node address (measured) |
| `slurm.key` read-only | `auth/slurm` |
| model store (read-only; Ollama read-write as today) | the weights |
| engine-cache volumes (`devai-engine-cache-<backend>-*`, as `gpu-arbiter/engine_cache.go` mounts them today) | FlashInfer / SGLang JIT caches survive jobs |
| vLLM parser plugin dir (vLLM nodes) | today mounted per launch by `resolvePluginLaunch` |
| `/var/cache/devai/jobs/results` read-write | job logs and results |
| `/laya` (trainer node) | the laya store, as today |

### 3.3 Router (gpu-arbiter)

| Kept | Replaced | New |
|---|---|---|
| listeners :11434-:11438; override parsing; Anthropic / Responses normalisation; reasoning policy; tool stripping; admission (429); SSE keepalive; drain; the model-store mutation guard; launch configuration from the probe caches (`computeLaunchConfig`, entrypoint builders, recovery flags) | `containerRecreate`, `containerStop` / `Remove`, `stopOtherBackends`, `unloadOllama`, `backendVanished`, the podman / `/health` heuristics of `reconcileBackendState`, the job-runner hold (`job_runner.go`) | a Slurm client (JWT, submit / cancel / signal / state); engine job specs built from the existing launch config; one GPU state record per GPU read from Slurm; the hold registry and control API (Sec. 7.4); trainer jobs behind :11438 |

During migration a switch `DEVAI_LAUNCHER=podman|slurm` selects the launch
layer (plan, "Migration").

### 3.4 Clients and tools

| Component | Change |
|---|---|
| lab agents, Open WebUI, aiagent | none: the same router endpoints |
| lab containers | no GPU device any more (D1); aiagent's `share` GPU mode goes away |
| `scripts/bench-sync.py` | submits workload jobs instead of `make bench-*` runs; `scripts/bench/clean_slate.py` and the deadline / SIGINT / SIGKILL handling are deleted (Slurm does both) |
| probers (`make probe-*`) | become probe jobs on the backend's node: the prober drives a local engine process instead of `podman run`; the largest refactor after the router |
| `devai-jobs` (new, `devai-tools/cmd/devai-jobs`) | history, queue control, suspend / resume, interrupt (Sec. 9) |
| devai-model-status (MCP) | optional `get_job_history` tool |
| devai-logger | unchanged; follows the new `devai-*` containers by name |
| `make` | `build-slurm`, `build-nodes`, `slurm-init`; `cache-up` / `cache-down` start and stop the control plane and nodes (`cache-down` cancels jobs first) |

### 3.5 Storage

| Path | Holds | Written by | Backup |
|---|---|---|---|
| `/var/cache/devai/jobs/` (new volume, D3) | `results/<jobid>/`, `mariadb/`, `slurmctld/` | jobs, the guard, MariaDB, slurmctld | `devai-backup`: database dump, `*.json`, logs; large artifacts optional |
| `/var/cache/devai/{ollama,vllm,sglang,laya}` | model stores | unchanged | unchanged |
| `devai-engine-cache-*` volumes | FlashInfer, SGLang JIT caches | engines | not backed up (rebuildable) |
| inside each vLLM node | vLLM compile cache (`/root/.cache/vllm`, `/tmp/torchinductor_*`) | engines | **wiped by the prolog before every engine job** (Sec. 6) |
| `~/.config/devai/slurm/` (mode 0700) | `slurm.key`, `jwt_hs256.key`, `db.env` | `make slurm-init` | `devai-backup` (like the age key) |

## 4. Resource model

- **License per GPU.** `Licenses=gpu0:1`; every engine and trainer job
  requests `gpu0:1`, so Slurm never runs two at once, even on different nodes
  (measured: a second engine job waited as `PENDING (Licenses)`).
- **GRES per engine node.** Engine nodes declare `Gres=gpu:1`
  (`gres.conf: AutoDetect=nvml`); the `gpu_nvml` plugin that accounts GPU use
  loads only on a node with a GPU GRES, and Slurm sets
  `CUDA_VISIBLE_DEVICES`.
- **Partitions.** `engines` (all engine nodes, `MaxTime=INFINITE`) and `work`
  (the work node).
- **Priority, no preemption.** `priority/multifactor` with three QOS --
  `trainer` > `workload` > `interactive` -- orders the queue. Slurm's own
  preemption stays off: it cannot drain a request in flight, so the router
  decides evictions (Sec. 7.3) and Slurm enforces exclusivity and order.
- **Node sizes.** `slurmd -C` counts 8 of this host's 24 CPUs (measured), so
  nodes declare `CPUs=` and `RealMemory=` with
  `SlurmdParameters=config_overrides`.
- **cgroups.** `CgroupPlugin=cgroup/v2`, `IgnoreSystemd=yes`, no `Constrain*`:
  `cpuset` is not delegated to user sessions here
  (`DelegateControllers=cpu memory pids`; measured).
- **Several GPUs** (not in scope): one license `gpuN` per card, nodes with
  `Gres=gpu:N`, and the engine job pins its card; mapping a license to a device
  is devai's job, not Slurm's.

## 5. Job model

| Kind | Name | Partition / node | GPU | Time limit | QOS | Submitted by | Output |
|---|---|---|---|---|---|---|---|
| engine | `engine/<backend>/<model>@<ctx>[::mtp]` | `engines` / the backend's node | `gpu0:1`, `gres/gpu:1` | none (keep-warm) | `interactive`, or `workload` while a hold binds it | router only | `results/<id>/engine.log` |
| workload | `bench/<model>/<task>`, `test/...` | `work` / `devai-slurm-work` | none | per kind: bench `--time=32 --signal=B:INT@120` | `workload` | bench-sync, devai-jobs, make | `results/<id>/result.json` + artifacts |
| trainer | `train/<job>` | `engines` / `devai-laya-trainer` | `gpu0:1`, `gres/gpu:1` | `LAYA_MAX_HOLD_S` (900 s) | `trainer` | router (:11438 API) | the laya run dir, as today, + `result.json` |
| probe | `probe/<backend>/<model>` | `engines` / the backend's node | `gpu0:1`, `gres/gpu:1` | 60 min | `workload` | `make probe-*` | probe cache update + `result.json` |

**Comment.** Every job carries a JSON comment, recorded by slurmdbd
(`AccountingStoreFlags=job_comment,job_script`). Through REST it arrives intact;
`#SBATCH --comment` strips the quotes (measured).

```json
{"kind": "engine", "backend": "vllm-devai", "model": "Qwen3.8-27B-W4A16-devai-AutoRound",
 "ctx": 131072, "mtp": false, "reasoning": "auto", "hold_for": null,
 "requested_by": "router", "git_commit": "b9640bd", "host_env_id": "..."}
```

**Engine job script.** The router builds it from the launch configuration it
computes today (`computeLaunchConfig`, the entrypoint builders, recovery flags
and env, parsers, MTP). The spike ran exactly this for the router's last launch
of the AutoRound model:

```bash
#!/bin/bash
. /run/devai/node-env.sh
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 FLASHINFER_DISABLE_VERSION_CHECK=1
exec python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3.8-27B-W4A16-devai-AutoRound --host 0.0.0.0 --port 11434 \
  --tensor-parallel-size 1 --max-model-len 131072 --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.93 --enable-prefix-caching --trust-remote-code \
  --served-model-name Qwen3.8-27B-W4A16-devai-AutoRound --max-num-seqs 4 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --language-model-only --mamba-ssm-cache-dtype float16 --max-num-batched-tokens 2048
```

`exec` is the point: the engine is the job's own process, so Slurm's cgroup,
signals and accounting apply to it directly. Measured: `VLLM::EngineCore`
(21.4 GiB) and Ollama's `llama-server` (18.6 GiB) sat in the job's cgroup;
`scancel` left no GPU process 0.23-0.30 s later, with vLLM shutting down
gracefully; Slurm accounted `gres/gpumem=21794M`, `gres/gpuutil=99-100`.

**REST submission** (v0.0.44), as used in the spike:

```json
{"job": {"name": "engine/vllm-devai/Qwen3.8-27B-W4A16-devai-AutoRound@131072",
         "partition": "engines", "required_nodes": ["devai-vllm-devai"],
         "licenses": "gpu0:1", "tres_per_node": "gres/gpu:1", "qos": "interactive",
         "current_working_directory": "/tmp",
         "standard_output": "/var/cache/devai/jobs/results/%j/engine.log",
         "environment": ["PATH=/usr/sbin:/usr/bin:/sbin:/bin"],
         "comment": "{\"kind\": \"engine\", ...}", "script": "#!/bin/bash\n..."}}
```

## 6. GPU guard (prolog and epilog)

The guard is the guarantee the router lacks today. Both scripts run as root in
the node for every GPU job (a GPU GRES or the `gpu0` license) and do nothing
for other jobs.

**Prolog**, before the job starts:

1. Wait up to `PROLOG_GPU_WAIT_S` (10 s) for the card to be idle: no compute
   process and at most 512 MiB used.
2. Still busy: apply `DEVAI_GPU_HOLDER_POLICY` -- `kill` (default) or
   `refuse`. `kill` sends each holder SIGTERM, waits 5 s, sends SIGKILL, and
   logs PID, command, container (from `/proc/<pid>/cgroup`) and memory to
   `results/<jobid>/prolog.log`, then waits up to 20 s more. Measured
   2026-09-30: from the vllm-devai node, a PyTorch process holding 2.2 GiB in
   another container ignored SIGTERM (PID 1 in its container), died on SIGKILL,
   and the card was back at 2 MiB. A holder is by definition a D1 violation --
   devai application containers do not use the GPU.
3. Still busy, or policy `refuse`: record the holder in the node's reason
   (`scontrol update node=<n> reason="gpu held by pid ..."`) and exit 1. Slurm
   then drains the node and holds the job; no engine reaches the card
   (measured S7: "gpu NOT idle after 30s: 4328 MiB used; holders: 723061,
   python3, 4318 MiB", node `draining (Prolog error)`, vLLM never started).
   Processes of other users cannot be killed from the node and end here.
4. On vLLM nodes: remove `/root/.cache/vllm` and `/tmp/torchinductor_*`. The
   node container outlives its jobs, and a restart on the same node reached
   `/health` in 27-29 s instead of 95 s (measured) -- that is the compile
   cache docs/router.md says must not persist, because it under-measures
   activation memory and OOM-killed engines on 2026-09-23.
5. Record NVML's total energy counter.

**Epilog**, after the job:

1. Check only **this job's** leftovers: Slurm has already killed the job's
   cgroup, so any surviving process of it is a fault (exit non-zero -> the node
   drains). It must *not* demand a globally idle card: with two nodes on one
   GPU, the previous engine's epilog and the next engine's prolog ran in the
   same second (measured), and a global check would see the next engine and
   drain the wrong node.
2. Read NVML's energy counter again and write `results/<jobid>/gpu.json`
   (`energy_j`, both counter readings). Slurm cannot supply GPU energy on
   NVIDIA: its NVML plugin does not read energy (`gpu_p_energy_read` is a stub;
   measured), while NVML's own `nvmlDeviceGetTotalEnergyConsumption` works
   (measured through `pynvml`; `nvidia-smi` on this driver does not expose it).

## 7. Router <-> Slurm

### 7.1 Authentication

The router signs its own HS256 JWT with the shared key (claims `iat`,
`exp` = now + 300 s, `sun`) for every burst of calls; slurmrestd accepted such
a token (measured 2026-09-30). No token files and no expiry to manage. The key
lets its holder act as any Slurm user, so only the router (read-only mount)
and `devai-jobs` (host file, mode 0600) get it. Jobs run as root inside the
node, which under rootless podman is the host's devai user; there is no second
identity.

### 7.2 Calls

| Router needs | Call |
|---|---|
| start an engine / trainer job | `POST /slurm/v0.0.44/job/submit` |
| stop it | `DELETE /slurm/v0.0.44/job/{id}` |
| pause / continue a workload | `DELETE /slurm/v0.0.44/job/{id}?signal=SIGSTOP` / `SIGCONT` (measured: state `STOPPED`, output frozen, resumed on SIGCONT) |
| job state | `GET /slurm/v0.0.44/job/{id}`; `GET /slurm/v0.0.44/jobs/` filtered by name prefix `engine/`, `train/` |
| node state and drain reason | `GET /slurm/v0.0.44/node/{name}` |
| return a node to service | `POST /slurm/v0.0.44/node/{name}` with `state: RESUME` |
| license use | `GET /slurm/v0.0.44/licenses/` |
| history | `GET /slurmdb/v0.0.44/jobs/` |

REST has no suspend operation in v0.0.44 (checked in slurmrestd's own OpenAPI
spec: `suspend` appears only as a job field and in node power saving). True
suspend, which also stops the job's clock, is `scontrol suspend`, which
`devai-jobs` runs in the controller container; `SIGSTOP` over REST is the
router's lighter pause.

### 7.3 GPU state and request handling

The router keeps one record per GPU -- the holder job (id, kind, backend,
model, ctx, MTP spec, hold) -- rebuilt from Slurm on boot and refreshed on
every decision (every 5 s while a launch is pending). This replaces both
`reconcileBackendState` and the per-backend `running` / `containerLaunched`
flags that let the router lose track of the card.

A request that needs an engine (the replacement for `ensureBackendRunning`):

1. Resolve backend, model, context, MTP spec -- unchanged.
2. An engine job for exactly this is RUNNING and healthy -> proxy.
3. One is PENDING or starting -> wait, with SSE keepalive, up to
   `HEALTH_TIMEOUT_SECONDS`.
4. The GPU is held by something else:
   - a trainer job -> 503 + `Retry-After` (as the laya hold today);
   - an engine bound to an active hold (Sec. 7.4) -> 503 + `Retry-After`,
     naming the workload job;
   - an interactive engine -> drain its in-flight requests (as today), cancel
     its job, submit the new one.
5. Follow the submitted job:

| Slurm says | Router does | Client sees | Breaker |
|---|---|---|---|
| `PENDING (Licenses)` | wait: the previous GPU job is completing | keepalive | -- |
| held / node draining after a prolog failure | cancel the job; read the node's reason | 503 naming the holder | not charged |
| `RUNNING` | poll the engine's `/health` | keepalive | -- |
| leaves `RUNNING` before healthy | read the tail of `results/<id>/engine.log` (router mounts `results/` read-only) | 502 with the reason | charged |
| `COMPLETED` / `FAILED` / `CANCELLED` while serving | drop the record; the next request relaunches | 502 for requests in flight | charged if it failed |
| slurmrestd unreachable | no launch or switch; warm engines keep serving | 503 "scheduler unavailable" | not charged |

`IDLE_TIMEOUT > 0` makes the router cancel an idle engine job. The breaker
keeps today's rule (a launch is repaid by the first real response) but counts
only launches that reached `RUNNING`.

Measured timings: submit -> `RUNNING` 1.0-1.26 s (REST 0.9-2.9 s in round 1);
the 27B AutoRound engine healthy 95 s after submit (a cold start); engine
switch vLLM -> Ollama: vLLM off the card 0.23 s after cancel, the queued Ollama
job `RUNNING` at 1.22 s and answering at 2.79 s.

### 7.4 Holds

A workload must keep its engine for its whole run; an agent asking for another
model must not evict a bench mid-task. The router exposes a small control API
on each backend port:

- `POST /devai/v1/holds` `{"job_id": <slurm job>, "model": "<name>@<ctx>[::mtp]"}`
  -- the router checks the job is `RUNNING` in partition `work`, starts or keeps
  the engine for that model, records the hold, and answers when the engine is
  healthy (or 503);
- `DELETE /devai/v1/holds/{job_id}` -- releases it; the engine stays warm as an
  interactive engine;
- `GET /devai/v1/gpu` -- the holder, active holds, the queue.

A hold is active while its workload job is `RUNNING`; while the job is
`SUSPENDED` or `STOPPED` it is paused (it blocks nothing); when the job is gone
the router drops it by itself, so a killed workload cannot leave a hold behind.
A workload script takes its hold at start and releases it in an `EXIT` trap;
its inference requests are ordinary requests, so the bench harness does not
change.

### 7.5 Trainer jobs (:11438)

The fine-tuning API aiagent uses stays as it is (docs/laya-trainer.md). Behind
it, *create* submits a `train/<job>` job on `devai-laya-trainer` that runs the
trainer command the controller runs as a subprocess today; *cancel* is
`DELETE`; *list* and *get* read the run records on the laya volume as today. The
trainer's own HTTP controller is no longer a long-running service. The API does
not change, but the internals need the agreement of the laya trainer's owner
(the aiagent session).

## 8. Flows

**8.1 Cold start** (an agent asks for a model; nothing on the GPU). 1 request
reaches the router -> 3 the router submits an engine job -> 4, 5 slurmctld
grants `gpu0` and launches it on the backend's node -> the prolog finds the
card idle, wipes the compile cache, records energy -> the engine starts (8) ->
the router sees `RUNNING`, polls `/health`, keeps the client alive -> 2 the
request is proxied.

**8.2 Engine switch** (another model on the same or another backend). The
router drains the current engine's requests, cancels its job (the card is free
0.23-0.30 s later; measured), and submits the new job; if it lands before the
old job's epilog finishes it waits as `PENDING (Licenses)`, then the prolog
checks the card and the engine starts.

**8.3 Bench as workload jobs.** `bench-sync` submits one workload job per
(model, task) with `--time=32 --signal=B:INT@120`. The job takes a hold for its
model, runs the task, writes `result.json` and releases the hold. At 30 minutes
Slurm sends SIGINT, inspect writes its cancelled log, the job's trap scores the
unbroken prefix (today's `harvest_truncated.py`) and exits; KillWait and SIGKILL
remain the backstop. Two benches can no longer overlap on the GPU: the second
one's hold waits.

**8.4 Interruption, same engine** (D1). `devai-jobs interrupt W --run Q`:
`scontrol suspend W`; Q runs against the engine W is holding; `scontrol resume
W`. Measured 2.9 s in all; W made no request while suspended and finished 20/20
with no retries.

**8.5 Interruption, another engine.** Suspend W (its hold pauses) -> Q takes a
hold on its model, and the router switches engines -> Q ends and releases ->
`devai-jobs` re-asserts W's hold (`POST /devai/v1/holds`), which switches the
engine back and answers when healthy -> `scontrol resume W`. Measured 38 s with
a warm compile cache; with the prolog wiping it, expect about 95 s for the 27B.

**8.6 A GPU holder outside Slurm.** The prolog kills it and the job starts
(policy `kill`), or the node drains and the router cancels the held job and
returns 503 naming the holder (`refuse`, or a holder it cannot kill).
`devai-jobs gpu` shows the node's reason; `devai-jobs resume-node <n>` returns
it to service. The router resubmits rather than releasing a held job: Slurm
delays a requeued job (it started 130 s after release; measured), while a fresh
job started in about 1 s every time.

**8.7 Engine crash.** The job ends `FAILED`; its cgroup is gone with the
process tree; the epilog checks and records; the router drops its record and
the next request relaunches, bounded by the breaker.

**8.8 Restarts.** *Router:* rebuilds its GPU record from Slurm; running engines
keep serving meanwhile, because the data path (2) never passes through Slurm.
*slurmctld:* reloads its state from `jobs/slurmctld/`; running jobs continue
under their slurmd. *A node container:* its job dies with it (cgroup), freeing
the card; the node re-registers on its fixed IP. *Host reboot:* engine jobs are
keep-warm only and relaunch on demand; workload jobs are submitted with
`--requeue`.

## 9. History and results

**In slurmdbd**, per job: id, name, partition, node, times (submit, start,
end), state, exit code, suspended time, allocated TRES (license, GRES), usage
TRES (`gres/gpumem` and `gres/gpuutil` max and average, CPU, memory), the job
script, and the JSON comment.

**In `/var/cache/devai/jobs/results/<jobid>/`:**

| File | Written by | Content |
|---|---|---|
| `engine.log` / `job.log` | Slurm (stdout, stderr) | the process's output |
| `result.json` | the job | the envelope below |
| `gpu.json` | prolog / epilog | energy (J) and both NVML counter readings |
| `prolog.log` | prolog | GPU holders found and what was done to them |
| `artifacts/` | the job | e.g. inspect `.eval` logs |

```json
{"schema": "devai.job-result/1", "job_id": 1234, "kind": "bench",
 "subject": {"backend": "vllm-devai", "model": "Qwen3.8-27B-W4A16-devai-AutoRound", "ctx": 131072, "mtp": false},
 "task": "gpqa", "state": "COMPLETED",
 "metrics": {"n": 100, "score": 0.61, "n_timeouts": 1},
 "truncated": null, "git_commit": "b9640bd", "host_env_id": "...",
 "artifacts": ["artifacts/2026-10-01T10-00-00_gpqa-task.eval"]}
```

**`devai-jobs`** joins the two (REST for Slurm's side, files for devai's):

| Command | Does |
|---|---|
| `devai-jobs list [--since yesterday] [--kind bench] [--model ...]` | one line per job: id, kind, subject, state, elapsed, GPU peak / utilisation / energy, headline metric |
| `devai-jobs show <id>` | the full record: Slurm fields, comment, `result.json`, `gpu.json`, log tail |
| `devai-jobs queue` / `gpu` | pending and running jobs; the GPU holder, holds, drained nodes and reasons |
| `devai-jobs cancel <id>` / `cancel --pending` / `hold` / `release` / `top` | queue control (all measured in S5) |
| `devai-jobs suspend` / `resume <id>` | `scontrol suspend` / `resume` in the controller container |
| `devai-jobs interrupt <id> --run <spec>` | flows 8.4 and 8.5 |
| `devai-jobs resume-node <node>` | return a drained node to service |

History is kept indefinitely (no slurmdbd purge; the volume is local). The
bench leaderboard (`deploy/.bench-cache.json`) stays, and each task entry
records its job id.

## 10. Configuration, secrets, build

**In the repository:** `deploy/slurm/slurm.conf`, `gres.conf`,
`cgroup.conf`, `slurmdbd.conf.in` (the password is filled in at start-up);
`deploy/slurm/node/` (the node scripts); `deploy/slurm/Dockerfile.node`;
`deploy/slurm/Dockerfile.slurm-debs`.

**Generated once by `make slurm-init`, never committed:**
`~/.config/devai/slurm/slurm.key` (1024 random bytes), `jwt_hs256.key` (32
bytes), `db.env` (MariaDB passwords).

**`slurm.conf` essentials**, all exercised in the spike:
`AuthType=auth/slurm`, `CredType=cred/slurm`, `AuthAltTypes=auth/jwt`,
`SlurmctldParameters=enable_configless`, `SlurmdParameters=config_overrides`,
`ProctrackType=proctrack/cgroup`, `TaskPlugin=task/cgroup`,
`JobAcctGatherType=jobacct_gather/cgroup`,
`AccountingStorageType=accounting_storage/slurmdbd`,
`AccountingStorageTRES=gres/gpu,gres/gpumem,gres/gpuutil,license/gpu0`,
`AccountingStoreFlags=job_comment,job_script`, `GresTypes=gpu`,
`Licenses=gpu0:1`, `SelectType=select/cons_tres`, `KillWait=10`,
`ReturnToService=2`, `Prolog=` / `Epilog=` the node scripts. MPI is not used;
the PMIx plugin errors slurmd logged at start-up in the spike are harmless and
go away with the PMIx runtime library in the node image (Phase 1).

**Slurm packages.** SchedMD's official node image lacks `gpu_nvml`, and adding
the plugin file is not enough: NVML autodetection is compiled into Slurm's core
(measured). `make build-slurm` therefore builds the whole package set with
SchedMD's own `debuild` step, source `slurm-26-05-4-1` (sha256
`0e522d39324b7b7da5e8096c678c4af00500ca4c3fe2e6da7e4f8d01f7082ec7`), plus the
NVML headers, once per node distribution: Debian trixie (`libnvidia-ml-dev`
from non-free; built in the spike) and Ubuntu 24.04 (multiverse). The control
plane uses SchedMD's images unchanged (it needs no NVML).

## 11. Failure modes

| Failure | Effect | Handling |
|---|---|---|
| slurmctld down | no launch, switch or cancel; warm engines keep serving | router 503 "scheduler unavailable"; compose restarts it; state in `jobs/slurmctld/` |
| slurmrestd down | the same, for the router | as above |
| slurmdbd or MariaDB down | jobs run; history is delayed | slurmctld spools accounting until slurmdbd is back |
| GPU held outside Slurm | the prolog kills it, or refuses and drains the node | Sec. 6; 503 naming the holder |
| engine crashes | job `FAILED`, card freed with the cgroup | Sec. 8.7 |
| unkillable engine process (stuck in the driver) | job stays `COMPLETING` | `UnkillableStepTimeout` drains the node; the next prolog refuses; operator |
| node container restarts | its job dies; card freed | node re-registers on its fixed IP |
| compile cache carried into a new job | activation memory under-measured, later OOM | prolog wipes it (Sec. 6) |
| router restarts | nothing is lost | state rebuilt from Slurm |
| JWT key leaked | full control of Slurm | key only in the router (read-only) and a 0600 host file |

## 12. Security

- Nodes are privileged (in the user namespace), share the host PID namespace
  and hold the GPU: they can signal every process of the devai user. That is
  what the guard needs, and it is not a boundary against the host user or
  software outside devai (out of scope, plan).
- Nothing new is published to the LAN. The control plane is on `devai-net`
  only; `devai-jobs` reaches slurmrestd through `podman exec` or a `127.0.0.1`
  publish, following the MCP gateway's posture.
- `slurm.key` (daemon authentication) and `jwt_hs256.key` (API
  authentication) are the two secrets that matter; MariaDB is reachable only on
  `devai-net`.

## 13. From today to this

| Today | Proposed | Plan phase |
|---|---|---|
| placeholders `sleep infinity`, recreated by the router per model | permanent node containers running `slurmd`, same names | 1 |
| router recreates engine containers (`containerRecreate`) | router submits engine jobs | 2 |
| router trusts `running` / `containerLaunched` flags | router reads the GPU holder from Slurm | 2 |
| no check that the card is free | prolog: idle, kill, or drain | 1 |
| laya busy hold (`job_runner.go`) | trainer jobs + license | 3 |
| bench clean slate and 30-minute deadline in `bench-sync.py` | workload jobs, time limit, holds | 3 |
| probers start engine containers themselves | probe jobs on the backend's node | 3 |
| results in markdown / JSON, often outside the tree | slurmdbd + `jobs/results/` + `devai-jobs` | 4 |
| lab containers have the GPU | lab containers have no GPU device | 5 |

## 14. Open points

1. **The gemma image is Ubuntu 22.04**, so its node needs a third Slurm build,
   for one model (diffusiongemma). Build it, or retire the override?
2. **Suspended benches and inspect's clock.** `scontrol suspend` stops Slurm's
   clock, but inspect's per-sample `working_limit` is wall time minus waits and
   keeps running while suspended, so a sample can time out after a resume.
   Interrupt between samples, or accept it; to be measured.
3. **Requests in flight at a suspend.** A different-engine interruption drains
   the engine first; a suspended client cannot read a long streamed response,
   so the drain can wait until `DRAIN_TIMEOUT`. `devai-jobs interrupt` should
   suspend at a request boundary; to be measured.
4. **The prober refactor** (engine as a local process in a probe job) is not
   designed yet.
5. **The energy reader** for images without Python (Ollama): a small static
   binary.
6. **MariaDB sizing** (buffer pool against host RAM).
7. **The laya trainer's internals** (Sec. 7.5) need the aiagent session's
   agreement.
8. **Several GPUs** (Sec. 4) are sketched, not designed.
