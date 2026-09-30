# Slurm gatekeeper -- architecture

**Status: proposed (2026-10-01), not built.** This is the design the phases of
[the plan](plans/slurm-gatekeeper.md) implement; the plan holds the operator's
decisions (D1-D9) and the Phase 0 spike. **(measured)** marks what the spike or
the follow-up checks measured on this host; everything else is proposal. Once
built, this document is the source of truth for Slurm in devai, as
[router.md](router.md) is for the router.

This revision replaces the first one (2026-09-30), which ran one Slurm node per
backend image. The operator judged that too many moving parts and set three
rules (D7-D9): one version per backend, all backends in one Debian trixie
image, and as few images as possible, following the parallel image-reduction
proposal (only `debian:trixie` pulled from upstream; everything else built
here). That removes most of what the first revision needed: several node
containers, a cluster-wide GPU license, configless config, fixed addresses,
per-distribution Slurm builds, and a race between one node's epilog and
another node's prolog.

## 1. What changes

Today the router starts engines itself: for every model switch it recreates a
podman container, and it decides what holds the GPU from its own flags. It
never looks at the card, so an engine can start on a GPU another process still
holds -- the failure that cost the re-bench of 2026-09-28 a day.

Proposed:

- **All backends live in one image, `devai-engines`**, run as **one privileged
  container**. The same container runs Slurm -- controller, accounting
  (with MariaDB), REST API and the node daemon -- as a single-node cluster.
- **Slurm is the only thing that starts GPU work.** An engine is a Slurm job
  that `exec`s the backend's server; Slurm runs one GPU job at a time, kills a
  job's whole process tree on cancel, refuses to start on a card someone else
  holds, queues and suspends work, and records every job with its GPU use.
- **The router stays a separate, unprivileged container.** It is still the only
  entry point for inference; its launch layer becomes a Slurm client.
- **One version per backend** (D8). A model that does not run on its backend's
  one version is dropped.
- devai application containers (the lab and its agents) talk to the router and
  never use the GPU (D1).

## 2. Architecture at a glance

![Slurm gatekeeper architecture](slurm-architecture.svg)

Source: [`scripts/diagrams/slurm_architecture.py`](../scripts/diagrams/slurm_architecture.py)
(explicit coordinates; stdlib only). Re-render with
`python3 scripts/diagrams/slurm_architecture.py`.

| # | From -> to | What | How | Status |
|---|---|---|---|---|
| 1 | lab agents -> router | inference (OpenAI, Anthropic, Ollama APIs); fine-tuning jobs on :11438 | HTTP, `devai-lab-egress` | unchanged |
| 2 | router -> engine | proxied requests to `devai-engines:<port>` | HTTP, `devai-net` | measured (spike: a job's vLLM served over `devai-net`) |
| 3 | router -> slurmrestd | submit, cancel, signal, job and node state | REST v0.0.44, JWT the router signs itself | measured |
| 4 | slurmrestd -> slurmctld | the same, as Slurm RPC | inside the container, `auth/slurm` | measured |
| 5 | slurmctld <-> slurmd | launch, signal, kill; slurmd runs each job's process in its own cgroup and accounts it (the jobs drawn inside the node) | inside the container | measured |
| 6 | slurmctld -> slurmdbd -> MariaDB | accounting records | inside the container | measured |
| 7 | engine job -> GPU | CUDA; the guard and accounting read NVML | device via CDI | measured |
| 8 | workload job -> router | inference requests; hold / release | HTTP | requests measured; holds proposed (Sec. 6.3) |
| 9 | operator -> Slurm, results | queue and history (REST); `scontrol suspend` / `resume`; result files | `devai-jobs` | proposed |
| 10 | GPU guard -> stray process | SIGTERM, then SIGKILL; if still busy, drain | signals (`--pid=host`) | measured |
| 11 | host volumes -> `devai-engines` | model stores, engine caches, parser plugins | mounts | measured (model store, FlashInfer cache) |
| 12 | `devai-engines` -> `jobs/` | results, MariaDB files, controller state | files | proposed |

## 3. The `devai-engines` image

Built here on `debian:trixie-slim`, the only upstream image (D9):

| Part | Source | Where |
|---|---|---|
| Slurm 26.05.4, all daemons, with `gpu_nvml` | SchedMD source `slurm-26-05-4-1` (sha256 `0e522d39324b7b7da5e8096c678c4af00500ca4c3fe2e6da7e4f8d01f7082ec7`), SchedMD's own `debuild` step plus `libnvidia-ml-dev` (measured: builds on trixie) | Debian packages |
| MariaDB 11.8, supervisor, tini | Debian trixie packages | system |
| Ollama | our build (`make build-ollama-dist`) | `/usr/lib/ollama`, `/usr/bin/ollama` |
| vLLM 0.28.0 + HyperQwen | our build (`make build-vllm-dist`), today's `vllm-devai` | venv `/opt/vllm` |
| SGLang 0.5.16 | PyPI, hash-locked | venv `/opt/sglang` |
| laya trainer | `laya-trainer/` + its lock | venv `/opt/laya` |
| workload clients (bench, probe) | a hash-locked subset of the lab lock (inspect-ai ...) | venv `/opt/workload` |
| nvcc and headers for FlashInfer's JIT | the host toolkit, as `Dockerfile.vllm` copies it today | `/usr/local/cuda` |
| devai scripts | `deploy/engines/`: entrypoint, prolog, epilog, GPU guard | `/usr/local/lib/devai/` |

Each engine keeps its own venv because each pins its own torch. The repo's
`scripts/` are mounted read-only, so bench and probe code changes need no
rebuild.

**It replaces eight images and four planned ones:** `devai-ollama`,
`devai/vllm-devai`, `vllm-openai` 0.22.1 / 0.25.1 / gemma, `lmsysorg/sglang`,
`devai-laya-trainer` and its `nvidia/cuda` base; plus the
three SchedMD images and `mariadb` the first revision planned. The inference
stack then has two images: `devai-engines` and the router's (a static binary on
an empty base, per the image-reduction proposal).

## 4. The `devai-engines` container

| Setting | Value | Why |
|---|---|---|
| flags | `--privileged --cgroupns=private --pid=host`, GPU via CDI | cgroup control for slurmd (rootless is enough); NVML reports host PIDs; the guard must see and signal GPU holders (all measured) |
| network | `devai-net` only | the lab reaches engines only through the router |
| ports on `devai-net` | 11434 Ollama, 11435 vLLM, 11436 SGLang, 11438 laya trainer, 6820 slurmrestd | one fixed port per backend |
| mounts | model stores (Ollama and laya read-write, vLLM / SGLang read-only), engine-cache volumes, vLLM parser plugins, `/var/cache/devai/jobs`, keys (read-only), repo `scripts/` (read-only) | |
| PID 1 | `tini` -> entrypoint -> `supervisord` | |

The entrypoint does three things, each needed in the spike: move the
container's processes out of its cgroup root so controllers can be delegated
(without it slurmd stops with "Controller memory is not enabled"); write the
image environment to `/run/devai/node-env.sh`, which every job sources (a job
gets only what its submitter passes: without it a REST-submitted vLLM job
failed with `No module named 'vllm'`); start supervisord, which runs MariaDB,
slurmdbd, slurmctld, slurmrestd and slurmd.

**Slurm configuration, single node:** one node `engines` with `Gres=gpu:1`
(`AutoDetect=nvml`) and explicit `CPUs=` / `RealMemory=` (`slurmd -C` counts 8
of 24 CPUs; measured); one partition; `auth/slurm` plus `auth/jwt` for the REST
API; `proctrack/cgroup`, `task/cgroup`, `jobacct_gather/cgroup` with
`IgnoreSystemd=yes` and no `Constrain*` (`cpuset` is not delegated here);
accounting TRES `gres/gpu,gres/gpumem,gres/gpuutil`;
`AccountingStoreFlags=job_comment,job_script`; `KillWait=10`;
`Prolog` / `Epilog` = the guard. MariaDB runs with
`--innodb-snapshot-isolation=OFF` (measured: 11.8's default makes slurmdbd warn
it will die on a write conflict).

## 5. Jobs

| Kind | GPU | Runs | Time limit | Submitted by |
|---|---|---|---|---|
| engine | `--gres=gpu:1` | `exec` the backend's server on its port | none (keep-warm) | router only |
| trainer | `--gres=gpu:1` | the laya trainer on one fine-tuning job | `LAYA_MAX_HOLD_S` (900 s) | router (:11438 API) |
| probe | `--gres=gpu:1` | the engine plus the prober client | 60 min | `make probe-*` |
| workload | none | bench and test clients, from `/opt/workload` | bench: `--time=32 --signal=B:INT@120` | bench-sync, `devai-jobs`, make |

**One GPU job at a time comes from the node having one GPU:** a second GPU job
waits as `PENDING (Resources)` (measured). CPU-only workload jobs run beside
it. Because there is one node, the node stays `COMPLETING` until the previous
job's epilog has finished, so the next job's prolog always follows it (Slurm's
documented behaviour; not measured here).

Every job carries a JSON comment (kind, backend, model, context, MTP, hold,
requester, git commit), recorded by slurmdbd; submitted through REST it arrives
intact (measured).

An **engine job script** is the launch the router builds today, as a script
instead of a container spec (measured with the router's exact vLLM arguments
for Qwen3.8-27B-W4A16-devai-AutoRound):

```bash
#!/bin/bash
. /run/devai/node-env.sh
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 FLASHINFER_DISABLE_VERSION_CHECK=1
exec /opt/vllm/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3.8-27B-W4A16-devai-AutoRound --host 0.0.0.0 --port 11435 \
  --max-model-len 131072 --kv-cache-dtype fp8 --gpu-memory-utilization 0.93 ...
```

`exec` makes the engine the job's own process, so Slurm's cgroup, signals and
accounting cover it: `scancel` left no GPU process 0.23-0.30 s later, vLLM shut
down gracefully, and Slurm recorded `gres/gpumem=21794M`, `gres/gpuutil=99-100`
(measured).

**GPU guard.** The prolog of every GPU job waits up to 10 s for an idle card
(no compute process, at most 512 MiB used); if something still holds it, it
sends the holder SIGTERM, then SIGKILL after 5 s (measured: it killed a PyTorch
process in another container and the card returned to 2 MiB); if the card is
still busy, or the holder belongs to another user, it exits 1, and Slurm drains
the node and holds the job, so nothing starts on a busy card (measured). The
prolog also deletes vLLM's compile cache (`/root/.cache/vllm`,
`/tmp/torchinductor_*`): the container outlives its jobs, and a restart with
that cache reached `/health` in 27-29 s instead of 95 s (measured) -- the cache
docs/router.md says must not persist, because it under-measures activation
memory and OOM-killed engines on 2026-09-23. Prolog and epilog read NVML's
energy counter (Slurm reads none on NVIDIA; measured) into
`results/<jobid>/gpu.json`.

## 6. Router

### 6.1 What changes in it

The request path stays: listeners, override parsing, API normalisation,
reasoning policy, tool stripping, admission, SSE keepalive, drain, the
mutation guard, and the launch configuration it computes from the probe caches.
The launch layer is replaced: no more podman API calls (`containerRecreate`,
`stopOtherBackends`, `unloadOllama`, `backendVanished`, the job-runner hold);
instead a Slurm client and one GPU record, rebuilt from Slurm on boot and
refreshed on every decision. The router signs its own HS256 JWT with the shared
key (measured: slurmrestd accepts it), so there are no token files.

| Router needs | REST call |
|---|---|
| start / stop a GPU job | `POST /slurm/v0.0.44/job/submit`, `DELETE /slurm/v0.0.44/job/{id}` |
| pause / continue a workload | `DELETE .../job/{id}?signal=SIGSTOP` / `SIGCONT` (measured) |
| job state, node reason | `GET .../job/{id}`, `GET .../jobs/`, `GET .../node/engines` |
| history | `GET /slurmdb/v0.0.44/jobs/` |

REST has no suspend operation (checked in its OpenAPI spec). True suspend, which
also stops the job's clock, is `scontrol suspend`, run by `devai-jobs`.

### 6.2 A request that needs an engine

1. The right engine job is running and healthy: proxy.
2. It is pending or starting: wait, with SSE keepalive.
3. The GPU holds something else: a trainer job or an engine bound to an active
   hold gets the client a 503 with `Retry-After`; an ordinary engine is
   drained, cancelled and replaced.
4. The submitted job waits for the previous one to finish (pending, keepalive);
   is refused by the guard (the router cancels it and returns 503 naming the
   holder, without charging the launch breaker -- it resubmits rather than
   releasing, because Slurm delays a released job by about 2 minutes;
   measured); runs (the router polls `/health`); or dies before it is healthy
   (502 with the tail of its log; the breaker is charged).

Measured: submit to running 1.0-1.26 s; the 27B engine healthy 95 s after
submit on a cold start; after cancelling vLLM the next engine job ran 1.2 s
later and answered at 2.8 s.

### 6.3 Holds

A workload keeps its engine for its whole run through a small router API:
`POST /devai/v1/holds {job_id, model}` starts or keeps the engine and answers
when it is healthy; `DELETE /devai/v1/holds/{job_id}` releases it. A hold is
active while its workload job runs, paused while the job is suspended, and
dropped by the router when the job ends. The workload's inference requests are
ordinary requests.

## 7. Flows

- **Switch.** Drain the current engine, cancel its job, submit the new one; the
  guard checks the card; the engine starts.
- **Bench.** One workload job per (model, task): take a hold, run the task,
  write `result.json`, release. At 30 minutes Slurm sends SIGINT and the job
  scores the unbroken prefix, as `harvest_truncated.py` does today. Two benches
  cannot overlap: the second one's hold waits.
- **Interrupt, same engine** (D1): `scontrol suspend` the workload, run the
  quick job, `scontrol resume` -- 2.9 s in all (measured; 20/20 requests, no
  retries).
- **Interrupt, another engine:** suspend the workload, run the quick job on its
  engine, re-take the workload's hold (the engine switches back), resume --
  38 s measured with a warm compile cache; about 95 s for the 27B once the
  prolog wipes it.
- **Restarts.** Router: rebuilds its record from Slurm; engines keep serving,
  because requests never pass through Slurm. `devai-engines`: running jobs die
  with it and the card is freed; the controller's state and the history are on
  the `jobs` volume.

## 8. History

slurmdbd keeps every job: times, state, exit code, suspended time, the comment,
the job script, and GPU memory and utilisation (measured). Each job's
`/var/cache/devai/jobs/results/<jobid>/` holds its log, `result.json` (written
by the job), `gpu.json` (energy) and artifacts. `devai-jobs` (devai-tools)
joins the two: `devai-jobs list --since yesterday`, `show <id>`, `queue`,
`cancel`, `hold` / `release`, `suspend` / `resume`, `interrupt`. The bench
leaderboard stays and records each task's job id.

## 9. Configuration

In the repository: `deploy/engines/Dockerfile`, `slurm.conf`, `gres.conf`,
`cgroup.conf`, `slurmdbd.conf.in`, `supervisord.conf`, the entrypoint and guard
scripts. Generated once by `make engines-init`, never committed:
`~/.config/devai/slurm/slurm.key`, `jwt_hs256.key` and the MariaDB password
(mode 0600; backed up by `devai-backup`).

## 10. Failure modes

| Failure | Effect | Handling |
|---|---|---|
| a Slurm daemon exits | supervisord restarts it; warm engines keep serving | the router answers 503 while it cannot launch |
| MariaDB or slurmdbd down | jobs run; history is delayed | slurmctld spools the records |
| GPU held outside Slurm | killed by the guard, or node drained | 503 naming the holder |
| engine crashes | job fails; card freed with its cgroup | next request relaunches (breaker) |
| process stuck in the driver | job stays `COMPLETING` | `UnkillableStepTimeout` drains the node |
| `devai-engines` restarts | running jobs lost; card freed | router resubmits on demand |

**Security.** `devai-engines` is privileged (in the user namespace), shares
the host PID namespace and holds the GPU: it can signal every process of the
devai user, which the guard needs. That is why the router, which handles the
lab's traffic, stays out of it. Nothing new is published to the LAN; the JWT
key and `slurm.key` are the secrets that matter.

## 11. From today to this

| Today | Proposed |
|---|---|
| 5 engine containers, recreated per model by the router | 1 `devai-engines` container, permanent |
| 8 engine images (with the trainer's CUDA base), 3 distributions | 1 image, Debian trixie |
| stock vLLM 0.22.1 and vllm-devai 0.28 on two ports, plus 2 per-model images | one vLLM, 0.28 + HyperQwen, on :11435; :11437 retired |
| the router trusts its own flags about the GPU | the router reads the GPU holder from Slurm |
| no check that the card is free | the prolog guard |
| laya busy hold in the router | trainer jobs |
| bench clean slate and deadline in `bench-sync.py` | workload jobs, time limit, holds |
| probers start engine containers | probe jobs |
| results scattered | slurmdbd + `jobs/results/` + `devai-jobs` |
| lab containers have the GPU | no GPU device in the lab |

**Models:** every vLLM row is re-probed on 0.28 and every SGLang row on the
image's SGLang; a model that fails is dropped. That covers the rows probed only
on stock 0.22.1, and the two models that needed their own image today
(Nemotron-Nano-9B-v2 on 0.25.1, diffusiongemma on the gemma build).

## 12. Open points

1. **SGLang 0.5.16 from PyPI on trixie** is not tested; the upstream image
   carries extras beyond its pip dependencies (the image-reduction proposal
   flags this). Keep SGLang only if it passes the re-probe?
2. **The laya trainer without system CUDA** (its base is `nvidia/cuda` Ubuntu
   today): not tested.
3. **Python versions:** vLLM's venv uses Debian's Python 3.13; SGLang, laya and
   the workload clients may need 3.14 from uv. Not yet resolved.
4. **Which models survive** one vLLM and one SGLang version: known only after
   the re-probe.
5. **inspect's clock during a suspend:** its per-sample limit keeps running;
   suspend between samples, or accept it. To be measured.
6. **The laya trainer's internals** move from its HTTP controller to Slurm
   jobs; its API toward aiagent does not change, but the aiagent session should
   agree.
