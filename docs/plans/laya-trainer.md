# laya trainer backend

_Add a GPU training backend to the router so aiagent can fine-tune laya "System 1" students by distillation from the 27B teacher, with the GPU swapped the standard router way._

## Status

Approved -- design decisions locked by the owner on 2026-09-24. Not scheduled. One value (`LAYA_MAX_HOLD_S`) is set from the Phase 4 measurement.

## Dependencies

- (placeholder) aiagent `feat/system1-distill` (devitops-com/aiagent). That branch produces the datasets this backend trains on, calls its API, and consumes its artifacts. The dataset and artifact contracts below are the interface. aiagent's design doc is `docs/design/laya-system1-distillation.md` in that repo, and this plan implements its section 6.

## Enables / Unblocks

- aiagent distillation campaigns. aiagent labels documents with the teacher, then this backend trains a laya student. aiagent then answers the confident cases on CPU in 24-130 ms per decision, instead of making multi-second teacher calls.
- A general pattern for non-LLM GPU work behind the router. It is the first backend that holds the GPU for a job instead of a request, so later training or batch jobs can reuse the busy hold (Phase 3).
- A written-out, revision-pinned store for encoder checkpoints, which today have no place in the download rule.

## Out of scope

- **aiagent code.** Dataset building, the onnxruntime inference runtime and the cascade are implemented in aiagent, not here.
- **A laya inference service** (`laya-serve`). Inference runs inside aiagent on CPU (owner decision D3), so no service container is needed.
- **A new first-party MCP server.** The standing policy in `docs/mcp.md` is unchanged.
- **Multi-host or cluster training.** Cluster mode stays frozen.
- **int8 or fp16 export.** Dynamic int8 was measured broken: argmax agreement with fp32 was 97/133.
- **Auto-restore after a host reboot.** `devai-infra.service` is not installed on this host, which is an existing gap and not caused by this plan.

## Open questions

1. What is `LAYA_MAX_HOLD_S`, the longest the router holds the GPU for a training job? -- recommendation: 1.5x the wall-clock of the first real job, measured in Phase 4. Until then, 7200 s.
2. Should the router port for the trainer be 11438? -- recommendation: yes. It is the next free port after vllm-devai's 11437, and aiagent needs to know it only as a configured base URL.

## Context

**What laya is.** laya (github.com/NandhaKishorM/laya, Apache-2.0, v0.3.20) is an open clone of TypeSafe's closed "Jev" System-1 decision model. It is a ModernBERT/mmBERT encoder with a small decision head that answers typed questions (`choice`, `score`, `noul`) in one forward pass, with no text generation.

**Why training is needed.** Measured on aiagent's own tasks, the shipped checkpoints are barely better than always guessing the most common label (for example sentiment 6/14 against a baseline of 4/14). Confidence barely separates right from wrong: AUROC 0.64. laya's own README calls it "a fast base to specialise, not a zero-shot decision engine". So aiagent distills. The 27B teacher labels the documents, a laya student is fine-tuned on those labels, and the student answers the easy cases on CPU in front of the LLM.

**Why the GPU is the constraint.** Fine-tuning needs torch and the GPU. The teacher (`Qwen3.8-27B-MTP-devai-NVFP4`, `--gpu-memory-utilization 0.96`) fills the 24 GB card, so the two cannot run together. The owner chose to swap the teacher out rather than use a smaller co-resident teacher.

A swap costs about 2 minutes for the teacher's cold start (119-124 s typical in `devai-vllm-devai.log`). A campaign has at most 3 training rounds, so at most 6 swaps. All teacher labeling happens before training, so training never needs the teacher.

**Why a router backend.** Today no GPU job can go through the router. The HF probers, `quant-local` and `model-sync` all work around it by stopping the stack. An agent in the lab has no channel to ask devai for anything: there is no podman socket, no admin API, and new first-party MCP servers are not allowed.

Making the trainer a router backend solves both problems. A request to the trainer's port is itself the GPU request, and the router's existing `stopOtherBackends` / `containerRecreate` does the swap. The protocol is OpenAI's fine-tuning jobs API, so aiagent's client is not devai-specific.

## Approach

**The trainer.** A new image `localhost/devai-laya-trainer`, built from `deploy/Dockerfile.base` (the lab's base) with an exact torch 2.14.0 cu130 build, runs a small controller. The controller implements a subset of the OpenAI fine-tuning jobs API. It runs one job at a time: import and validate a dataset, train on the GPU, then export to ONNX, calibrate, check parity and write golden answers on the CPU.

**The router.** It gets a fifth backend entry, `laya-trainer`, plus a busy hold. While the trainer reports `busy`, requests for other backends get 503 with `Retry-After` instead of evicting it, up to `LAYA_MAX_HOLD_S`. The router adopts a busy trainer after a restart.

**Files, not HTTP bodies.** Datasets and artifacts move through a plain directory, `/var/cache/devai/laya`, which the lab mounts: read-write on `inbox/`, read-only elsewhere. The router only accepts JSON bodies of 32 MB or less with a `model` field, so files never travel through it.

**Base checkpoints** come in through `select-models.py`, as a new written-out `LAYA_STORE` with pinned revisions.

---

## Phase 1 -- laya store and base checkpoints

### Goal

`/var/cache/devai/laya` exists with its layout. The pinned base checkpoints are in it, downloaded only by the sanctioned script. The lab can see the store.

### Deliverables

```
scripts/select-models.py                    modify -- LAYA_STORE written out in the Storage layout block; laya rows download with a pinned revision + sha256
scripts/model-families.yaml                 modify -- laya family: laya-multilingual (default student), laya-english
tests/python/test_select_models_stores.py   modify -- StorageLayoutTest covers LAYA_STORE; revision pin required for laya rows
bin/devai-agent                             modify -- optional_mounts: laya store (inbox/ rw, rest ro)
Makefile                                    modify -- MODEL_CACHE_MOUNT: the same mounts for make lab-*/shell-*
CLAUDE.md                                   modify -- store list and mount-point convention: laya/ is a plain directory by owner decision
```

### Detailed steps

1. **Layout.** Create `/var/cache/devai/laya/{base,inbox,datasets,runs}` as a plain directory on the existing `vgais-cache` filesystem. The owner decided this on 2026-09-24, so no new LV is needed.
   - Amend the mount-point convention in `CLAUDE.md`, which says every top-level folder is its own LV, to name `laya/` as the deliberate exception.
2. **Store definition.** Add `LAYA_STORE = DEVAI_ROOT / "laya"` to the "Storage layout" block of `select-models.py`, written out like the other stores. `base/` holds `<name>@<rev12>/` checkpoint directories.
3. **Downloads.** laya rows download with a **pinned revision**. `pull_hf` currently pins none (`select-models.py:509`). The pinned revision is `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` (repo `convaiinnovations/laya`; subfolders `.` for English and `multilingual/` for the default student).
   - Record the sha256 of `model.safetensors` and `tokenizer.json`.
   - Retry 3 times, per the download rule.
4. **Tokenizer fix at staging.** After the download, run laya's `_fix_tokenizer_config` **once**, using the trainer image, because it rewrites `tokenizer_config.json` in place (`laya/agent.py:33-84`). Then make `base/<name>@<rev12>/` read-only.
5. **Lab mounts,** in both places the lab starts from: `bin/devai-agent` `optional_mounts` and the Makefile's `MODEL_CACHE_MOUNT`.
   - `inbox/` is read-write, so aiagent can write datasets.
   - `datasets/`, `runs/` and `base/` are read-only.
   - The mount path inside the lab is `/laya`. aiagent is configured with it as its distill directory.
6. **Ownership.** Datasets written from the lab are owned by the host user (keep-id). Check that the trainer container, also started with keep-id, can read them.

### Exit criteria

- `make model-pull NAME=laya-multilingual` fills `base/laya-multilingual@55cf4c4ebb4e/`, with sha256 recorded and the tokenizer fixed.
- `StorageLayoutTest` passes and covers `LAYA_STORE`.
- A lab started by `devai-agent` and one started by `make lab-gpu` both see `/laya/inbox` read-write and `/laya/runs` read-only.

### Phase 1 risks

| Risk | Mitigation |
| --- | --- |
| The plain directory is mistaken later for a missing LV mount. | Name the exception explicitly in `CLAUDE.md`; `StorageLayoutTest` pins the path. |
| An unpinned download silently changes the student's base. | Revision pin plus sha256 in the catalog row; the trainer refuses a base whose hash differs from the dataset manifest (exit 4). |

---

## Phase 2 -- trainer image, controller and training script

### Goal

The trainer image runs a job end to end on the GPU when started by hand, with the teacher stopped by hand. This phase needs no router changes.

### Deliverables

```
laya-trainer/                               new -- Python package laya_trainer (source dir built into the image, like gpu-arbiter/)
  laya_trainer/controller.py                new -- FastAPI app: fine-tuning jobs subset, /health, /v1/models
  laya_trainer/dataset.py                   new -- dataset contract check (re-tokenize, compare token-id hashes)
  laya_trainer/train.py                     new -- training loop (from the laya notebook's cell 8, single GPU)
  laya_trainer/export.py                    new -- ONNX export, opset 18, dynamic axes
  laya_trainer/calibrate.py                 new -- temperature fit on ONNX fp32 logits of the calib split
  laya_trainer/parity.py, golden.py         new -- torch-vs-ONNX parity; golden answers incl. input_ids
  laya_trainer/tests/                       new -- CPU unit tests with a tiny model
deploy/Dockerfile.laya-trainer              new -- FROM devai-base; torch step first; hash-locked rest; sm_120 build check
deploy/laya/requirements-trainer.txt        new -- hash-locked (laya==0.3.20, transformers==5.17.0, onnx, onnxscript, onnxruntime, fastapi, uvicorn)
Makefile                                    modify -- build-laya-trainer
```

### Detailed steps

1. **Image.**
   - Start from `deploy/Dockerfile.base` (Python 3.14.7).
   - Install `torch==2.14.0` (cu130) in its own step, as `Dockerfile.lab:125-144` does, then the hash-locked requirements with `--require-hashes`. Hash-locked Python requirements are **new for devai**; the lab has no lock.
   - The build **fails unless `sm_120` is in `torch.cuda.get_arch_list()`**.
   - laya 0.3.20 is verified on Python 3.14.7 with CPU torch 2.14.0 and transformers 5.17.0 (forward and backward). GPU training on 3.14 is verified in Phase 4.
2. **Controller API** (OpenAI fine-tuning subset):

   | Method and path | Behaviour |
   | --- | --- |
   | `POST /v1/fine_tuning/jobs` | Body `{"model": "laya-multilingual", "training_file": "<ds-id>", "hyperparameters": {...}, "suffix": "...", "metadata": {...}}`. `training_file` names a dataset directory in `inbox/` (OpenAI file ids are opaque strings, so no `/v1/files` upload is needed). |
   | `GET /v1/fine_tuning/jobs[/{id}]` | `fine_tuning.job` objects: `status` is validating_files / queued / running / succeeded / failed / cancelled; `fine_tuned_model`; `result_files`; `error`. |
   | `GET /v1/fine_tuning/jobs/{id}/events` | Progress lines (epoch, loss, phase). |
   | `POST /v1/fine_tuning/jobs/{id}/cancel` | Stops the job, keeping the last epoch checkpoint. |
   | `GET /health` | `{"status": "ok" | "busy", "job": ..., "phase": ..., "hold_until": ...}`. The router reads this in Phase 3. |
   | `GET /v1/models` | Base checkpoints plus finished runs. |

   - One job at a time; a second POST gets 409.
   - Job state is written to `runs/<job>/job.json` and `events.jsonl` on the volume, which is the source of truth. aiagent reads it from there when the trainer is not resident.
   - `/health` reports `busy` from job acceptance until packaging finishes.
3. **Import.** The controller copies `inbox/<ds-id>/` to `datasets/<ds-id>/` (immutable) and validates:
   - `SHA256SUMS` and the schema version;
   - that the base checkpoint's hash matches the manifest;
   - `max_len` and `head_max_len` **from the manifest only**;
   - every row, by re-tokenizing it with laya's own `build_sequence` and comparing against the row's `student_tokens.ids_sha256`.

   Any mismatch exits 3 before training starts.
4. **Training** (`train.py`). Start from the laya notebook's cell 8 (soft-target cross-entropy plus the policy-gradient term, using `laya.common` `build_model` / `collate_items` / `proper_reward`), with these changes:
   - single GPU; bf16 autocast with no GradScaler (the notebook's fp16 plus scaler was for T4s);
   - default student `laya-multilingual`;
   - frozen 197M-parameter vocabulary embedding plus gradient checkpointing ("lean", estimated 4-5 GiB), with the "fast" mode (about 9-10 GiB) behind a hyperparameter;
   - a per-epoch checkpoint;
   - fp32 weights saved;
   - non-finite loss exits 6.
5. **Packaging, on CPU after training.**
   - ONNX export: opset 18 with dynamic axes, traced with batch >= 2, sequence >= 300 and >= 3 markers. Upstream traces 1/16/2, and its batched export runs about 5x slower.
   - Temperatures fitted on **ONNX fp32 logits** of the `calib` split and bounded to `[0.5, 5.0]` (laya's runtime clamp), with raw and applied values recorded. `temperature_by_options` is dropped.
   - Parity, torch against ONNX Runtime on held-out: max absolute probability difference <= 1e-3, otherwise exit 7.
   - `golden.jsonl` with 40 rows, including the expected `input_ids`.
   - `manifest.json` with the fields below, `SHA256SUMS`, `NOTICE`.
   - Finally `chmod -R a-w runs/<job>/`.
6. **Artifact manifest** (contract with aiagent, `format_version: 1`):

   | field | content |
   | --- | --- |
   | `artifact_id` | hash over binds, base, config and files |
   | `binds` | copied from the dataset's opaque `producer` object, plus `dataset_manifest_sha256` |
   | `laya` | version and commit |
   | `trainer` | image digest, versions, GPU, hyperparameters, seed, timings |
   | `calibration`, `export`, `parity`, `golden` | as in step 5 |
   | `files` | sha256 and size per file |

   devai reports only loss and parity. **The ship decision is aiagent's**, made through its ONNX runtime.
7. **Exit codes:**

   | code | meaning |
   | --- | --- |
   | 0 | ok |
   | 1 | unexpected error |
   | 3 | dataset contract violation |
   | 4 | base checkpoint missing or hash mismatch |
   | 5 | GPU unavailable or out of memory |
   | 6 | training diverged |
   | 7 | export or parity failure |
   | 124 | timeout |

8. **Logs:** tee to `/var/cache/devai/logs/devai-laya-trainer.log`, because the logger discovers containers only when it starts (`deploy/logging.sh:50-58`).

### Exit criteria

- `make build-laya-trainer` succeeds, and the `sm_120` check passes.
- The CPU unit tests pass with a tiny model: contract check, export, parity, calibration bounds, job state machine.
- With the teacher stopped by hand and the container started by hand, one real dataset from aiagent trains and packages. aiagent's `distill eval` then accepts the artifact (golden answers reproduce within 1e-3).

### Phase 2 risks

| Risk | Mitigation |
| --- | --- |
| laya's API changes in a patch release (it is young and moves fast). | Pin `laya==0.3.20` by hash; aiagent's golden answers catch any drift. |
| The export's parity drifts on a fine-tuned checkpoint (only the base was verified). | Parity is a hard gate (exit 7) on the held-out split. |
| The training footprint is higher than estimated (the estimate is CPU-derived). | Lean mode by default; exit 5 on OOM; Phase 4 measures it. |

---

## Phase 3 -- router backend and busy hold

### Goal

A job request on `:11438` makes the router evict the teacher, start the trainer and hold the GPU until the job ends. After that, the next teacher request swaps back in the normal way.

### Deliverables

```
gpu-arbiter/main.go                         modify -- laya-trainer backendConfig; busy hold; boot adoption; model-less proxying; /health detail
gpu-arbiter/*_test.go                       modify -- table tests for the above (containerStateStub seam)
deploy/docker-compose.yaml                  modify -- router env: LAYA_TRAINER_PORT, LAYA_TRAINER_IMAGE, LAYA_MAX_HOLD_S
scripts/_probe_hf_common.py                 modify -- MUTEX_CONTAINERS += devai-laya-trainer, devai-vllm-devai, devai-ollama
docs/router.md, docs/aiagent.md             modify -- the backend, the hold, the aiagent settings
```

### Detailed steps

1. **Backend entry.** Add `laya-trainer` to the backend list (`main.go` around lines 1704-1770):
   - port `LAYA_TRAINER_PORT` (default 11438), container `devai-laya-trainer`, image `localhost/devai-laya-trainer:latest`;
   - `HealthPath: "/health"`;
   - `ModelsDir: /var/cache/devai/laya` mounted at `/laya` with `MountRW: true`. The controller enforces the directory discipline; alternatively, extend `backendConfig` with extra mounts;
   - an entrypoint function that starts the controller.

   Register the base model names so the allowlist accepts them.
2. **Busy hold.**
   - When a request for another backend arrives while the trainer's `/health` says `busy`, answer **503 with `Retry-After`** and an OpenAI-style error body, instead of calling `ensureBackendRunning` / `stopOtherBackends` (`main.go:3137`, `3196`, `3545`).
   - Past `LAYA_MAX_HOLD_S`, the router evicts anyway. The job is marked failed and keeps its last epoch checkpoint.
   - Today, a request for another backend drains in-flight *requests* for up to 30 s and then stops the container, which would kill a training run.
3. **Boot adoption.** At router start (`main.go:2112-2163`), a running `devai-laya-trainer` whose `/health` says `busy` is adopted and held, so a `make cache-up` during a job does not relaunch vLLM onto a GPU that is in use. The router's in-memory exclusivity is the only GPU lock in this design, so this step is required.
4. **Model-less requests** (GET job status or events, POST cancel) on the trainer port go to the resident trainer with **no lifecycle decision**, and get 503 if it is not resident.
   - Today, POST bodies must be JSON with a `model` field of 32 MB or less, and a missing model surfaces as a 503. Check this against `makeRequestHandler` before changing it.
   - Job-create bodies do carry `model`, so they already fit the existing path.
5. **`/health` detail.** Add `current_context` and `current_spec`, so a warm-up can recreate exactly what was running before the swap (`main.go:4949-4972`; recreate triggers at `3236-3250`).
6. **Guard list.** Add `devai-laya-trainer` to the probers' `MUTEX_CONTAINERS`, and fix the list's existing gap: it lacks `devai-vllm-devai` and `devai-ollama` (`_probe_hf_common.py:133`).

### Exit criteria

- Go table tests pass (`go test -race`) for:
  - busy -> 503 on the other ports;
  - hold cap -> eviction;
  - boot adoption of a busy trainer;
  - model-less proxying;
  - the normal swap after the job ends.
- Live: a job request from the lab evicts the teacher, and the job runs to the end while a concurrent teacher request gets 503 with `Retry-After`. After the job, a warm-up with the exact labeling model string brings the teacher back in its previous configuration.

### Phase 3 risks

| Risk | Mitigation |
| --- | --- |
| Other lab users are blocked for the whole job. | By design (owner accepted); the 503 carries `Retry-After`; `LAYA_MAX_HOLD_S` caps it. |
| A router restart mid-job relaunches vLLM into a used GPU. | Boot adoption (step 3), covered by a test. |
| The launch breaker (`DEVAI_MAX_FAILED_LAUNCHES=3`) trips on an OOM during a race. | A hold makes the race impossible while busy; the Phase 4 run checks it. |

---

## Phase 4 -- live verification and the hold cap

### Goal

One real aiagent campaign round runs on this host, and the measurements that set the remaining values are taken.

### Detailed steps

1. aiagent labels a small corpus for the `polarity` pilot skill and starts a job through `:11438`.
2. Measure:
   - wall-clock time for import, training, packaging and the teacher's cold start;
   - peak VRAM, lean and fast.
3. Set `LAYA_MAX_HOLD_S` to 1.5x the measured job time.
4. Confirm that GPU training works on Python 3.14.7 with the sm_120 torch build.
5. aiagent's `distill eval` gives a verdict on the artifact, and `distill install` accepts it.

### Exit criteria

- One artifact is produced and accepted by aiagent. The measured numbers are recorded in this plan and in `docs/router.md`, and `LAYA_MAX_HOLD_S` is set.

---

## Combined risk register

| Risk | Phase | Mitigation |
| --- | --- | --- |
| The contract drifts between aiagent (producer and consumer) and the trainer. | 2 | devai owns a minimal laya-generic dataset schema; aiagent's metadata travels as an opaque `producer` object; golden answers plus hash bindings are checked on the aiagent side. |
| The stack stays down after a crash or reboot. | 3 | Existing gap (no `devai-infra.service`); recovery is `make cache-up`, which is documented in `docs/router.md`. |
| HF download of base checkpoints. | 1 | Only through `select-models.py` on the host (not the locked lab); revision pin and sha256. |

## Migration / rollback story

- **Rollback:** revert the PRs. The new backend is opt-in: it is used only when something requests `:11438`, and nothing else changes when it is idle.
- **Existing installs** see one extra router port and a new store directory. Phase 3 has a behaviour change for everyone: requests get 503 while a training job holds the GPU. That only happens during a job an agent started explicitly.

## Estimated effort

| Phase | Engineering effort | Wall-clock |
| --- | --- | --- |
| Phase 1 | 1 PR, ~150 LoC plus tests | 0.5 day |
| Phase 2 | 1-2 PRs, ~800 LoC Python plus tests, Dockerfile, lock | 3-4 days |
| Phase 3 | 1 PR, ~250 LoC Go plus table tests | 2 days |
| Phase 4 | measurements only, 1 GPU window | 0.5 day |
| Total | 3-4 PRs | ~1.5 weeks |

## References

- aiagent design: `docs/design/laya-system1-distillation.md` in devitops-com/aiagent (branch `feat/system1-distill`). Section 6 is this plan; sections 8-9 hold the contracts and the ship gate. Measurements are in its appendix A and in `docs/design/laya-system1/`.
- laya: github.com/NandhaKishorM/laya, commit 23a1752 (v0.3.20). `laya/common.py` (`build_sequence`, `collate_items`, `proper_reward`), `scripts/export_onnx.py`, `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` cell 8, `docs/finetune_browser_agent.md` (fully local single-GPU precedent).
- OpenAI fine-tuning jobs API: `POST /v1/fine_tuning/jobs`, the `fine_tuning.job` object.
- devai precedents: `scripts/model-sync.py` (restore in `finally`), `scripts/prepare-checkpoint.py` (torch job image plus manifest), `scripts/_probe_hf_common.py` (GPU mutual exclusion).
