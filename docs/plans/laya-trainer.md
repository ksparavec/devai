# laya trainer backend

_Add a GPU training backend to the router so aiagent can fine-tune laya "System 1" students by distillation from the 27B teacher, with the GPU swapped the standard router way._

## Status

In Progress. Design decisions locked by the owner on 2026-09-24, revised the same day after a review of the plan against the code (see "Owner answers, second round"). **Phases 1-3 are implemented** (merged in PR #21) and verified on CPU (see "Implementation notes"). Phase 4 has the owner's go-ahead (2026-09-25): GPU training, the live swap and the 503 hold are verified on the synthetic reference dataset (`make laya-check`); the real aiagent campaign and the `LAYA_MAX_HOLD_S` measurement are still open (see Phase 4, "Progress").

## Dependencies

- aiagent `feat/system1-distill` (devitops-com/aiagent PR #15). That branch produces the datasets this backend trains on, calls its API, and consumes its artifacts. The dataset and artifact contracts below are the interface. aiagent's design doc is `docs/design/laya-system1-distillation.md` in that repo, and this plan implements its section 6.
  - On 2026-09-24 that branch existed only in the owner's local clone; it was pushed as PR #15 on 2026-09-25 with the contracts agreed with devai (see "Implementation notes"). The code form of the contract is its `src/aiagent/system1/contract.py` and `src/aiagent/distill/dataset.py`.

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
- **laya rows in `deploy/models.yaml`.** Every reader of that catalog (picker, `model-fit`, `model-sync`, bench discovery, the model-status MCP server) assumes LLM rows, and `generate-catalog.py` refuses a family without an LLM `arch_ref`. laya rows live in their own file (owner answer, second round).

## Open questions

1. What is `LAYA_MAX_HOLD_S`, the longest the router holds the GPU for a training job? -- recommendation: 1.5x the wall-clock of the first real job, measured in Phase 4. Until then, 7200 s.

Resolved: the router port is 11438 (the next free port after vllm-devai's 11437; nothing in the repo uses it).

## Owner answers, second round (2026-09-24)

A review of the first version against the code found five problems. The owner's answers:

| # | Problem found | Answer |
| --- | --- | --- |
| 1 | The router tracks a model per backend. A job for `laya-english` while the trainer was launched for `laya-multilingual` is a model change, so the router would recreate the trainer -- killing a running job -- before the controller could answer 409. | The trainer is **model-agnostic** in the router: one container serves every base, and a model change never recreates it. The allowlist of base names comes from the **laya catalog rows**. |
| 2 | `make cache-down` does not know the trainer container, so a job survives it and keeps the GPU. | The trainer gets a **`sleep infinity` placeholder** in compose like the four HF backends, and joins `CACHE_BACKEND_SERVICES` and the `cache-down` removal list. `cache-down` therefore kills a running job, which keeps its last epoch checkpoint. The placeholder also makes the logger follow the container, so the trainer needs no log tee. |
| 3 | laya rows in `model-families.yaml` / `models.yaml` break `generate-catalog.py` and every LLM reader of the catalog; `pull_hf` supports neither a revision nor a subfolder. | A **hand-written, pinned `deploy/laya-models.yaml`**, read by `select-models.py` (download) and by the router (allowlist). `models.yaml` is untouched. |
| 4 | Both a router setting and a trainer `hold_until` claimed to own the hold; after a router restart the router could not know when the job started. | The trainer reports **`started_at`**; the router computes the deadline itself from that plus `LAYA_MAX_HOLD_S`. A router restart cannot reset the clock. The hold also covers dataset import and CPU packaging; accepted for the first version. |
| 5 | OpenAI's cancel call has no body, and the router answers an empty-body POST with 400. | The router accepts an **empty-body POST on the trainer port only**, so a stock OpenAI client can cancel. |

Also decided: the laya rows live in their own file rather than in `models.yaml` (question 1 and 3 together), implementation lands as one commit per phase on `feat/laya-trainer` with no push until the owner says so, and this session covers Phases 1-3 without GPU work.

## Implementation notes (2026-09-24)

What was built differs from the phase text below in these places; everything else is as written. The reference for what exists is [docs/laya-trainer.md](../laya-trainer.md).

- **Contracts, agreed with the aiagent session.** The aiagent side (its implementation spec, sections 3 and 11) pinned details this plan left open, and devai follows them: a dataset id is `"ds-"` + sha256(manifest.json)[:12] and is checked; the manifest carries a `files` table; pool rows have no `gold`/`teacher`; no nulls in a row outside `questions`; probability keys in option order; the label is the first argmax; no synthetic rows in calib/held-out; disjoint groups; no truncated states. On the artifact side `manifest.files` is the artifact (never `checkpoint/`), `SHA256SUMS` adds `checkpoint/`, `golden.jsonl` uses aiagent's row shape with answers from laya's own `Agent` on CPU in fp32 (not from ONNX), the tokenizer is copied byte for byte, and the export is `torch.onnx.export(dynamo=True)`. The lab mount is `/laya` and the API base `http://devai-router:11438/v1`, both aiagent's defaults.
- **Phase 1.** The staging tokenizer fix was dropped (a verified no-op at the pinned revision; every file is sha256-pinned instead). `laya-trainer/laya_trainer/catalog.py` is shared by `select-models.py` and the trainer.
- **Phase 2.** More modules than listed (`contract.py`, `jobs.py`, `package.py`, `fixture.py`); the lock lives in `laya-trainer/requirements.lock` and includes torch; the controller is `http.server`; `batch_size` has OpenAI's meaning (examples per update) and devai adds `micro_batch_size` and `memory_mode`. An independent review found three real bugs, all fixed with tests: a checkpoint tokenizer directory that inherited the sealed base's 0555 mode (a second epoch failed for any non-root user), a stop-vs-finish race that could record a finished job twice, and an OOM at model placement reported as exit 1. The suite now also runs as uid 1000.
- **Phase 3.** `stopOtherBackends` returns an error, and the hold is checked there -- after draining the job runner, so a job submission still in flight cannot be killed by a concurrent request for another backend. It therefore covers every eviction path, including Ollama's model-less requests, which call `stopOtherBackends` too. The idle sweep skips a busy runner. The router review (router-policy-reviewer) found a CRITICAL flaw, fixed with a regression test that fails without the fix: a request on the trainer's own port ran the engine liveness probe, which could declare a busy, silent trainer dead, so the next switch launched an engine onto its GPU. A job runner is now judged dead by podman state only; a silent `/health` (timeout or error, 3 attempts) with a running container holds, bounded by the cap, while a refused connection (the placeholder) holds nothing; boot adoption keeps a silent busy trainer; hold verdicts are cached for 30 s; a streaming refusal carries the hold in-band. A re-review then caught that the first fix counted a DNS failure as "nothing listening" (one resolver hiccup would evict a job), that a libpod 404 read as "running", and that a placeholder put back under a running trainer made port 11438 return 502; all three fixed with tests. A second re-review approved (no CRITICAL/HIGH left); its remaining optional LOW was fixed too: `containerState` now reads only a libpod 404 as gone, and any other podman error as unknown. `docs/router.md` also gained the missing 11437 row.
- **Verified, CPU only:** the base download and lab mounts; 77 trainer tests (root and uid 1000); 35 router job-runner tests plus the full `make test-router`; 1153 `make test-python` tests; and a whole job on the real `laya-multilingual` base through the controller in the image at `max_len` 1024/256 (2 epochs: loss 1.84 -> 0.82, parity 1.3e-6, golden answers reproduced by onnxruntime alone within 4.9e-5). The router image was not rebuilt and the running stack was not touched.

## Context

**What laya is.** laya (github.com/NandhaKishorM/laya, Apache-2.0, v0.3.20) is an open clone of TypeSafe's closed "Jev" System-1 decision model. It is a ModernBERT/mmBERT encoder with a small decision head that answers typed questions (`choice`, `score`, `noul`) in one forward pass, with no text generation.

**Why training is needed.** Measured on aiagent's own tasks, the shipped checkpoints are barely better than always guessing the most common label (for example sentiment 6/14 against a baseline of 4/14). Confidence barely separates right from wrong: AUROC 0.64. laya's own README calls it "a fast base to specialise, not a zero-shot decision engine". So aiagent distills. The 27B teacher labels the documents, a laya student is fine-tuned on those labels, and the student answers the easy cases on CPU in front of the LLM.

**Why the GPU is the constraint.** Fine-tuning needs torch and the GPU. The teacher (`Qwen3.8-27B-MTP-devai-NVFP4`, `--gpu-memory-utilization 0.96`) fills the 24 GB card, so the two cannot run together. The owner chose to swap the teacher out rather than use a smaller co-resident teacher.

A swap costs about 2 minutes for the teacher's cold start (119-124 s typical in `devai-vllm-devai.log`). A campaign has at most 3 training rounds, so at most 6 swaps. All teacher labeling happens before training, so training never needs the teacher.

**Why a router backend.** Today no GPU job can go through the router. The HF probers, `quant-local` and `model-sync` all work around it by stopping the stack. An agent in the lab has no channel to ask devai for anything: there is no podman socket, no admin API, and new first-party MCP servers are not allowed.

Making the trainer a router backend solves both problems. A request to the trainer's port is itself the GPU request, and the router's existing `stopOtherBackends` / `containerRecreate` does the swap. The protocol is OpenAI's fine-tuning jobs API, so aiagent's client is not devai-specific.

**What the router already does** (read from the code on 2026-09-24):

- A request with no `model` on a running backend is proxied without a lifecycle decision (`needRecreate` stays false, `gpu-arbiter/main.go:3252`); on a backend that is not running it gets 503 "model name required" (`main.go:3264`). So status reads on a resident trainer already work, and a status read can never launch it.
- `reconcileBackendState` (`main.go:2142`) already adopts any serving non-Ollama backend at router start, with the model unknown. The busy hold (Phase 3) builds on it.
- A POST whose body is empty gets 400, because the body is parsed as JSON (`main.go:3601`). Phase 3 changes this for the trainer port only.
- `checkModelWeights` (`main.go:3067`) looks for `<ModelsDir>/<model>`, which does not match the laya store layout; Phase 3 exempts the model-agnostic trainer (the trainer checks its base itself, exit 4).

## Approach

**The trainer.** A new image `localhost/devai-laya-trainer`, built from the lab's GPU base image (`devai-base-gpu`, from `deploy/Dockerfile.base`) with a hash-locked `torch==2.14.0` (PyPI's cu130 build, the same build the lab image has: verified `2.14.0+cu130`, arch list includes `sm_120`). It runs a small controller that implements a subset of the OpenAI fine-tuning jobs API. It runs one job at a time, in a subprocess: import and validate a dataset, train on the GPU, then export to ONNX, calibrate, check parity and write golden answers on the CPU.

**The router.** It gets a fifth backend entry, `laya-trainer`, marked as a job runner: model-agnostic (the base model is a job setting, not a launch setting), allowlisted from the laya catalog, and holding the GPU while busy. While the trainer's `/health` says `busy`, a request that would need the GPU for another backend gets 503 with `Retry-After` instead of evicting it, until `started_at + LAYA_MAX_HOLD_S`. Because the hold is read from the trainer's own `/health`, a restarted router honours it without any extra state.

**Files, not HTTP bodies.** Datasets and artifacts move through a plain directory, `/var/cache/devai/laya`, which the lab mounts at `/laya`: read-write on `inbox/`, read-only elsewhere. The router only accepts JSON bodies of 32 MB or less, so files never travel through it.

**Base checkpoints** come in through `select-models.py`, from the hand-written catalog `deploy/laya-models.yaml`, into a new written-out `LAYA_STORE` with pinned revisions and a pinned sha256 for every file.

---

## Phase 1 -- laya store, catalog and base checkpoints

### Goal

`/var/cache/devai/laya` exists with its layout. The pinned base checkpoints are in it, downloaded only by the sanctioned script and verified file by file. The lab can see the store.

### Deliverables

```
deploy/laya-models.yaml                     new -- hand-written laya catalog: repo, revision, subfolder, sha256 + size per file
scripts/select-models.py                    modify -- LAYA_STORE written out in the Storage layout block; `--name <laya row>` pulls from the laya catalog
tests/python/test_select_models_stores.py   modify -- StorageLayoutTest covers LAYA_STORE
tests/python/test_laya_store.py             new -- catalog validation; pull with a fake hf runner (verify, move, read-only, skip, mismatch refusal)
bin/devai-agent                             modify -- optional_mounts: /laya (ro) and /laya/inbox (rw)
Makefile                                    modify -- MODEL_CACHE_MOUNT: the same mounts for make lab-*/shell-*
CLAUDE.md                                   modify -- store list and mount-point convention: laya/ is a plain directory by owner decision
```

### Detailed steps

1. **Layout.** `/var/cache/devai/laya/{base,inbox,datasets,runs}` as a plain directory on the existing `vgais-cache` filesystem. The owner decided this on 2026-09-24, so no new LV is needed. The directory already exists on this host (empty); the laya pull creates the four subdirectories idempotently.
   - Amend the mount-point convention in `CLAUDE.md`, which says every top-level folder is its own LV, to name `laya/` as the deliberate exception.
2. **Catalog.** `deploy/laya-models.yaml`, `schema_version: 1`, one row per checkpoint:

   | field | content |
   | --- | --- |
   | `name` | `laya-multilingual` (the default student) or `laya-english` |
   | `repo`, `revision` | `convaiinnovations/laya` at `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` (a full 40-hex commit; a branch or tag is refused) |
   | `subfolder` | `multilingual` or `""` (the English checkpoint sits at the repo root) |
   | `files` | every file of the checkpoint (`model.safetensors`, `rl_agent_config.json`, `encoder/config.json`, `tokenizer/tokenizer.json`, `tokenizer/tokenizer_config.json`) with `sha256` and `size` |
   | `default`, `license`, `encoder` | informational; exactly one row is the default |

   The sha256 values were checked against upstream on 2026-09-24: each LFS file's hash equals its LFS oid, and each small file's git blob id equals the one in the repo tree at the pinned revision.
3. **Store definition.** `LAYA_STORE = DEVAI_ROOT / "laya"` in the "Storage layout" block of `select-models.py`, written out like the other stores, with `LAYA_BASE = LAYA_STORE / "base"`. A checkpoint lives in `base/<name>@<rev12>/` with the subfolder stripped.
4. **Downloads.** `make model-pull NAME=laya-multilingual` reaches `select-models.py --name`, which finds the row in the laya catalog when it is not in `models.yaml`:
   - `hf download <repo> --revision <rev> --include <subfolder>/<file> ... --local-dir <LAYA_STORE>/.staging/<name>@<rev12>`, through the existing `run_download` (3 attempts);
   - every file is checked for size and sha256, and a mismatch deletes the staging directory and fails;
   - only the listed files are moved into `base/<name>@<rev12>/` (so the hf CLI's own `.cache/` never lands there), then files are made 0444 and directories 0555;
   - a checkpoint already present and verified is skipped.
5. **No tokenizer fix.** The first version ran laya's `_fix_tokenizer_config` at staging. At the pinned revision it is a no-op: both upstream `tokenizer_config.json` files already have the shape it produces (verified by running it on copies: byte-identical before and after). The per-file sha256 pin is what keeps the base unchanged; the trainer re-verifies it at job start (exit 4). A future revision that needs the fix gets it in the pull step then.
6. **Lab mounts,** in both places the lab starts from: `bin/devai-agent` `optional_mounts` and the Makefile's `MODEL_CACHE_MOUNT`.
   - `/var/cache/devai/laya` -> `/laya`, read-only.
   - `/var/cache/devai/laya/inbox` -> `/laya/inbox`, read-write, so aiagent can write datasets.
   - Both only when the directories exist, like the other stores. aiagent is configured with `/laya` as its distill directory.
7. **Ownership.** Rootless podman: the lab runs as `devai` with `--userns=keep-id` (host uid), and a container started by the router runs as container-root, which is also the host uid. So files either side writes are owned by the host user and readable by the other, without keep-id on the trainer. Checked on a CPU-only container in Phase 2.

### Exit criteria

- `make model-pull NAME=laya-multilingual` fills `base/laya-multilingual@55cf4c4ebb4e/` with every file verified, read-only.
- `StorageLayoutTest` covers `LAYA_STORE`; `test_laya_store.py` passes.
- A lab started by `devai-agent` and one started by `make lab-gpu` both see `/laya/inbox` read-write and `/laya/runs` read-only.

### Phase 1 risks

| Risk | Mitigation |
| --- | --- |
| The plain directory is mistaken later for a missing LV mount. | Name the exception explicitly in `CLAUDE.md`; `StorageLayoutTest` pins the path. |
| An unpinned download silently changes the student's base. | Revision pin plus a sha256 per file in the catalog; the trainer refuses a base whose hash differs from the dataset manifest (exit 4). |

---

## Phase 2 -- trainer image, controller and training script

### Goal

The trainer image runs a job end to end when started by hand. CPU unit tests with a tiny model cover everything but the GPU. This phase needs no router changes.

### Deliverables

```
laya-trainer/                               new -- source dir built into the image, like gpu-arbiter/
  laya_trainer/controller.py                new -- HTTP API: fine-tuning jobs subset, /health, /v1/models
  laya_trainer/jobs.py                      new -- job state on the volume (job.json, events.jsonl), startup reconciliation
  laya_trainer/catalog.py                   new -- laya catalog + base checkpoint verification
  laya_trainer/dataset.py                   new -- dataset import and contract check (re-tokenize, compare token-id hashes)
  laya_trainer/run_job.py                   new -- the job subprocess: import -> train -> package, exit codes
  laya_trainer/train.py                     new -- training loop (from the laya notebook's cell 8, single GPU)
  laya_trainer/export.py                    new -- ONNX export, opset 18, dynamic axes
  laya_trainer/calibrate.py                 new -- temperature fit on ONNX fp32 logits of the calib split
  laya_trainer/parity.py, golden.py         new -- torch-vs-ONNX parity; golden answers incl. input_ids
  laya_trainer/fixture.py                   new -- tiny laya checkpoint (small ModernBERT, small tokenizer) for tests and for aiagent
  laya_trainer/tests/                       new -- unittest suite, run inside the image
  requirements.in, requirements.lock        new -- hash-locked (torch==2.14.0, laya==0.3.20, transformers==5.17.0, onnx, onnxscript, onnxruntime)
deploy/Dockerfile.laya-trainer              new -- FROM devai-base-gpu; hash-locked install; sm_120 build check; catalog baked in
Makefile                                    modify -- build-laya-trainer, test-laya-trainer
```

### Detailed steps

1. **Image.**
   - `FROM devai-base-gpu` (Python 3.14.7, the lab's GPU base).
   - `uv pip install --system --require-hashes -r requirements.lock`. The lock includes `torch==2.14.0` from PyPI (the cu130 build the lab already runs) and its CUDA libraries. Hash-locked Python requirements are **new for devai**; the lab has no lock.
   - The build **fails unless `sm_120` is in `torch.cuda.get_arch_list()`**.
   - `deploy/laya-models.yaml` is copied in; `HF_HUB_OFFLINE=1`, so nothing is downloaded at run time.
   - laya 0.3.20 is verified on Python 3.14.7 with CPU torch 2.14.0 and transformers 5.17.0 (forward and backward). GPU training on 3.14 is verified in Phase 4.
2. **Controller API** (OpenAI fine-tuning subset), on port 11434 inside the container:

   | Method and path | Behaviour |
   | --- | --- |
   | `POST /v1/fine_tuning/jobs` | Body `{"model": "laya-multilingual", "training_file": "<ds-id>", "hyperparameters": {...}, "suffix": "...", "metadata": {...}}`. `training_file` names a dataset directory in `inbox/` (OpenAI file ids are opaque strings, so no `/v1/files` upload is needed). Returns the `fine_tuning.job` object. |
   | `GET /v1/fine_tuning/jobs[/{id}]` | `fine_tuning.job` objects: `status` is validating_files / queued / running / succeeded / failed / cancelled; `fine_tuned_model`; `result_files`; `error`. |
   | `GET /v1/fine_tuning/jobs/{id}/events` | `fine_tuning.job.event` objects (phase changes, epoch, loss). |
   | `POST /v1/fine_tuning/jobs/{id}/cancel` | Stops the job (body optional). Epoch checkpoints already written stay in `runs/<job>/checkpoint/`. |
   | `GET /health` | `{"status": "ok" | "busy", "job": ..., "phase": ..., "started_at": <unix seconds>}`. The router reads this in Phase 3. |
   | `GET /v1/models` | Base checkpoints plus finished runs. |

   - Built on the standard library's `http.server` rather than FastAPI: six small endpoints need no framework, and it keeps three packages and their transitive dependencies out of the hash lock.
   - One job at a time; a second POST gets 409.
   - The job runs as a **subprocess** (`python -m laya_trainer.run_job`), so a crash or OOM cannot take the controller down and GPU memory is released when it exits. Its exit code becomes the job's error code.
   - Job state is written to `runs/<job>/job.json` and `events.jsonl` on the volume, which is the source of truth. aiagent reads it from there when the trainer is not resident.
   - At startup the controller marks any job left non-terminal as failed ("trainer stopped during the job"): that is what a router eviction at the hold cap, a `cache-down` or a crash leaves behind. On SIGTERM it stops the job subprocess and writes the same status itself.
   - `/health` reports `busy` from job acceptance until packaging finishes.
3. **Import.** The controller copies `inbox/<ds-id>/` to `datasets/<ds-id>/` (immutable afterwards; an existing copy with different `SHA256SUMS` is a contract violation) and validates:
   - `SHA256SUMS` and the `schema_version`;
   - that `base_checkpoint` (name, revision, `weights_sha256`, `tokenizer_sha256`) matches the catalog row and the files on disk (exit 4 otherwise);
   - `max_len` and `head_max_len` **from the manifest only**;
   - every row: split, question shape (laya's own question check), probabilities (keys match the question type, sum to 1 +- 1e-6, `label` is the argmax), and a re-tokenization with laya's own `build_sequence` whose token-id hash must equal the row's `student_tokens.<question>.ids_sha256` (canonical hash `sha256(json.dumps(ids, separators=(",", ":")))`).

   Any mismatch exits 3 before training starts.
4. **Training** (`train.py`). Start from the laya notebook's cell 8 (soft-target cross-entropy plus the policy-gradient term, using `laya.common` `build_model` / `proper_reward`), with these changes:
   - single GPU; bf16 autocast with no GradScaler (the notebook's fp16 plus scaler was for T4s);
   - default student `laya-multilingual`;
   - hyperparameters `n_epochs` (4), `batch_size` (8), `learning_rate_multiplier` (1.0, scaling the notebook's 2.5e-5 encoder / 1e-4 head rates), plus devai's `memory_mode` and `seed`; OpenAI's `"auto"` means the default;
   - `memory_mode: lean` (default): frozen vocabulary embedding plus gradient checkpointing, estimated 4-5 GiB; `fast`: everything trainable, no checkpointing, about 9-10 GiB;
   - a per-epoch checkpoint, fp32 weights;
   - non-finite loss exits 6; CUDA unavailable or out of memory exits 5.
5. **Packaging, on CPU after training.**
   - ONNX export: opset 18 with dynamic axes, traced with batch >= 2, sequence >= 300 and >= 3 markers. Upstream traces 1/16/2, and its batched export runs about 5x slower.
   - Temperatures fitted per question type on **ONNX fp32 logits** of the `calib` split and bounded to `[0.5, 5.0]` (laya's runtime clamp), with raw and applied values recorded. The artifact's `rl_agent_config.json` carries the applied values; `temperature_by_options` is dropped.
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
   | `metrics` | torch-side loss only, for information |
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

8. **Logs:** stdout and stderr only. The compose placeholder (Phase 3) makes the logger follow `devai-laya-trainer` like the other router-recreated backends, so no tee into the logs directory is needed.
9. **Tests.** `make test-laya-trainer` runs the unittest suite inside the image, on CPU, with the tiny fixture checkpoint: catalog and base verification, dataset contract, the job state machine and controller API, training steps, export, parity, calibration bounds, golden answers.

### Exit criteria

- `make build-laya-trainer` succeeds, and the `sm_120` check passes.
- `make test-laya-trainer` passes.
- Phase 4 (GPU): with the teacher stopped and the container started by hand, one real dataset from aiagent trains and packages, and aiagent's `distill eval` accepts the artifact (golden answers reproduce within 1e-3).

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
gpu-arbiter/laya_trainer.go                 new -- backend entry, laya catalog allowlist, busy hold
gpu-arbiter/main.go                         modify -- JobRunner lifecycle (model-agnostic), hold check before every eviction, empty-body POST, /health detail
gpu-arbiter/laya_trainer_test.go            new -- table tests (httptest trainer + containerStateStub)
deploy/docker-compose.yaml                  modify -- laya-trainer placeholder service; router env + laya catalog mount
Makefile                                    modify -- CACHE_SERVICES, CACHE_BACKEND_SERVICES, cache-down removal list; cache-up skips the placeholder while the image is not built
deploy/logging.sh                           modify -- devai-laya-trainer in the fallback target list
scripts/_probe_hf_common.py                 modify -- MUTEX_CONTAINERS += devai-laya-trainer, devai-vllm-devai, devai-ollama
docs/router.md, docs/aiagent.md, CLAUDE.md  modify -- the backend, the hold, the aiagent settings
```

### Detailed steps

1. **Backend entry.** Add `laya-trainer` to the backend list:
   - port `LAYA_TRAINER_PORT` (default 11438), container `devai-laya-trainer`, image `LAYA_TRAINER_IMAGE` (default `localhost/devai-laya-trainer:latest`), URL `http://laya-trainer:11434`;
   - `HealthPath: "/health"`;
   - `ModelsDir: /var/cache/devai/laya` mounted at `/laya` with `MountRW: true`;
   - an entrypoint that starts the controller;
   - marked as a **job runner** (`JobRunner: true` on `backendConfig`), which the steps below key on.
2. **Allowlist from the laya catalog.** The router reads `deploy/laya-models.yaml` (mounted at `/etc/devai/laya-models.yaml`) and registers its row names as the trainer's model names. `/v1/models` on the trainer port lists them.
3. **Model-agnostic lifecycle.** For a job runner, only "is it running" decides a launch: a different base model is never a model change, and the trainer is launched with no model. The weights-on-disk check is skipped (the trainer verifies its base, exit 4), and the launch circuit breaker counts per backend rather than per model.
4. **Busy hold.**
   - Immediately before any eviction (`stopOtherBackends`, from both the HF and the Ollama launch paths) and in the idle sweep, the router asks every *other* running job runner for its `/health`.
   - If it says `busy` and `now < started_at + LAYA_MAX_HOLD_S`, the request gets **503 with `Retry-After: 30`** and an OpenAI-style error body naming the job and phase, instead of an eviction.
   - Past the deadline the router logs it and evicts anyway; the controller marks the job failed (on SIGTERM, or at its next start).
   - If `/health` does not answer while podman reports the container running, the runner holds -- as its last busy answer, or from the moment it went silent -- until the deadline (a controller busy packaging must not lose the GPU to a slow reply); a refused connection (nothing listening) or a container podman reports gone holds nothing.
   - Today, a request for another backend drains in-flight *requests* for up to 30 s and then stops the container, which would kill a training run.
5. **Boot adoption.** At router start, `reconcileBackendState` already adopts a serving `devai-laya-trainer`. Because the hold is read from the trainer's `/health` (step 4), a busy trainer adopted this way is held with its original deadline, so a `make cache-up` during a job does not relaunch vLLM onto a GPU that is in use. A test covers it.
6. **Model-less requests** (GET job status or events, POST cancel) on the trainer port go to the resident trainer with no lifecycle decision, as they already do; when the trainer is not resident they get 503 naming `/laya/runs/<job>/job.json`, and they never launch it. An empty-body POST is accepted on job-runner ports only.
7. **`/health` detail.** Add `current_context` and `current_spec` per backend, so a warm-up can recreate exactly what was running before the swap.
8. **Compose and Makefile.** A `laya-trainer` placeholder service (`sleep infinity`, `pull_policy: never`); the service joins `CACHE_SERVICES` and `CACHE_BACKEND_SERVICES`, and `devai-laya-trainer` joins the `cache-down` removal list. `cache-up` skips the placeholder, with a note, while the trainer image is not built, so hosts without it keep working.
9. **Guard list.** Add `devai-laya-trainer` to the probers' `MUTEX_CONTAINERS`, and fix the list's existing gap: it lacks `devai-vllm-devai` and `devai-ollama` (`_probe_hf_common.py:133`).

### Exit criteria

- Go table tests pass (`go test -race`) for:
  - busy -> 503 with `Retry-After` on the other ports, from both the HF and the Ollama launch paths;
  - hold cap -> eviction;
  - boot adoption of a busy trainer;
  - model-agnostic launch (a second base name does not recreate), model-less proxying, empty-body POST;
  - the normal swap after the job ends.
- Phase 4 (GPU): a job request from the lab evicts the teacher, and the job runs to the end while a concurrent teacher request gets 503 with `Retry-After`. After the job, a warm-up with the exact labeling model string brings the teacher back in its previous configuration.

### Phase 3 risks

| Risk | Mitigation |
| --- | --- |
| Other lab users are blocked for the whole job. | By design (owner accepted); the 503 carries `Retry-After`; `LAYA_MAX_HOLD_S` caps it. |
| A router restart mid-job relaunches vLLM into a used GPU. | Boot adoption plus the `/health`-read hold (steps 4-5), covered by a test. |
| A hung controller holds the GPU. | The deadline is computed by the router from `started_at`; past it the router evicts regardless of what `/health` says. |
| The launch breaker (`DEVAI_MAX_FAILED_LAUNCHES=3`) trips on an OOM during a race. | A hold makes the race impossible while busy; the Phase 4 run checks it. |

---

## Phase 4 -- live verification and the hold cap

### Goal

One real aiagent campaign round runs on this host, and the measurements that set the remaining values are taken. Needs the owner's go-ahead: it evicts the teacher.

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

### Progress (2026-09-25)

- Owner go-ahead given. Before the real campaign, a GPU smoke job ran on aiagent's synthetic 80-row interop dataset (`ds-a6c9c8248242`): the live swap, the 503 hold and GPU training on Python 3.14.7 with the sm_120 torch build (step 4) are verified, and the lab's aiagent 0.5.0 accepted the artifact. Numbers: docs/laya-trainer.md, "Reference check". The owner kept that dataset as the reference for teacher/trainer coordination; `make laya-check` reruns it.
- The real campaign ran the same day from aiagent's runbook (aiagent 0.5.0 in the lab image; corpus of 2,400 openly licensed multilingual documents; teacher `Qwen3.8-27B-MTP-devai-NVFP4::mtp::nothink@118784`):
  - Label: 6,177 teacher calls in 23.6 min (k=3, concurrency 4), 0 parse failures; splits 1,448 / 277 / 334 / 341.
  - Train, three rounds (repair adds the pool rows the student is least sure of): 1,448 / 1,704 / 1,789 train rows x 4 epochs, lean. **Hold (started -> finished) 123 / 136 / 140 s**; of that, training 39 / 45 / 47 s, and parity (334 held-out rows on the CPU) 45 / 50 / 47 s. The job request with the swap: about 3 s; the teacher's cold start afterwards: 2 min 5 s each time, back in exactly the configuration it had.
  - Peak VRAM: lean 3.2 GiB (torch) / 3.7 GiB (nvidia-smi); fast (one extra job, not evaluated) 6.7 / 8.0 GiB with no speed gain -- lean stays the default.
  - aiagent verified all three artifacts. Gate at precision 0.95: repair (accuracy 0.731, CP-lower 0.892), repair (0.743, 0.910), stop (0.755, 0.934, no rounds left). The owner then had round 1 re-certified at 0.90: ship, and `distill install` accepted it; 100 shadow calls agreed with the teacher on 0.94 of rows and on all 69 the student would have answered.
  - Step 3: `LAYA_MAX_HOLD_S` = **900 s** (owner decision): the plan's 1.5x would be 210 s, which a dataset 1.5-2x larger would already exceed; 900 s is about 6x the measured holds.
- Status stays In Progress (owner decision, taken before the 0.90 re-certification).

---

## Combined risk register

| Risk | Phase | Mitigation |
| --- | --- | --- |
| The contract drifts between aiagent (producer and consumer) and the trainer. | 2 | devai owns a minimal laya-generic dataset schema; aiagent's metadata travels as an opaque `producer` object; golden answers plus hash bindings are checked on the aiagent side. |
| The stack stays down after a crash or reboot. | 3 | Existing gap (no `devai-infra.service`); recovery is `make cache-up`, which is documented in `docs/router.md`. |
| HF download of base checkpoints. | 1 | Only through `select-models.py` on the host (not the locked lab); revision pin and sha256 per file. |

## Migration / rollback story

- **Rollback:** revert the commits. The new backend is opt-in: it is used only when something requests `:11438`, and nothing else changes when it is idle.
- **Existing installs** see one extra router port, one placeholder container (skipped while its image is not built) and a new store directory. Phase 3 has a behaviour change for everyone: requests get 503 while a training job holds the GPU. That only happens during a job an agent started explicitly.

## Estimated effort

| Phase | Engineering effort | Wall-clock |
| --- | --- | --- |
| Phase 1 | 1 commit, ~250 LoC plus tests | 0.5 day |
| Phase 2 | 1 commit, ~1200 LoC Python plus tests, Dockerfile, lock | 3-4 days |
| Phase 3 | 1 commit, ~400 LoC Go plus table tests, compose, Makefile | 2 days |
| Phase 4 | measurements only, 1 GPU window | 0.5 day |
| Total | 3 commits + a measurement | ~1.5 weeks |

## References

- aiagent design: `docs/design/laya-system1-distillation.md` in devitops-com/aiagent (PR #15, branch `feat/system1-distill`). Section 6 is this plan; sections 8-9 hold the contracts and the ship gate; the measurements are in its Appendix A. The architecture diagram is `docs/design/laya-system1-architecture.svg`. (aiagent's research folder `docs/design/laya-system1/` is not published.)
- laya: github.com/NandhaKishorM/laya, commit 23a1752 (v0.3.20). `laya/common.py` (`build_sequence`, `collate_items`, `proper_reward`), `laya/agent.py` (`_fix_tokenizer_config`, question validation), `scripts/export_onnx.py`, `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` cell 8, `docs/finetune_browser_agent.md` (fully local single-GPU precedent).
- OpenAI fine-tuning jobs API: `POST /v1/fine_tuning/jobs`, the `fine_tuning.job` and `fine_tuning.job.event` objects.
- devai precedents: `scripts/model-sync.py` (restore in `finally`), `scripts/prepare-checkpoint.py` (torch job image plus manifest), `scripts/_probe_hf_common.py` (GPU mutual exclusion).
