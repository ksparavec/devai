# laya trainer (`laya-trainer`, router port 11438)

The laya trainer fine-tunes laya "System 1" decision models for aiagent by
distillation from the 27B teacher. It is a router backend, but a **job runner**,
not an inference engine: a request to its port starts a training job that holds
the GPU for minutes to hours after the request has returned. The design and its
owner decisions are in [docs/plans/laya-trainer.md](plans/laya-trainer.md); this
page is the reference for what is built.

- Image: `localhost/devai-laya-trainer:latest` (`make build-laya-trainer`), on the
  lab's GPU base image (`devai-base-gpu`, Python 3.14.7), hash-locked Python
  packages (`laya-trainer/requirements.lock`, `torch==2.14.0` = the lab's cu130
  build).
- Container: `devai-laya-trainer`. Compose starts it as a `sleep infinity`
  placeholder (skipped by `make cache-up` while the image is not built); the
  router recreates it with the controller on the first job request.
- Store: `/var/cache/devai/laya`, mounted into the trainer at `/laya`
  (read-write) and into the lab at `/laya` (read-only, `inbox/` read-write).
- Catalog: `deploy/laya-models.yaml` (base checkpoints; see "Model stores" in
  CLAUDE.md). `make model-pull NAME=laya-multilingual` installs the default
  student's base.

## Verification status

| What | Status |
| --- | --- |
| Base checkpoint download (`make model-pull NAME=laya-multilingual`) | Verified 2026-09-24: 5 files, sha256-checked, installed read-only. |
| Lab mounts (`/laya` ro, `/laya/inbox` rw) | Verified: EROFS on `runs/` and `base/`, writes to `inbox/` land as the host user. |
| Trainer unit tests (`make test-laya-trainer`, CPU) | 77 pass, as root and as uid 1000: job store, controller API and lifecycle (incl. stop/finish races), dataset contract, a whole job on a tiny fixture model. |
| A whole job on the real `laya-multilingual` base, **CPU** | Verified 2026-09-24 through the controller in the image: import, validation, 1 training epoch, ONNX export (batch-2 trace), calibration, parity (max abs difference 8.3e-7, 18/18 argmax), golden answers (reproduced by onnxruntime alone within 4.5e-8), sealed artifact. |
| Router job-runner behaviour (hold, adoption, model-agnostic launch, empty-body POST) | Go table tests (`make test-router`). |
| **Round trip with aiagent's own code** (devitops-com/aiagent PR #15, `feat/system1-distill` at 7573715), **CPU** | Verified 2026-09-24: aiagent's `write_dataset` built an 80-row dataset (English, German, Croatian) for the real laya-multilingual base; the trainer accepted it -- every `student_tokens` hash from aiagent's torch-free tokenizer port matched laya's `build_sequence` -- trained 1 epoch through the controller, and aiagent's `verify_artifact` accepted the artifact (hashes, binds, 1024/256 limits) and reproduced all 12 golden rows through its onnxruntime runtime. |
| **GPU training, the live router swap and the hold**, on the reference dataset | Verified 2026-09-25 (see "Reference check" below): the job request evicted the teacher (trainer ready 14 s after it), a teacher request during training got 503 + `Retry-After: 30` + `gpu_held_by_job`, the job succeeded on the GPU (sm_120, torch 2.14.0+cu130, Python 3.14.7, lean mode) in 31 s of phases, parity 5.7e-7 with 12/12 argmax, the lab's aiagent 0.5.0 accepted the artifact and reproduced 12 golden rows, and the teacher came back as it was (engine cold start 2 min 5 s; the warm-up request 126.6 s). |
| **`LAYA_MAX_HOLD_S` from a real aiagent job** | **Not verified.** Plan Phase 4: needs a real dataset from aiagent; the reference job is too small to size the cap. |

## API (OpenAI fine-tuning jobs subset)

Through the router: `http://devai-router:11438/v1` (aiagent's default trainer API
base; `devai-router` is in the lab's `NO_PROXY`). aiagent's default distill
directory is `/laya`, the lab mount; the lab's home volume is persistent, which
aiagent's installed students (`~/.local/share/aiagent`, about 1.3 GB each) need.

| Method and path | Behaviour |
| --- | --- |
| `POST /v1/fine_tuning/jobs` | Start a job. Body: `model` (a catalog name), `training_file` (a dataset id, `ds-<12 hex>`, in `inbox/`), optional `hyperparameters`, `seed`, `suffix` (`[a-z0-9-]`, max 40), `metadata` (max 16 string pairs). One job at a time: 409 `job_in_progress` otherwise. Returns the `fine_tuning.job` object. |
| `GET /v1/fine_tuning/jobs[?limit&after]` | Jobs, newest first. |
| `GET /v1/fine_tuning/jobs/{id}` | One job: `status` validating_files / running / succeeded / failed / cancelled, `fine_tuned_model`, `result_files`, `error {code, message}`, `devai {phase, started_at, exit_code, run_dir}`. |
| `GET /v1/fine_tuning/jobs/{id}/events` | `fine_tuning.job.event` objects, newest first (phase changes, per-epoch loss). |
| `POST /v1/fine_tuning/jobs/{id}/cancel` | Stop the job; the body may be empty (the router accepts that on this port only). |
| `GET /v1/models` | Base checkpoints (the router lists the catalog names). |

`hyperparameters`: `n_epochs` (4, 1-50), `batch_size` (32 examples per optimizer
update, 1-512), `learning_rate_multiplier` (1.0; scales 2.5e-5 encoder / 1e-4
head), and devai's `micro_batch_size` (8, examples per forward pass) and
`memory_mode` (`lean`: frozen vocabulary embedding + activation checkpointing,
estimated 4-5 GiB; `fast`: everything trainable, about 9-10 GiB). `"auto"` means
the default. `seed` defaults to 42.

**Status reads never launch the trainer.** A request without a model on port
11438 is proxied to a resident trainer, and answered 503 when it is not resident:
the job record is on the volume at `/laya/runs/<job>/job.json` (+ `events.jsonl`),
which is the source of truth. The router answers `GET /health` on every port
itself; on 11438 its body carries `job_runner {status, job, phase, started_at,
hold_until}` relayed from the trainer.

## The GPU hold

While the trainer's `/health` says `busy` -- from the moment a job is accepted
until its record is final, packaging included -- a request that would need the
GPU for any other backend gets **503 with `Retry-After: 30`** and
`error.code = "gpu_held_by_job"`, instead of evicting the trainer. It applies to
every eviction path (vLLM, SGLang, vllm-devai, Ollama including its model-less
surfaces) and to the idle sweep.

- **Deadline:** `started_at + LAYA_MAX_HOLD_S` (default 7200 s; 0 = no cap),
  computed by the router from the trainer's own `started_at`. Past it the router
  evicts anyway; the trainer marks the job failed (`trainer_stopped`).
- **Boot adoption:** a restarted router adopts the trainer unless nothing listens
  on its port (the placeholder) or podman reports its container gone, and reads
  the hold from its `/health`, so the original deadline stands.
- **Race:** the trainer's in-flight requests are drained before its `/health` is
  read, so a job submission still in flight cannot be killed by a concurrent
  request for another backend.
- **Silence:** a refused connection means nothing listens (the placeholder, or a
  controller still starting): it holds nothing. A `/health` that times out,
  errors or fails to resolve, three times, while podman reports the container
  running (or paused), HOLDS -- as its last busy answer if it had one, else from
  the moment it went silent -- until the deadline. A container podman reports
  gone (exited, stopped, dead, created, or unknown to libpod) holds nothing.
  The router never declares a listening trainer dead from `/health` alone.
  **With `LAYA_MAX_HOLD_S=0` a hung controller (listening, never answering)
  holds the GPU until someone stops it** (`make cache-down`); keep a cap.
- **Cost:** a hold verdict is reused for 30 s (= `Retry-After`), so a burst of
  refused requests costs one probe; a silent trainer's probe takes up to about
  7 s. A streaming request refused after the SSE keepalive grace gets the hold
  in-band (`code: gpu_held_by_job`, `retry_after`).

After a job, the next teacher request swaps back the normal way (about 2 minutes
of cold start). Send it with the exact model string used for labeling (including
`::<reasoning>@<ctx>`); the router's `/health` on the teacher's port shows
`current_model`, `current_context` and `current_spec` before the swap.

## Dataset contract (aiagent -> devai), `schema_version: 1`

Agreed with aiagent on 2026-09-24 (its implementation spec, section 3.3, holds
the same rules; its validator mirrors the trainer's). A directory
`inbox/ds-<12 hex>/`, where the id is `"ds-"` + the first 12 hex of the sha256 of
the `manifest.json` bytes (checked: an id that does not name its manifest is
refused). The trainer imports it to `datasets/<id>/`, read-only; an existing id
with other contents is refused.

| File | Content |
| --- | --- |
| `manifest.json` | `schema_version`; `base_checkpoint {name, revision, weights_sha256, tokenizer_sha256}` (must equal the catalog row: sha256 of `model.safetensors` and of `tokenizer/tokenizer.json`); `max_len`, `head_max_len` (the ONLY source of the token limits; aiagent uses the base's, 1024/256 for laya-multilingual); `splits {method, counts}`; `files {name: {sha256, rows}}` for exactly the row files; `producer` (opaque, except `producer.binds`). |
| `train.jsonl`, `calib.jsonl`, `heldout.jsonl` | Required, non-empty. |
| `pool.jsonl` | Optional for the trainer (aiagent always writes it); counted and token-checked, never trained on. |
| `SHA256SUMS` | `sha256sum` lines covering exactly the files above. No other files, no subdirectories, no symlinks. |

Each row: `id` (unique), `group_id`, `split` (= its file), `synthetic` (bool;
`false` in calib and held-out), `state` (string, object or list),
`questions {qid: laya question}`, `student_tokens {qid: {n, ids_sha256}}`, and,
in train/calib/held-out only, `gold {qid: {label, probabilities}}` and a
`teacher` object (pool rows carry neither key). Same question keys throughout.

- **No nulls** anywhere in a row except inside `questions` (laya's question
  objects, where a null criterion description is legitimate).
- `probabilities` keys: the option keys **in order** (choice keys in criteria
  order, `"0".."n-1"` for `score`, `"false","true"` for `noul`); values finite
  and >= 0, summing to 1 +- 1e-6; `label` must be the **first** argmax.
- `student_tokens.<qid>.ids_sha256 = sha256(json.dumps(ids, separators=(",", ":")))`
  of the ids laya's own `build_sequence` produces for the row (state tokenized
  once; `truncate_left=False` for object states) -- exactly what laya's runtime
  feeds the model. The trainer re-tokenizes every row (pool included) and refuses
  a mismatch before any GPU time.
- The state is never truncated: it must fit the room `max_len` leaves after the
  question and options.
- The `group_id`s of calib, held-out and train+pool are pairwise disjoint.

Any violation is exit 3. The trainer trains on train, calibrates on calib, and
checks parity and writes golden answers on held-out.

## Artifact contract (devai -> aiagent), `format_version: 1`

`runs/<job>/`. **The artifact is `manifest.json` plus every key of
`manifest.files`**; when the job succeeds the whole run directory is made
read-only (files 0444, directories 0555, so readable by the lab user).

| File | Content |
| --- | --- |
| `model.onnx` (+ `model.onnx.data`) | `torch.onnx.export(dynamo=True)`, opset 18, fp32, external data at the default threshold; inputs `input_ids`, `attention_mask`, `marker_pos`, `marker_mask` (bool), `qtype`; outputs `logits`, `act_logits`; dynamic batch / sequence / marker axes (traced at batch 2, sequence 320, 3 markers; questions need at least 2 options). |
| `tokenizer/` | the base checkpoint's tokenizer files, copied byte for byte (never re-saved). |
| `rl_agent_config.json` | the base config with the dataset's `max_len` / `head_max_len`, the **applied** per-type temperatures and `"temperature_by_options": {}`. |
| `golden.jsonl` | one row per held-out dataset row (at most 40): `{id, state, questions, expected: {qid: {input_ids, markers, answer}}}`, where `answer` is laya's own `Agent.predict(...)["answers"][qid]` without `action`, computed on CPU in fp32 (`LAYA_CPU_AMP` unset) from the calibrated checkpoint. aiagent must reproduce `input_ids` exactly and the answer within 1e-3. |
| `NOTICE` | attribution. |
| `manifest.json` | `format_version`, `artifact_id` (64 hex), `job_id`, `fine_tuned_model` (`ft:<base>:devai:<suffix>:<12 hex of the job id>`), `base_checkpoint`, `dataset {id, manifest_sha256, counts}`, `binds` (the dataset's `producer.binds` copied verbatim, plus `dataset_manifest_sha256` = sha256 of the dataset's `manifest.json` bytes), `laya {version, commit}`, `trainer {build, image, versions, device, hyperparameters, seed, timings, peak_vram_gib}`, `calibration {temperature_raw, temperature_applied, clamp [0.5, 5.0], per_type_n}` (fitted on the `calib` split's ONNX fp32 logits; fewer than 10 items of a type keeps 1.0), `export`, `parity {max_abs_prob_diff, argmax_agreement, tolerance 1e-3}`, `golden {n, tolerance, file}`, `metrics` (torch-side loss only), `files {path: {sha256, size}}` (never under `checkpoint/`). |

Not part of the artifact, in the same directory:

- `checkpoint/`: the torch-loadable fp32 checkpoint (laya's `Agent` loads it; its
  `rl_agent_config.json` equals the artifact's). Listed in `SHA256SUMS`, not in
  `manifest.files`; aiagent never copies it.
- `SHA256SUMS`: exactly `manifest.json`, the `manifest.files` keys, and the
  `checkpoint/` files.
- `job.json` (the `fine_tuning.job` object from creation on), `events.jsonl` (one
  `fine_tuning.job.event` per line), `outcome.json`: job bookkeeping, listed
  nowhere.

devai reports loss and parity only: **the ship decision is aiagent's**, made
through its own ONNX runtime. `laya_trainer.fixture` builds a tiny checkpoint and
dataset for devai's own tests; aiagent generates its own fixture.

## Failures and exit codes

| Exit | `error.code` | Meaning |
| --- | --- | --- |
| 3 | `dataset_contract_violation` | anything in the dataset contract above |
| 4 | `base_checkpoint_mismatch` | base missing, not matching the catalog, or the dataset built for another base |
| 5 | `gpu_unavailable_or_oom` | no CUDA in the container, or out of GPU memory (try `memory_mode: lean`) |
| 6 | `training_diverged` | non-finite loss |
| 7 | `export_or_parity_failure` | ONNX export failed, or torch-vs-ONNX parity above 1e-3 |
| 124 | `timeout` | the job exceeded `LAYA_JOB_TIMEOUT_S` (0 = none, the default) |
| 1 | `unexpected_error` | anything else (traceback in the container log) |
| -- | `trainer_stopped` | the container was stopped mid-job: hold cap, `make cache-down`, crash |

A job runs as a subprocess of the controller, so a crash or OOM ends the job, not
the trainer; epoch checkpoints already written stay in `runs/<job>/checkpoint/`.
Logs: `make logs SERVICE=devai-laya-trainer`.

## Operator commands

```bash
make model-pull NAME=laya-multilingual   # base checkpoint (default student); laya-english likewise
make build-laya-trainer                  # the image (needs devai-base-gpu; builds it via build-base-gpu)
make test-laya-trainer [VERBOSE=1]       # unit tests in the image, CPU, no network
make laya-trainer-lock                   # regenerate the hash lock from laya-trainer/requirements.in
make laya-check [TEACHER=...] [KEEP_RUN=1]  # GPU, evicts the teacher ~3 min: the reference check below
python -m laya_trainer.fixture <dir>     # (in the image) a tiny laya checkpoint for aiagent's tests
```

## Reference check (`make laya-check`)

One small, known job run through the router the way aiagent runs one, to check
every hand-off between the teacher and the trainer. Run it after rebuilding the
router or the trainer image, after an aiagent upgrade in the lab image, or
whenever the two sides seem out of step. **It is GPU-exclusive and evicts the
teacher for about three minutes** (the job about 1 minute, the teacher's cold
start about 2); teacher requests in that window get 503 with `Retry-After`.

The reference dataset is `ds-a6c9c8248242`, kept in the store (`datasets/`, and
`inbox/`) since 2026-09-25: 80 synthetic sentences in English, German and
Croatian for aiagent's `polarity` skill, labelled by construction, with
placeholder binds and teacher `"interop"`. **Never train a real student on it,
evaluate it with `aiagent distill eval`, or install its artifact.** It was written
by aiagent's own writer (`scripts/laya_check/write_dataset.py`); when it is gone
from both `inbox/` and `datasets/`, the check rewrites it the same way, and the
rewrite must come out under the same id (the id hashes the content; verified
byte-identical with the lab's aiagent 0.5.0). A different id means aiagent's
dataset output changed: the check stops before touching the GPU.

| Check | Passes when |
| --- | --- |
| `dataset` | the reference dataset is present, or rewritten under its id, and the base checkpoint it was built for is staged in `base/` (checked before anything is evicted: the trainer would refuse the job only after the swap) |
| `swap` | a teacher was resident on the teacher port, and the job request is accepted (the router evicted it and started the trainer); with no resident teacher nothing is evicted and the check fails as not exercised |
| `hold` | a teacher request, sent the first time the job is seen running, gets 503 with `Retry-After` and `gpu_held_by_job` -- a 200 means the router evicted a running job |
| `job` | the job succeeds, exit code 0, within 30 minutes (otherwise it is cancelled) |
| `data` | the manifest names the reference dataset, its split counts (50/10/12/8) and 8172 trained tokens |
| `parity` | torch-vs-ONNX difference within the manifest's tolerance, argmax agreement n/n |
| `aiagent` | the lab's aiagent `verify_artifact` accepts the artifact and `check_golden` reproduces all 12 golden rows |
| `teacher` | the warm-up returns 200 and `/health` on the teacher port shows the model, context and MTP spec of the teacher string. The router keeps a hold verdict for `Retry-After` (30 s) after a job ends, so a warm-up refused with `gpu_held_by_job` is retried per `Retry-After`, within the launch timeout (900 s) |

The teacher string is `TEACHER=` when given, else what the teacher port's
`/health` reports as loaded (model, context, MTP on/off). Its suffixes are read
in any order, as the router reads them. When nothing is known -- the router
adopted the teacher at boot with the model unknown -- the check stops before the
job request; pass `TEACHER=` (for aiagent's labeling:
`Qwen3.8-27B-MTP-devai-NVFP4::nothink::mtp@118784`). `TEACHER_PORT` (default
11437) is the teacher's router port. The check also stops before the job request
when a job is already running on the trainer. After the job request it always
restores the teacher -- on success, failure, a 30-minute timeout (which cancels
the job) or an error in the driver -- except after a 409, when another job holds
the trainer. It stops polling when the router serves the teacher during the job
(the job was evicted) or when five polls in a row are not answered with the
job, and then reads the job's final record from the volume.

Measured on the first run, 2026-09-25, and printed next to every run's own
numbers (informational; they depend on the host and its caches):

| Measure | Reference |
| --- | --- |
| Job request, including the swap (drain, stop teacher, start trainer) | 13.6 s |
| Phases: validating / training / exporting / calibrating / checking_parity / packaging | 5.5 / 4.1 / 12.2 / 1.8 / 4.9 / 2.1 s (total 30.7 s) |
| Peak VRAM, torch (`max_memory_allocated`, printed) / nvidia-smi (sampled by hand on the first run, not measured by the check) | 3.17 / 3.6 GiB |
| Parity, max abs probability difference | 5.7e-7 |
| Teacher warm-up (vllm-devai, Qwen3.8-27B-MTP-devai-NVFP4, 118784, MTP) | 126.6 s |

The job is driven from a container on `devai-net` (the router publishes no host
ports; `scripts/laya_check/drive_job.py`, stdlib only, in the trainer image), and
aiagent's steps run in the lab image with aiagent's own Python, without network
or GPU, `/laya` read-only. A passing run deletes its own run directory (about
1.3 GB); a failing one keeps it, as does `KEEP_RUN=1`. The first run's directory,
`runs/ftjob-f91b6d747c913af35c209ad9`, is kept as the reference artifact and is
never removed by the check. Pinned by `tests/python/test_laya_check.py`.

First `make laya-check`, 2026-09-25: PASS, every check. The job request with the
swap took 3.4 s, the phases 30.2 s, peak VRAM 3.17 GiB, parity 5.7e-7. The first
warm-up got the router's cached hold (503 `gpu_held_by_job`, sent 2 s after the
job ended); the second, 31 s later, brought the teacher back as it was (159 s in
all). That is the case the retry exists for.

## Configuration (router env, set in deploy/docker-compose.yaml)

| Variable | Default | Purpose |
| --- | --- | --- |
| `LAYA_TRAINER_PORT` | `11438` | router listen port |
| `LAYA_TRAINER_URL` | `http://laya-trainer:11434` | upstream |
| `LAYA_TRAINER_CONTAINER` | `devai-laya-trainer` | name to recreate |
| `LAYA_TRAINER_IMAGE` | `localhost/devai-laya-trainer:latest` | image to launch (also recorded in manifests) |
| `LAYA_STORE_DIR` | `/var/cache/devai/laya` | host path bound to `/laya` (read-write) |
| `LAYA_CATALOG_FILE` | `/etc/devai/laya-models.yaml` | allowlist of base names |
| `LAYA_MAX_HOLD_S` | `7200` | hold cap from the job's start; 0 = none. To be set from the first measured job. |
| `LAYA_JOB_TIMEOUT_S` | `0` | per-job wall-clock limit inside the trainer (exit 124); 0 = none |
