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

The evidence is of four kinds. They are reported separately because each is
read differently; the terms are defined in
[statistics-primer.md](statistics-primer.md).

1. **Functional checks:** a behaviour happened, or it did not.
2. **Deterministic numerical checks:** two computations are compared against a
   fixed tolerance. Nothing is sampled, so there is no sampling error (primer
   Sec. 12).
3. **Measurements:** timings and memory from single runs on one host. Each value
   is one observation, not an estimate of a distribution (primer Sec. 1 and
   13).
4. **Statistical estimates:** the student's quality, estimated by aiagent on
   held-out data and on a fresh sample, reported with sample sizes and
   confidence intervals (primer Sec. 3 and 10).

### Functional checks

| What | Date | Result |
| --- | --- | --- |
| Base checkpoint download (`make model-pull NAME=laya-multilingual`) | 2026-09-24 | 5 files, each sha256-checked against the catalog, installed read-only. |
| Lab mounts (`/laya` ro, `/laya/inbox` rw) | 2026-09-24 | Writes to `runs/` and `base/` fail with EROFS; writes to `inbox/` land as the host user. |
| Trainer unit tests (`make test-laya-trainer`, CPU) | 2026-09-24 | 77 pass, as root and as uid 1000: job store, controller API and lifecycle (incl. stop/finish races), dataset contract, a whole job on a tiny fixture model. |
| A whole job on the real `laya-multilingual` base, **CPU** | 2026-09-24 | Through the controller in the image: import, validation, 1 training epoch, ONNX export (batch-2 trace), calibration, parity, golden answers, sealed artifact. |
| Router job-runner behaviour (hold, adoption, model-agnostic launch, empty-body POST) | -- | Go table tests (`make test-router`). |
| **Round trip with aiagent's own code** (devitops-com/aiagent PR #15, `feat/system1-distill` at 7573715), **CPU** | 2026-09-24 | aiagent's `write_dataset` built an 80-row dataset (English, German, Croatian) for the real base. Every `student_tokens` hash from aiagent's torch-free tokenizer port matched laya's `build_sequence`. The trainer trained 1 epoch through the controller, and aiagent's `verify_artifact` accepted the artifact (hashes, binds, 1024/256 limits) and reproduced all 12 golden rows through its onnxruntime runtime. |
| **GPU training, the live router swap and the hold**, on the reference dataset (`make laya-check`, see "Reference check") | 2026-09-25 | The job request evicted the teacher; a teacher request during training got 503 + `Retry-After: 30` + `gpu_held_by_job`; the job succeeded on the GPU (sm_120, torch 2.14.0+cu130, Python 3.14.7, lean mode); the lab's aiagent 0.5.0 accepted the artifact and reproduced its 12 golden rows; the teacher came back in the configuration it had. |
| **A real aiagent campaign** (aiagent 0.5.0 in the lab image, plan Phase 4) | 2026-09-25 | Three training rounds through `:11438`; the 503 hold seen live; the teacher restored to its previous configuration after every round; aiagent's `verify_artifact` accepted all three artifacts, and `distill install` accepted the one that shipped. Measurements and quality estimates below. |

### Deterministic numerical checks

Definitions, from `laya_trainer/parity.py` and `golden.py`:

- **Parity:** computed over every held-out item (row x question). It compares
  the softmax of the torch fp32 logits (model on the CPU) with the softmax of
  ONNX Runtime's fp32 logits. Both are at temperature 1, that is, before the
  fitted calibration is applied. Two values are reported: the maximum
  absolute probability difference over all items and options, and the number of
  items whose top option is the same in both. Above 1e-3 the job fails (exit 7).
- **Golden answers:** the first 40 held-out rows in file order (all of them
  when there are fewer). laya's own `Agent` answers them on the CPU in fp32,
  from the calibrated checkpoint. aiagent's runtime must reproduce the
  `input_ids` exactly and every answer within 1e-3. This is how the calibrated
  path is checked.

A pass means the exported ONNX model computes the same function as the trained
torch model, up to rounding. It says nothing about how good that function is.
In the one case where the same job's parity was recorded twice, it was
bit-identical (5.708210356059062e-7) in both GPU runs of the reference job.
The third run of that job (`86f4`) left no parity value.

| Job | Date | Held-out items | Parity, max abs difference | Same top option | Golden rows | Run directory kept |
| --- | --- | --- | --- | --- | --- | --- |
| CPU job on the real base | 2026-09-24 | 18 | 8.3e-7 | 18/18 | reproduced by onnxruntime within 4.5e-8 | no |
| Reference, `ftjob-f91b6d747c913af35c209ad9` | 2026-09-25 | 12 | 5.7e-7 | 12/12 | 12 of 12, reproduced by aiagent 0.5.0 | yes |
| First `make laya-check`, `ftjob-34be007f3f04f25fb3c26b11` | 2026-09-25 | 12 | 5.7e-7 (bit-identical to the reference) | 12/12 | 12 of 12, reproduced by aiagent 0.5.0 | no (a passing check deletes it) |
| Campaign round 0, `ftjob-6af5edfd74fc76e4556837b9` | 2026-09-25 | 334 | 1.35e-5 | 334/334 | 40 written | yes |
| Campaign round 1 (the installed student), `ftjob-d0e411c9b85d1168ff7ea1fa` | 2026-09-25 | 334 | 4.7e-6 | 334/334 | 40 written | yes |
| Campaign round 2, `ftjob-a40f596f637ba6c331490aea` | 2026-09-25 | 334 | 6.9e-6 | 334/334 | 40 written | yes |
| `fast`-mode job on round 0's data, `ftjob-220e2e9aefbd968f47ab66d4` | 2026-09-25 | 334 | 5.9e-6 | 334/334 | 40 written | yes |

### Measurements (single runs, one host)

Every value below is one observation. All come from one host (RTX PRO 4000
Blackwell, 24 GB) on 2026-09-25, with training seed 42. "Hold" is job started
-> finished in the trainer log (millisecond timestamps). The phase times and
peak VRAM (torch `max_memory_allocated`) come from each run's `manifest.json`,
except for `34be`: the passing check deleted its run directory, so its
values come from the check's log.

| Job | Rows: train / calib / held-out | Hold (s) | Training (s) | Parity on the CPU (s) | Calibration (s) | Export (s) | Peak VRAM, torch (GiB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Reference `f91b` (lean) | 50 / 10 / 12 | 37.4 | 4.1 | 4.9 | 1.8 | 12.2 | 3.17 |
| Reference `34be` (lean) | 50 / 10 / 12 | 36.0 | 4.0 | 5.2 | 1.7 | 12.3 | 3.17 |
| Reference `86f4` (lean) | 50 / 10 / 12 | 36.5 | -- | -- | -- | -- | -- |
| Campaign round 0 (lean) | 1,448 / 277 / 334 | 123.3 | 38.9 | 45.2 | 14.6 | 11.9 | 3.21 |
| Campaign round 1 (lean) | 1,704 / 277 / 334 | 135.2 | 45.2 | 49.6 | 15.2 | 12.1 | 3.21 |
| Campaign round 2 (lean) | 1,789 / 277 / 334 | 140.4 | 47.3 | 47.3 | 16.4 | 12.1 | 3.20 |
| `fast` mode, round 0's data | 1,448 / 277 / 334 | 126.3 | 39.4 | 46.9 | 15.5 | 11.9 | 6.70 |

How to read it:

- **What drives the hold time.**
  - Training grew with the number of training rows: about 26-27 ms per row
    for 4 epochs, over 1,448-1,789 rows.
  - The calibration and held-out splits had the same size in every
    campaign round (277 and 334 rows). So "about 0.14 s per held-out row"
    and "about 0.055 s per calibration row" are ratios at one size, not
    slopes. The small reference job gives 0.41 and 0.18 s per row.
  - Export took about 12 s regardless of size.
  - A straight line through the reference job and round 0 (hold = 34.3 +
    89.0 k s, with k the dataset size relative to round 0) is consistent
    with the observations. With two points it cannot be tested.
- **Run-to-run noise is of the same order as the size effect between
  rounds.** The three runs of the identical reference job differ by 1.4 s.
  On the same 334 held-out rows, the parity phase took 45.2 to 49.6 s, and
  packaging 2.9 to 6.8 s, across the campaign rounds. From round 1 to round
  2 the hold rose 5.2 s, while training rose only 2.1 s.
- **Short texts.** The campaign's texts were short (median 73 and maximum 284
  student tokens in round 0's training split, of 1,024 allowed), so these
  timings are not bounds for datasets of long texts.
- **`fast` mode.** On the same data, `fast` held 126.3 s against lean's 123.3 s
  and used 6.70 GiB against 3.21 GiB. It was no faster in this single
  comparison, so lean stays the default.
- **nvidia-smi figures.** Values quoted elsewhere from nvidia-smi (3.6, 3.7 and
  8.0 GiB) were sampled by hand and are not retained.
- **Teacher cold start** (router log, "waiting for vllm-devai" -> "vllm-devai
  ready", 1 s resolution; Qwen3.8-27B-MTP-devai-NVFP4 at 118784 with MTP): 125,
  126 and 127 s after the three campaign rounds, and 122-135 s over 13 cold starts
  between 2026-09-23 and 2026-09-25. It depends on the engine's caches:
  [nvfp4-coldstart.md](nvfp4-coldstart.md) gives launch-to-ready times by
  cache state, measured from the router's "starting" line, so slightly
  longer. With empty caches on 2026-09-22 they were 261-278 s. From the
  eviction to the teacher being ready again, a campaign round took 280-282
  s.

### Quality of the installed student (aiagent's evaluation)

devai reports parity and loss only; **the ship decision is aiagent's**. Its
evaluation of the 2026-09-25 campaign is summarised here because it determined
what was installed. The source data are in the lab home volume
(`~/devai-home/phase4/`: `eval-r{0,1,2}.json`, `eval-r1-p090.json`,
`fresh-shadow-log.jsonl`) and in the dataset store
(`datasets/ds-e9443d334cb2/heldout.jsonl`). The scripts that recompute every
number are in `scripts/stats/`.

**What is estimated.** Every number measures **agreement between the student and
the teacher's label, not correctness**. The teacher's label for a row is the top
option of the mean of 3 teacher samples at temperature 0.7. On the 334 held-out
rows the 3 samples were unanimous on 281 and split 2-to-1 on 53, with no ties.
Only human labels could measure correctness.

**Sampling frame.** The corpus has 2,400 documents from openly licensed sources,
assembled by quota per source and per source label. It is a purposive corpus,
not a sample of real traffic. Each document is one row, split by the hash of
the document: sha256 mod 100 of 0-59 goes to train, 60-69 to calib, 70-84 to
held-out and 85-99 to pool. The splits are therefore disjoint by document.
Held-out has 334 rows.

**Threshold and gate.**

1. On the calib split (277 rows), aiagent chooses the confidence threshold tau:
   the lowest confidence at which the accepted calib rows reach an empirical
   precision of 0.98.
2. On held-out, the rows with confidence >= tau are *accepted*. Precision is
   correct / accepted, and coverage is accepted / 334.
3. The gate passes a round when the **one-sided 95% Clopper-Pearson lower
   bound** on precision is at least the target, and the coverage (a point
   estimate) is at least 0.20. That bound equals the lower end of the
   two-sided 90% interval.

| Round (aiagent counts from 0) | tau | Correct / accepted | Precision | 95% CI (two-sided) | One-sided 95% lower bound | Coverage (95% CI) | Agreement on all 334 rows (95% CI) | Gate at target 0.95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.929 | 54 / 56 | 0.964 | 0.877-0.996 | 0.892 | 56/334 = 0.168 (0.129-0.212) | 244/334 = 0.731 (0.680-0.777) | repair |
| 1 | 0.894 | 81 / 84 | 0.964 | 0.899-0.993 | 0.910 | 84/334 = 0.251 (0.206-0.302) | 248/334 = 0.743 (0.692-0.789) | repair |
| 2 | 0.957 | 44 / 44 | 1.000 | 0.920-1.000 | 0.934 | 44/334 = 0.132 (0.097-0.173) | 252/334 = 0.754 (0.705-0.800) | stop (no rounds left) |

Intervals are Clopper-Pearson.

**How round 1 came to be installed.** At the design target of 0.95 no round
passed. After seeing these results, the owner lowered the target to 0.90. At
0.90, round 1 is the only round that passes both criteria. Round 0 fails both:
its bound is 0.892 and its coverage 0.168. Round 2's coverage of 0.132 is below
0.20. What this means statistically (primer Sec. 7):

- **Lowering the target is not the statistical problem.** A valid lower bound
  does not depend on the target it is compared with. So the chance of a false
  pass stays at most 5% at 0.90 as at 0.95. The change is a change of
  requirement, and that is the owner's decision.
- **Picking the round is.** The installed round was the one of three that
  passed, judged on the same 334 held-out rows. The 5% error rate applies to
  a single pre-chosen round, not to the best of three. A bound that is valid
  for all three at once (Bonferroni, one-sided alpha 0.05/3) is 0.893 for
  round 1, below 0.90. On the held-out data alone, the claim "round 1's
  precision is at least 0.90" is therefore not supported at the 5% level.
- **aiagent therefore drew new data.** After the install it drew 500 new
  documents ("fresh500"): same sources, per-source quotas proportional to the
  corpus, a seeded random sample within each source, and checked to be disjoint
  from the corpus (exact and near-duplicate). On these, the student's answers
  were compared with one live teacher answer per text, not a 3-sample majority.

| fresh500 subset | Documents | Answered (coverage, 95% CI) | Agreement among answered | 95% CI (two-sided) | One-sided 95% lower bound | Exact one-sided p against 0.90 |
| --- | --- | --- | --- | --- | --- | --- |
| all | 500 | 123 (0.246, 0.209-0.286) | 117/123 = 0.951 | 0.897-0.982 | 0.906 | 0.032 |
| without the 53 Wikipedia lead sentences | 447 | 75 (0.168, 0.134-0.206) | 69/75 = 0.920 | 0.834-0.970 | 0.848 | 0.37 |

The p-values test H0: agreement <= 0.90 against H1: agreement > 0.90.

**What the data show:**

- **On the new sample as a whole, precision among answered rows of at least
  0.90 is supported** at the one-sided 5% level (117/123, lower bound 0.906).
  The two-sided 95% interval (0.897-0.982) just includes 0.90. Three
  qualifications apply:
  - This is precision among the rows the student answered. Agreement on
    *all* 500 documents was 354/500 = 0.708.
  - The reference label differs from the gate's: one live teacher answer
    instead of the majority of 3 samples.
  - The population is the pilot's source mix. Wikipedia lead sentences are
    10.6% of the documents but 39% of the answered rows (48/123).
- **It is not confirmed for opinion text alone.** The student answered 48 of
  the 53 Wikipedia lead sentences and agreed with the teacher on all 48. Without
  them the lower bound is 0.848, so the data are compatible with a true
  agreement below 0.90 on opinion text.
- **No difference between held-out and fresh500 is established,** but they
  are not shown to be equal either. The comparisons are accepted-row
  agreement (0.964 vs 0.951), coverage (0.251 vs 0.246) and agreement on all
  rows (0.743 vs 0.708). None is distinguishable at the 5% level. The
  Newcombe intervals for the differences (-0.056 to +0.072, -0.053 to
  +0.066, -0.028 to +0.095) still allow gaps of 5-10 points. The comparison
  is also confounded (primer Sec. 4.3):
  - the reference labels differ (3-sample majority vs one live answer);
  - the corpus was quota-sampled per source label, while fresh500 was
    sampled per source only.
- **Rounds are not shown to improve.** Agreement on all held-out rows rose from
  0.731 to 0.743 to 0.754, a net change of 4 rows per round. Neither step is
  distinguishable at the 5% level: even if every changed row moved the same
  way, an exact paired test cannot give p below 0.125. The per-row outcomes that
  a round 0 -> 2 test would need were not kept.
- **A first shadow batch of 100 texts is not an estimate.** It reported
  agreement on 94/100, and on all 69 texts the student would have answered. But
  85 of the 100 texts were pool rows left over after repair, which had taken
  the rows the student was least sure of, so the leftovers are the ones it was
  most sure of. The other 15 were held-out rows already used in the gate. The
  sample favours the student by construction.

**Calibration.** The fitted temperature (primer Sec. 11) was 2.12, 2.26 and 2.29
for rounds 0-2 (277 calibration items each). T is fitted against the
teacher's vote shares (the fraction of 3 teacher samples choosing each
option). A value above 1 means the raw student was over-confident relative
to those vote shares, and the fit softens its probabilities. Its sampling
uncertainty cannot be computed, because the calibration logits were not kept.

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
`memory_mode` (`lean`: frozen vocabulary embedding + activation checkpointing;
`fast`: everything trainable). Peak torch VRAM measured on the 2026-09-25
campaign's data (single runs, short texts): lean 3.2 GiB, fast 6.7 GiB. `"auto"`
means the default. `seed` defaults to 42.

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

- **Deadline:** `started_at + LAYA_MAX_HOLD_S` (default 900 s; 0 = no cap),
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

After a job, the next teacher request swaps back the normal way. The cold start
took about 2 minutes: 122-135 s over 13 cold starts between 2026-09-23 and
2026-09-25. It depends on the engine's caches (see "Measurements"). Send it with the exact model string used for labeling (including
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
start about 2; the three reference runs on 2026-09-25 took 178, 198 and 168 s
from eviction to the router's "vllm-devai ready"). Teacher requests in that
window get 503 with `Retry-After`.

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
| `swap` | the teacher port reported `running` before the job request (a teacher adopted by a restarted router, model unknown, counts), and the request is accepted (the router evicted it and started the trainer); with nothing running there nothing is evicted and the check fails as not exercised |
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
numbers. These are single observations, for orientation only; they depend on the
host and its caches:

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
| `LAYA_MAX_HOLD_S` | `900` | Hold cap from the job's start; 0 = none. An engineering margin set by the owner, not an estimate: 6.4x the longest of the three campaign holds (123.3 / 135.2 / 140.4 s, see "Measurements"). A straight line through two job sizes (hold = 34.3 + 89.0 k s, k = dataset size relative to campaign round 0) puts 900 s at roughly 10x the campaign's dataset (about 20,000 labelled rows) at the same short texts, 4 epochs and lean mode -- an extrapolation the two points cannot test. A job past the cap fails as `trainer_stopped`; raise it for larger datasets, longer texts or more epochs. |
| `LAYA_JOB_TIMEOUT_S` | `0` | per-job wall-clock limit inside the trainer (exit 124); 0 = none |
