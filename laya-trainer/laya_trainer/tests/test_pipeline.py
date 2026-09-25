"""A whole job on CPU with the tiny fixture, and the artifact it leaves.

The artifact is checked the way aiagent will check it: through its own files
(SHA256SUMS, manifest.json) and by reproducing golden.jsonl from model.onnx +
rl_agent_config.json with onnxruntime alone.
"""

from __future__ import annotations

import json
import stat
import unittest
from pathlib import Path

import numpy as np

from ..contract import ExitCode, JobError, read_sha256sums, sha256_file
from ..jobs import JobStore, resolve_hyperparameters
from .util import FixtureStore, needs_laya, tmpdir


def _job(store: JobStore, fx: FixtureStore, ds_id: str, **hp) -> dict:
    from .. import catalog as cat
    return store.create(model=fx.row["name"], training_file=ds_id,
                        hyperparameters=resolve_hyperparameters({"n_epochs": 2, "batch_size": 8, **hp}),
                        seed=7, suffix="polarity", metadata=None,
                        base_dirname=cat.checkpoint_dirname(fx.row))


@needs_laya
class PipelineTest(unittest.TestCase):
    """One real (tiny) job, shared by the assertions below."""

    @classmethod
    def setUpClass(cls) -> None:
        from .. import run_job
        holder = unittest.TestCase()
        cls._cleanups = holder
        cls.root = tmpdir(holder)
        cls.fx = FixtureStore(cls.root)
        cls.store = JobStore(cls.fx.store)
        cls.ds_id = cls.fx.dataset()
        cls.job = _job(cls.store, cls.fx, cls.ds_id)
        cls.outcome = run_job.run(cls.fx.store, cls.fx.catalog, cls.job["id"], "cpu")
        cls.run_dir = cls.store.job_dir(cls.job["id"])
        cls.manifest = json.loads((cls.run_dir / "manifest.json").read_text())

    @classmethod
    def tearDownClass(cls) -> None:
        cls._cleanups.doCleanups()

    def test_outcome(self) -> None:
        self.assertEqual(self.outcome["fine_tuned_model"],
                         f"ft:laya-fixture:devai:polarity:{self.job['id'][6:18]}")
        self.assertGreater(self.outcome["trained_tokens"], 0)
        self.assertEqual(self.outcome["result_files"], [f"runs/{self.job['id']}/manifest.json"])

    def test_the_artifact_files_are_there(self) -> None:
        for rel in ("model.onnx", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
                    "rl_agent_config.json", "golden.jsonl", "manifest.json", "SHA256SUMS", "NOTICE",
                    "checkpoint/model.safetensors", "checkpoint/encoder/config.json"):
            self.assertTrue((self.run_dir / rel).is_file(), rel)

    def test_sha256sums_are_the_artifact_plus_checkpoint(self) -> None:
        sums = read_sha256sums(self.run_dir / "SHA256SUMS")
        for rel, digest in sums.items():
            self.assertEqual(sha256_file(self.run_dir / rel), digest, rel)
        artifact = {"manifest.json", *self.manifest["files"]}
        extra = {rel for rel in sums if rel not in artifact}
        self.assertTrue(extra and all(rel.startswith("checkpoint/") for rel in extra), extra)
        self.assertEqual(artifact - set(sums), set())
        for job_state in ("job.json", "events.jsonl", "outcome.json"):
            self.assertNotIn(job_state, sums)

    def test_manifest_files_are_the_artifact_only(self) -> None:
        files = set(self.manifest["files"])
        self.assertIn("model.onnx", files)
        self.assertIn("golden.jsonl", files)
        self.assertIn("tokenizer/tokenizer.json", files)
        self.assertFalse([f for f in files if f.startswith("checkpoint/")])
        self.assertFalse(files & {"manifest.json", "SHA256SUMS", "job.json", "events.jsonl"})

    def test_manifest_contract(self) -> None:
        m = self.manifest
        self.assertEqual(m["format_version"], 1)
        self.assertRegex(m["artifact_id"], r"^[0-9a-f]{64}$")
        self.assertEqual(m["base_checkpoint"]["name"], "laya-fixture")
        ds_manifest = self.fx.store / "datasets" / self.ds_id / "manifest.json"
        self.assertEqual(m["binds"], {"signature_sha256": "a" * 64,
                                      "question_set_sha256": m["binds"]["question_set_sha256"],
                                      "skill_source_sha256": "b" * 64,
                                      "dataset_manifest_sha256": sha256_file(ds_manifest)})
        self.assertEqual(m["laya"], {"version": "0.3.20", "commit": "23a1752"})
        self.assertEqual(m["export"]["opset"], 18)
        self.assertEqual(m["golden"]["n"], 12)
        self.assertEqual(m["parity"]["split"], "heldout")
        self.assertLessEqual(m["parity"]["max_abs_prob_diff"], 1e-3)
        self.assertEqual(len(m["metrics"]["loss_per_epoch"]), 2)
        for rel, meta in m["files"].items():
            self.assertEqual(sha256_file(self.run_dir / rel), meta["sha256"], rel)

    def test_the_tokenizer_is_the_base_tokenizer_byte_for_byte(self) -> None:
        for name in ("tokenizer.json", "tokenizer_config.json"):
            self.assertEqual((self.run_dir / "tokenizer" / name).read_bytes(),
                             (self.fx.base_dir / "tokenizer" / name).read_bytes())

    def test_calibration_is_clamped_and_recorded(self) -> None:
        cal = self.manifest["calibration"]
        self.assertEqual(cal["fitted_on"], "onnx_fp32_logits")
        self.assertEqual(len(cal["temperature_raw"]), 3)
        for t in cal["temperature_applied"]:
            self.assertTrue(0.5 <= t <= 5.0, t)
        cfg = json.loads((self.run_dir / "rl_agent_config.json").read_text())
        self.assertEqual(cfg["temperature"], cal["temperature_applied"])
        self.assertEqual(cfg["temperature_by_options"], {})
        self.assertEqual((cfg["max_len"], cfg["head_max_len"]), (128, 48))
        ckpt_cfg = json.loads((self.run_dir / "checkpoint" / "rl_agent_config.json").read_text())
        self.assertEqual(ckpt_cfg, cfg)

    def test_golden_rows_have_aiagents_shape(self) -> None:
        rows = [json.loads(line) for line in (self.run_dir / "golden.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 12)  # one per held-out dataset row
        r = rows[0]
        self.assertEqual(list(r), ["id", "state", "questions", "expected"])
        self.assertEqual(set(r["expected"]), set(r["questions"]))
        for exp in r["expected"].values():
            self.assertEqual(set(exp), {"input_ids", "markers", "answer"})
            self.assertNotIn("action", exp["answer"])
            self.assertIn("answer_confidence", exp["answer"])

    def test_golden_reproduces_with_onnxruntime_alone(self) -> None:
        """What aiagent's torch-free runtime does: ids -> ONNX -> logits / T -> softmax."""
        import onnxruntime as ort
        cfg = json.loads((self.run_dir / "rl_agent_config.json").read_text())
        sess = ort.InferenceSession(str(self.run_dir / "model.onnx"), providers=["CPUExecutionProvider"])
        qtypes = {"choice": 0, "score": 1, "noul": 2}
        for line in (self.run_dir / "golden.jsonl").read_text().splitlines():
            r = json.loads(line)
            for qid, exp in r["expected"].items():
                k, qt = len(exp["markers"]), qtypes[r["questions"][qid]["type"]]
                feeds = {"input_ids": np.array([exp["input_ids"]], dtype=np.int64),
                         "attention_mask": np.ones((1, len(exp["input_ids"])), dtype=np.int64),
                         "marker_pos": np.array([exp["markers"]], dtype=np.int64),
                         "marker_mask": np.ones((1, k), dtype=bool),
                         "qtype": np.array([qt], dtype=np.int64)}
                z = sess.run(["logits"], feeds)[0][0, :k].astype(np.float64) / cfg["temperature"][qt]
                p = np.exp(z - z.max())
                p /= p.sum()
                ans = exp["answer"]
                want = [ans["noul"]] if ans["type"] == "noul" else list(ans["probabilities"].values())
                got = [p[1]] if ans["type"] == "noul" else list(p)
                self.assertLess(max(abs(a - b) for a, b in zip(got, want)), 1e-3)

    def test_golden_input_ids_are_layas_tokenization(self) -> None:
        from laya.agent import Agent
        from laya.common import build_sequence
        r = json.loads((self.run_dir / "golden.jsonl").read_text().splitlines()[0])
        for qid, exp in r["expected"].items():
            q = Agent._to_internal(r["questions"][qid])
            ids, markers = build_sequence(self.fx.tok, r["state"], q, 128, 48)
            self.assertEqual((ids, markers), (exp["input_ids"], exp["markers"]))

    def test_the_checkpoint_loads_in_laya(self) -> None:
        from laya.agent import Agent
        agent = Agent(str(self.run_dir / "checkpoint"), device="cpu")
        answer = agent.predict({"text": "the delivery was great"},
                               {"p": {"type": "noul", "instructions": "Is the passage positive?"}})
        self.assertIn("noul", answer["answers"]["p"])

    def test_events_record_the_phases(self) -> None:
        messages = [e["message"] for e in self.store.events(self.job["id"])]
        for phase in ("importing", "validating", "training", "exporting", "calibrating",
                      "checking_parity", "packaging"):
            self.assertIn(f"Phase: {phase}", messages)
        self.assertTrue(any(m.startswith("Epoch 2/2") for m in messages))


@needs_laya
class JobFailureTest(unittest.TestCase):

    def test_a_contract_violation_is_exit_3_before_training(self) -> None:
        from .. import run_job
        fx = FixtureStore(tmpdir(self))
        store = JobStore(fx.store)
        ds_id = fx.dataset()
        (fx.store / "inbox" / ds_id / "SHA256SUMS").write_text("")
        job = _job(store, fx, ds_id)
        with self.assertRaises(JobError) as cm:
            run_job.run(fx.store, fx.catalog, job["id"], "cpu")
        self.assertEqual(cm.exception.code, ExitCode.DATASET)
        self.assertNotIn("Phase: training", [e["message"] for e in store.events(job["id"])])

    def test_cuda_missing_is_exit_5(self) -> None:
        import torch
        if torch.cuda.is_available():
            self.skipTest("a GPU is present")
        from .. import run_job
        fx = FixtureStore(tmpdir(self))
        store = JobStore(fx.store)
        job = _job(store, fx, fx.dataset())
        with self.assertRaises(JobError) as cm:
            run_job.run(fx.store, fx.catalog, job["id"], "cuda")
        self.assertEqual(cm.exception.code, ExitCode.GPU)

    def test_main_writes_the_outcome(self) -> None:
        from .. import run_job
        fx = FixtureStore(tmpdir(self))
        store = JobStore(fx.store)
        job = store.create(model="laya-fixture", training_file="ds-000000000000",
                           hyperparameters=resolve_hyperparameters({}), seed=1, suffix=None,
                           metadata=None, base_dirname="x")
        rc = run_job.main(["--store", str(fx.store), "--catalog", str(fx.catalog),
                           "--job", job["id"], "--device", "cpu"])
        self.assertEqual(rc, 3)
        outcome = json.loads((store.job_dir(job["id"]) / "outcome.json").read_text())
        self.assertEqual(outcome["exit_code"], 3)
        self.assertIn("inbox/ds-000000000000 not found", outcome["message"])


@needs_laya
class CheckpointTest(unittest.TestCase):

    def test_a_checkpoint_can_be_replaced_every_epoch_from_a_sealed_base(self) -> None:
        """The base is sealed (0555 dirs); its tokenizer copy must not inherit
        that, or the second epoch's checkpoint cannot replace the first for any
        user but root."""
        from .. import train
        fx = FixtureStore(tmpdir(self))
        model = train.load_model(fx.base_dir, fx.cfg)
        out = fx.store / "runs" / "ckpt"
        for epoch in (1, 2):
            train.save_checkpoint(model, out, base_dir=fx.base_dir, cfg=fx.cfg, meta={"epoch": epoch})
            for d in (out, out / "tokenizer", out / "encoder"):
                self.assertTrue(stat.S_IMODE(d.stat().st_mode) & 0o200, f"{d} not owner-writable")
        self.assertEqual(json.loads((out / "checkpoint_meta.json").read_text())["epoch"], 2)


class ParityGateTest(unittest.TestCase):

    def test_drift_beyond_tolerance_is_exit_7(self) -> None:
        from ..parity import check_parity
        items = [{"markers": [1, 2]}]
        with self.assertRaises(JobError) as cm:
            check_parity(items, [np.array([0.0, 0.0])], [np.array([0.0, 0.1])])
        self.assertEqual(cm.exception.code, ExitCode.EXPORT)

    def test_agreement_within_tolerance_passes(self) -> None:
        from ..parity import check_parity
        report = check_parity([{"markers": [1, 2]}], [np.array([0.3, -0.2])], [np.array([0.3, -0.2])])
        self.assertEqual(report["argmax_agreement"], "1/1")


@needs_laya
class CalibrationTest(unittest.TestCase):

    def test_a_sharpening_fit_is_clamped_to_laya_bounds(self) -> None:
        from ..calibrate import calibrate
        # Logits that are too flat for confident targets: the raw fit sharpens
        # (T < 0.5), laya would clamp it, so the applied value is 0.5.
        items = [{"qtype": 2, "target": [0.02, 0.98]} for _ in range(20)]
        logits = [np.array([0.0, 0.2]) for _ in range(20)]
        cal = calibrate(items, logits)
        self.assertLess(cal["temperature_raw"][2], 0.5)
        self.assertEqual(cal["temperature_applied"][2], 0.5)
        self.assertIsNone(cal["temperature_raw"][0])  # no choice items: not fitted
        self.assertEqual(cal["temperature_applied"][0], 1.0)


if __name__ == "__main__":
    unittest.main()
