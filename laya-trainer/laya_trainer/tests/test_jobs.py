"""Job store and hyperparameters (standard library only)."""

from __future__ import annotations

import json
import unittest

from ..contract import ExitCode, canonical_hash, error_code, read_sha256sums, write_sha256sums
from ..jobs import HYPERPARAMETER_DEFAULTS, JobStore, resolve_hyperparameters
from .util import tmpdir


class HyperparameterTest(unittest.TestCase):

    def test_defaults(self) -> None:
        self.assertEqual(resolve_hyperparameters(None), HYPERPARAMETER_DEFAULTS)
        self.assertEqual(resolve_hyperparameters({}), HYPERPARAMETER_DEFAULTS)

    def test_auto_means_the_default(self) -> None:
        hp = resolve_hyperparameters({"n_epochs": "auto", "batch_size": "auto"})
        self.assertEqual(hp["n_epochs"], 4)
        self.assertEqual(hp["batch_size"], 32)

    def test_values_are_taken(self) -> None:
        hp = resolve_hyperparameters({"n_epochs": 2, "batch_size": 8,
                                      "learning_rate_multiplier": 2, "memory_mode": "fast"})
        self.assertEqual((hp["n_epochs"], hp["batch_size"], hp["memory_mode"]), (2, 8, "fast"))
        self.assertEqual(hp["learning_rate_multiplier"], 2.0)

    def test_bad_values_name_the_key(self) -> None:
        for raw, key in (({"n_epochs": 0}, "n_epochs"), ({"n_epochs": True}, "n_epochs"),
                         ({"batch_size": 1000}, "batch_size"),
                         ({"learning_rate_multiplier": 0}, "learning_rate_multiplier"),
                         ({"memory_mode": "huge"}, "memory_mode"), ({"warmup": 1}, "warmup")):
            with self.assertRaises(ValueError) as cm:
                resolve_hyperparameters(raw)
            self.assertIn(key, str(cm.exception))


class JobStoreTest(unittest.TestCase):

    def setUp(self) -> None:
        self.store = JobStore(tmpdir(self))

    def _create(self) -> dict:
        return self.store.create(model="laya-multilingual", training_file="ds-0123456789ab",
                                 hyperparameters=resolve_hyperparameters({}), seed=42,
                                 suffix="polarity", metadata={"round": "0"},
                                 base_dirname="laya-multilingual@55cf4c4ebb4e")

    def test_create_writes_an_openai_job_object(self) -> None:
        job = self._create()
        self.assertRegex(job["id"], r"^ftjob-[0-9a-f]{24}$")
        on_disk = json.loads((self.store.job_dir(job["id"]) / "job.json").read_text())
        self.assertEqual(on_disk, job)
        for key in ("object", "model", "created_at", "finished_at", "fine_tuned_model", "status",
                    "training_file", "hyperparameters", "result_files", "error", "seed"):
            self.assertIn(key, job)
        self.assertEqual(job["object"], "fine_tuning.job")
        self.assertEqual(job["status"], "validating_files")
        self.assertEqual(job["devai"]["run_dir"], f"runs/{job['id']}")

    def test_update_merges_devai(self) -> None:
        job = self._create()
        self.store.update(job["id"], status="running", devai={"phase": "training"})
        again = self.store.read(job["id"])
        self.assertEqual(again["status"], "running")
        self.assertEqual(again["devai"]["phase"], "training")
        self.assertEqual(again["devai"]["run_dir"], job["devai"]["run_dir"])

    def test_events_are_newest_first(self) -> None:
        job = self._create()
        self.store.add_event(job["id"], "second")
        self.store.add_event(job["id"], "third", data={"epoch": 1})
        events = self.store.events(job["id"])
        self.assertEqual([e["message"] for e in events], ["third", "second", "Job created"])
        self.assertEqual(events[0]["type"], "metrics")
        self.assertEqual(events[0]["object"], "fine_tuning.job.event")

    def test_a_torn_event_line_is_skipped(self) -> None:
        job = self._create()
        with open(self.store.job_dir(job["id"]) / "events.jsonl", "a") as f:
            f.write('{"torn": ')
        self.assertEqual(len(self.store.events(job["id"])), 1)

    def test_reconcile_fails_what_a_stopped_trainer_left(self) -> None:
        running = self._create()
        self.store.update(running["id"], status="running")
        done = self._create()
        self.store.update(done["id"], status="succeeded")
        self.assertEqual(self.store.reconcile(), [running["id"]])
        failed = self.store.read(running["id"])
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error"]["code"], "trainer_stopped")
        self.assertIsNotNone(failed["finished_at"])
        self.assertEqual(self.store.read(done["id"])["status"], "succeeded")

    def test_job_ids_cannot_address_other_paths(self) -> None:
        with self.assertRaises(ValueError):
            self.store.job_dir("../../etc")
        self.assertIsNone(self.store.read("../../etc/passwd"))


class ContractTest(unittest.TestCase):

    def test_canonical_hash_is_compact_json_in_given_order(self) -> None:
        import hashlib
        self.assertEqual(canonical_hash([1, 2, 3]), hashlib.sha256(b"[1,2,3]").hexdigest())
        self.assertNotEqual(canonical_hash({"a": 1, "b": 2}), canonical_hash({"b": 2, "a": 1}))

    def test_error_codes(self) -> None:
        self.assertEqual(error_code(ExitCode.DATASET), "dataset_contract_violation")
        self.assertEqual(error_code(4), "base_checkpoint_mismatch")
        self.assertEqual(error_code(99), "exit_99")

    def test_sha256sums_round_trip(self) -> None:
        d = tmpdir(self)
        (d / "a.txt").write_text("x")
        (d / "b.txt").write_text("y")
        write_sha256sums(d, ["a.txt", "b.txt"])
        self.assertEqual(set(read_sha256sums(d / "SHA256SUMS")), {"a.txt", "b.txt"})


if __name__ == "__main__":
    unittest.main()
