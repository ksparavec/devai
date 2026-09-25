"""The dataset contract: import, checksums, manifest, base binding, rows.

The rules mirror aiagent's own validator (docs/laya-trainer.md, dataset contract).
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import unittest
from pathlib import Path

from ..contract import ExitCode, JobError, write_sha256sums
from .util import FixtureStore, needs_laya, tmpdir


@needs_laya
class DatasetTest(unittest.TestCase):

    def setUp(self) -> None:
        from .. import dataset
        self.dataset = dataset
        self.fx = FixtureStore(tmpdir(self))
        self.ds_id = self.fx.dataset()

    @property
    def inbox(self) -> Path:
        return self.fx.store / "inbox" / self.ds_id

    def edit(self, name: str, fn) -> None:
        """Edit one file and re-seal the dataset as aiagent would: the files
        table, SHA256SUMS, and the id (which names the manifest's hash)."""
        d = self.inbox
        if name.endswith(".jsonl"):
            p = d / name
            rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            fn(rows)
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
            m = json.loads((d / "manifest.json").read_text())
            m["files"][name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                "rows": m["files"][name]["rows"]}
        else:
            m = json.loads((d / "manifest.json").read_text())
            fn(m)
        body = json.dumps(m, indent=2).encode()
        (d / "manifest.json").write_bytes(body)
        write_sha256sums(d, [p.name for p in d.iterdir() if p.name != "SHA256SUMS"])
        new_id = "ds-" + hashlib.sha256(body).hexdigest()[:12]
        d.rename(d.with_name(new_id))
        self.ds_id = new_id

    def assertViolation(self, fn, fragment: str, code: ExitCode = ExitCode.DATASET) -> None:
        with self.assertRaises(JobError) as cm:
            fn()
        self.assertEqual(cm.exception.code, code, cm.exception.message)
        self.assertIn(fragment, cm.exception.message)

    def _validate(self) -> dict:
        d = self.dataset.import_dataset(self.fx.store, self.ds_id)
        m = self.dataset.load_manifest(d)
        self.dataset.check_base(m, self.fx.row, self.fx.base_dir)
        return self.dataset.encode_dataset(d, m, self.fx.tok)

    # -- import ----------------------------------------------------------------

    def test_import_copies_and_seals_the_dataset(self) -> None:
        d = self.dataset.import_dataset(self.fx.store, self.ds_id)
        self.assertEqual(d, self.fx.store / "datasets" / self.ds_id)
        self.assertEqual(sorted(p.name for p in d.iterdir()),
                         sorted(p.name for p in self.inbox.iterdir()))
        self.assertEqual(stat.S_IMODE(d.stat().st_mode) & 0o222, 0)

    def test_import_is_idempotent(self) -> None:
        a = self.dataset.import_dataset(self.fx.store, self.ds_id)
        self.assertEqual(self.dataset.import_dataset(self.fx.store, self.ds_id), a)

    def test_a_reused_id_with_other_contents_is_refused(self) -> None:
        self.dataset.import_dataset(self.fx.store, self.ds_id)
        with open(self.inbox / "pool.jsonl", "a") as f:
            f.write("\n")
        write_sha256sums(self.inbox, [p.name for p in self.inbox.iterdir() if p.name != "SHA256SUMS"])
        self.assertViolation(lambda: self.dataset.import_dataset(self.fx.store, self.ds_id),
                             "different contents")

    def test_the_id_must_name_the_manifest(self) -> None:
        self.inbox.rename(self.inbox.with_name("ds-000000000000"))
        self.ds_id = "ds-000000000000"
        self.assertViolation(lambda: self.dataset.import_dataset(self.fx.store, self.ds_id),
                             "does not name its manifest")

    def test_a_checksum_mismatch_is_refused(self) -> None:
        with open(self.inbox / "train.jsonl", "a") as f:
            f.write("\n")
        self.assertViolation(lambda: self.dataset.import_dataset(self.fx.store, self.ds_id), "sha256")

    def test_unknown_files_are_refused(self) -> None:
        (self.inbox / "notes.txt").write_text("hi")
        self.assertViolation(lambda: self.dataset.import_dataset(self.fx.store, self.ds_id),
                             "unexpected files")

    def test_a_symlinked_file_is_refused(self) -> None:
        target = self.fx.store / "elsewhere.jsonl"
        (self.inbox / "train.jsonl").rename(target)
        os.symlink(target, self.inbox / "train.jsonl")
        self.assertViolation(lambda: self.dataset.import_dataset(self.fx.store, self.ds_id),
                             "not regular files")

    def test_a_bad_id_is_refused(self) -> None:
        self.assertViolation(lambda: self.dataset.import_dataset(self.fx.store, "../base"),
                             "not a dataset id")

    # -- manifest and base -----------------------------------------------------

    def test_an_unknown_schema_version_is_refused(self) -> None:
        self.edit("manifest.json", lambda m: m.update(schema_version=2))
        self.assertViolation(self._validate, "schema_version")

    def test_the_files_table_must_match(self) -> None:
        self.edit("manifest.json", lambda m: m["files"]["train.jsonl"].update(sha256="0" * 64))
        self.assertViolation(self._validate, "files['train.jsonl']")

    def test_file_row_counts_must_match(self) -> None:
        self.edit("manifest.json", lambda m: m["files"]["train.jsonl"].update(rows=99))
        self.assertViolation(self._validate, "files[train.jsonl].rows")

    def test_split_counts_must_match(self) -> None:
        self.edit("manifest.json", lambda m: m["splits"]["counts"].update(train=99))
        self.assertViolation(self._validate, "manifest 99 != rows 24")

    def test_a_dataset_for_another_base_is_exit_4(self) -> None:
        self.edit("manifest.json", lambda m: m["base_checkpoint"].update(weights_sha256="0" * 64))
        self.assertViolation(self._validate, "weights_sha256", ExitCode.BASE)

    def test_a_tampered_base_is_exit_4(self) -> None:
        from .. import catalog as cat
        cat.make_writable(self.fx.base_dir)
        with open(self.fx.base_dir / "rl_agent_config.json", "a") as f:
            f.write(" ")
        self.assertViolation(self._validate, "rl_agent_config.json", ExitCode.BASE)

    # -- rows ------------------------------------------------------------------

    def test_a_valid_dataset_encodes_every_labeled_question(self) -> None:
        splits = self._validate()
        self.assertEqual({s: len(v) for s, v in splits.items()},
                         {"train": 24 * 3, "calib": 30 * 3, "heldout": 12 * 3, "pool": 0})
        item = splits["train"][0]
        self.assertEqual(len(item["target"]), len(item["markers"]))
        self.assertAlmostEqual(sum(item["target"]), 1.0)

    def test_choice_targets_follow_criteria_order(self) -> None:
        item = next(i for i in self._validate()["train"] if i["qid"] == "polarity")
        self.assertEqual(item["keys"], ["negative", "mixed", "positive"])

    def test_token_hash_mismatch_is_caught(self) -> None:
        def edit(rows):
            rows[0]["student_tokens"]["polarity"]["ids_sha256"] = "0" * 64
        self.edit("train.jsonl", edit)
        self.assertViolation(self._validate, "tokenized differently")

    def test_pool_token_hashes_are_checked_too(self) -> None:
        def edit(rows):
            rows[0]["student_tokens"]["stars"]["n"] += 1
        self.edit("pool.jsonl", edit)
        self.assertViolation(self._validate, "tokenized differently")

    def test_a_pool_row_carries_no_gold(self) -> None:
        def edit(rows):
            rows[0]["gold"] = {"polarity": {"label": "mixed", "probabilities": {}}}
        self.edit("pool.jsonl", edit)
        self.assertViolation(self._validate, "a pool row carries no gold")

    def test_a_labeled_row_needs_a_teacher(self) -> None:
        self.edit("calib.jsonl", lambda rows: rows[0].pop("teacher"))
        self.assertViolation(self._validate, "needs a teacher")

    def test_null_is_a_violation_outside_questions(self) -> None:
        def edit(rows):
            rows[0]["teacher"]["k"] = None
        self.edit("train.jsonl", edit)
        self.assertViolation(self._validate, "null at teacher.k")

    def test_null_criterion_descriptions_are_legitimate(self) -> None:
        from .. import fixture
        fx = FixtureStore(tmpdir(self))
        questions = {"polarity": {**fixture.QUESTIONS["polarity"],
                                  "criteria": {"negative": None, "mixed": None, "positive": None}}}
        ds = fx.dataset(questions=questions)
        d = self.dataset.import_dataset(fx.store, ds)
        items = self.dataset.encode_dataset(d, self.dataset.load_manifest(d), fx.tok)
        self.assertEqual(len(items["train"]), 24)

    def test_probability_keys_must_be_in_option_order(self) -> None:
        def edit(rows):
            p = rows[0]["gold"]["polarity"]["probabilities"]
            rows[0]["gold"]["polarity"]["probabilities"] = dict(reversed(list(p.items())))
        self.edit("train.jsonl", edit)
        self.assertViolation(self._validate, "in order")

    def test_probabilities_must_sum_to_one(self) -> None:
        def edit(rows):
            rows[0]["gold"]["polarity"]["probabilities"]["mixed"] = 0.5
        self.edit("train.jsonl", edit)
        self.assertViolation(self._validate, "sum to")

    def test_the_label_must_be_the_first_argmax(self) -> None:
        def edit(rows):
            g = rows[0]["gold"]["positive"]
            g["probabilities"] = {"false": 0.5, "true": 0.5}
            g["label"] = "true"
        self.edit("train.jsonl", edit)
        self.assertViolation(self._validate, "first argmax")

    def test_no_synthetic_rows_in_calib(self) -> None:
        self.edit("calib.jsonl", lambda rows: rows[0].update(synthetic=True))
        self.assertViolation(self._validate, "synthetic rows are not allowed in calib")

    def test_calib_must_not_share_a_document_with_train(self) -> None:
        train = [json.loads(line) for line in (self.inbox / "train.jsonl").read_text().splitlines()]
        self.edit("calib.jsonl", lambda rows: rows[0].update(group_id=train[0]["group_id"]))
        self.assertViolation(self._validate, "share 1 group_id")

    def test_a_truncated_state_is_refused(self) -> None:
        fx = FixtureStore(tmpdir(self))
        ds = fx.dataset(max_len=32, head_max_len=16)  # no room left for the state
        d = self.dataset.import_dataset(fx.store, ds)
        with self.assertRaises(JobError) as cm:
            self.dataset.encode_dataset(d, self.dataset.load_manifest(d), fx.tok)
        self.assertIn("never be truncated", cm.exception.message)

    def test_a_row_in_the_wrong_file_is_refused(self) -> None:
        self.edit("train.jsonl", lambda rows: rows[0].update(split="calib"))
        self.assertViolation(self._validate, "does not match the file")

    def test_duplicate_row_ids_are_refused(self) -> None:
        self.edit("train.jsonl", lambda rows: rows[1].update(id=rows[0]["id"]))
        self.assertViolation(self._validate, "duplicate row id")

    def test_gold_must_cover_every_question(self) -> None:
        self.edit("train.jsonl", lambda rows: rows[0]["gold"].pop("stars"))
        self.assertViolation(self._validate, "gold keys")


if __name__ == "__main__":
    unittest.main()
