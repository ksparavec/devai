"""`make model-prepare NAME=<row>`: the checkpoint preparation, made repeatable.

Two checkpoints were prepared by hand on 2026-09-21 (HyperQwen's CPU scripts
run inside the devai-vllm image, then scripts/mixed_quant_groups.py for the
NVFP4 one). The commands lived only in the session that ran them, and the
store's copy of each checkpoint was rewritten with nothing recording when,
by what, or from which originals -- at odds with the repo's own rule that the
model stores are changed only by scripts that enforce the paths and record
outcomes. scripts/prepare-checkpoint.py is that script.

What is pinned here:
  * it refuses to run on a directory that is missing, already prepared, or
    not a compressed-tensors checkpoint (the vendored scripts write that
    format and no other), and without the patched image (stock vLLM cannot
    load a quantized embedding for this architecture);
  * the step sequence per shard layout, in order, with the exact arguments;
  * mixed_quant_groups.py runs only when the body is not int-quantized;
  * a step failure stops the run and writes no manifest;
  * the PREPARED.json manifest records the vendored-script commit, every
    step, and sha256 + size of every rewritten or added file before and
    after -- what "repeat exactly" needs.

The container runtime is faked; nothing is quantized here. The scripts'
own behaviour is upstream's (third_party/hyperqwen/).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "prepare_checkpoint", REPO_ROOT / "scripts" / "prepare-checkpoint.py")
pc = importlib.util.module_from_spec(_SPEC)
sys.modules["prepare_checkpoint"] = pc
_SPEC.loader.exec_module(pc)

MAKEFILE = (REPO_ROOT / "Makefile").read_text()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class _Fixture:
    """A fake checkpoint directory and a fake runner that imitates what the
    upstream scripts do to the files (rewrite + leave backups + new files)."""

    def __init__(self, root: Path, *, shards: int, body_type: str):
        self.dir = root / "M"
        self.dir.mkdir()
        wm = {}
        if shards == 1:
            (self.dir / "model.safetensors").write_bytes(b"body+heads" * 100)
            wm = {"lm_head.weight": "model.safetensors",
                  "model.language_model.embed_tokens.weight": "model.safetensors",
                  "model.layers.0.mlp.up_proj.weight": "model.safetensors"}
            (self.dir / "model-mtp-bf16.safetensors").write_bytes(b"mtp" * 100)
            wm["mtp.fc.weight"] = "model-mtp-bf16.safetensors"
        else:
            for i in range(1, shards + 1):
                (self.dir / f"model-{i:05d}-of-{shards:05d}.safetensors").write_bytes(bytes([i]) * 1000)
            wm = {"lm_head.weight": f"model-{shards:05d}-of-{shards:05d}.safetensors",
                  "model.language_model.embed_tokens.weight": f"model-{shards-1:05d}-of-{shards:05d}.safetensors",
                  "model.layers.0.mlp.up_proj.weight": f"model-00001-of-{shards:05d}.safetensors"}
            (self.dir / "model_extra_tensors.safetensors").write_bytes(b"extra" * 100)
            wm["mtp.fc.weight"] = "model_extra_tensors.safetensors"
        (self.dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
        body = {"format": "pack-quantized" if body_type == "int" else "nvfp4-pack-quantized",
                "input_activations": None if body_type == "int" else {"num_bits": 4, "type": "float"},
                "output_activations": None, "targets": ["Linear"],
                "weights": {"num_bits": 4, "type": body_type, "strategy": "group" if body_type == "int" else "tensor_group",
                            "group_size": 128 if body_type == "int" else 16, "symmetric": True}}
        (self.dir / "config.json").write_text(json.dumps({
            "quantization_config": {"quant_method": "compressed-tensors",
                                    "format": body["format"], "ignore": ["lm_head"],
                                    "config_groups": {"group_0": body}}}))
        self.calls: list[list[str]] = []
        self.fail_on: str | None = None

    def runner(self, argv: list[str]) -> int:
        self.calls.append(list(argv))
        script = next((a for a in argv if a.startswith("prepare/")), "")
        if self.fail_on and self.fail_on in script:
            return 1
        # Act on whatever is mounted at /model, as the real scripts do: in
        # derive mode that is the derived copy, never the fixture's source.
        mount = next(a for a in argv if a.endswith(":/model:rw"))
        d = Path(mount[: -len(":/model:rw")])
        wm = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
        cfg = json.loads((d / "config.json").read_text())

        def quantise(keys, backup_suffix, rename=False):
            for key in keys:
                shard = wm.pop(key, None)
                if shard is None:
                    continue
                src = d / shard
                if rename:
                    if not (d / (shard + backup_suffix)).exists():
                        src.rename(d / (shard + backup_suffix))
                else:
                    (d / (shard + backup_suffix)).write_bytes(src.read_bytes())
                src.write_bytes(b"int8:" + key.encode())
                for s in ("weight_packed", "weight_scale", "weight_shape"):
                    wm[key.replace(".weight", "." + s)] = shard

        def add_group(name, targets):
            g = json.loads(json.dumps(cfg["quantization_config"]["config_groups"]["group_0"]))
            g["targets"] = targets
            g["weights"]["num_bits"] = 8
            cfg["quantization_config"]["config_groups"][name] = g
            cfg["quantization_config"]["ignore"] = [
                i for i in cfg["quantization_config"]["ignore"] if i != "lm_head"]

        def backup_metadata(suffix):
            # quant_lm_head.py / quant_heads_stream.py copy config + index as
            # .bak-quant before touching them; quant_mtp.py as .bak-mtp.
            for f in ("config.json", "model.safetensors.index.json"):
                if not (d / (f + suffix)).exists():
                    (d / (f + suffix)).write_bytes((d / f).read_bytes())

        if "quant_lm_head.py" in script or "quant_heads_stream.py" in script:
            backup_metadata(".bak-quant")
        elif "quant_mtp.py" in script:
            backup_metadata(".bak-mtp")

        if "quant_lm_head.py" in script:
            quantise(["lm_head.weight"], ".bak"); add_group("group_1", ["re:.*lm_head$"])
        elif "quant_embed.py" in script:
            quantise(["model.language_model.embed_tokens.weight"], ".bak_embed"); add_group("group_2", ["re:.*embed_tokens$"])
        elif "quant_mtp.py" in script:
            quantise(["mtp.fc.weight"], ".bak-mtp"); add_group("group_3", ["re:^mtp\\..*"])
        elif "quant_heads_stream.py" in script:
            quantise(["lm_head.weight", "model.language_model.embed_tokens.weight", "mtp.fc.weight"], ".bak-orig", rename=True)
            add_group("group_1", ["re:.*lm_head$"]); add_group("group_2", ["re:.*embed_tokens$"]); add_group("group_3", ["re:^mtp\\..*"])
        elif "build_draft_vocab.py" in script:
            extra = d / "model_extra_tensors.safetensors"
            if extra.exists():
                (d / "model_extra_tensors.safetensors.bak-draft").write_bytes(extra.read_bytes())
            extra.write_bytes(b"draft-head")
            wm["mtp.draft_lm_head.weight_packed"] = "model_extra_tensors.safetensors"
            (d / "mtp_draft_vocab_ids.pt").write_bytes(b"ids")
        (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
        (d / "config.json").write_text(json.dumps(cfg))
        return 0


class PrepareCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _run(self, fx: _Fixture, *extra: str, image_present: bool = True) -> int:
        return pc.main(["--dir", str(fx.dir), "--name", "M", *extra],
                       runner=fx.runner, image_exists=lambda _img: image_present)

    # ── refusals ─────────────────────────────────────────────────────────
    def test_missing_directory_is_an_error(self) -> None:
        rc = pc.main(["--dir", str(self.root / "nope"), "--name", "M"],
                     runner=lambda a: 0, image_exists=lambda _i: True)
        self.assertNotEqual(rc, 0)

    def test_already_prepared_is_refused(self) -> None:
        fx = _Fixture(self.root, shards=7, body_type="int")
        (fx.dir / "PREPARED.json").write_text("{}")
        self.assertNotEqual(self._run(fx), 0)
        self.assertEqual(fx.calls, [], "must not touch a prepared checkpoint")

    def test_partially_prepared_is_refused_too(self) -> None:
        # A .bak file with no manifest: an interrupted or hand-run
        # preparation. Running the scripts again would double-quantize.
        fx = _Fixture(self.root, shards=7, body_type="int")
        (fx.dir / "config.json.bak-quant").write_text("{}")
        self.assertNotEqual(self._run(fx), 0)
        self.assertEqual(fx.calls, [])

    def test_non_compressed_tensors_checkpoint_is_refused(self) -> None:
        fx = _Fixture(self.root, shards=3, body_type="int")
        cfg = json.loads((fx.dir / "config.json").read_text())
        cfg["quantization_config"]["quant_method"] = "modelopt"
        (fx.dir / "config.json").write_text(json.dumps(cfg))
        self.assertNotEqual(self._run(fx), 0)
        self.assertEqual(fx.calls, [])

    def test_missing_patched_image_is_refused(self) -> None:
        fx = _Fixture(self.root, shards=7, body_type="int")
        self.assertNotEqual(self._run(fx, image_present=False), 0)
        self.assertEqual(fx.calls, [])

    # ── the sequences ────────────────────────────────────────────────────
    def test_sharded_int4_checkpoint_runs_upstreams_four_step_recipe(self) -> None:
        fx = _Fixture(self.root, shards=7, body_type="int")
        self.assertEqual(self._run(fx), 0)
        scripts = [next(a for a in c if a.startswith("prepare/")) for c in fx.calls]
        self.assertEqual(scripts, ["prepare/quant_lm_head.py", "prepare/quant_embed.py",
                                   "prepare/quant_mtp.py", "prepare/build_draft_vocab.py"])
        self.assertIn("--ids", fx.calls[-1])
        self.assertIn("prepare/draft_vocab_ids.json", fx.calls[-1])
        self.assertFalse((fx.dir / "config.json.bak-mixed").exists(),
                         "an int body needs no group repair")

    def test_single_shard_checkpoint_uses_the_streaming_script(self) -> None:
        fx = _Fixture(self.root, shards=1, body_type="float")
        self.assertEqual(self._run(fx), 0)
        scripts = [next(a for a in c if a.startswith("prepare/")) for c in fx.calls]
        self.assertEqual(scripts, ["prepare/quant_heads_stream.py", "prepare/build_draft_vocab.py"])

    def test_nvfp4_body_gets_the_group_repair_after_the_scripts(self) -> None:
        fx = _Fixture(self.root, shards=1, body_type="float")
        self.assertEqual(self._run(fx), 0)
        cfg = json.loads((fx.dir / "config.json").read_text())["quantization_config"]
        self.assertEqual(cfg["config_groups"]["group_0"]["format"], "nvfp4-pack-quantized")
        for g in ("group_1", "group_2", "group_3"):
            self.assertEqual(cfg["config_groups"][g]["format"], "pack-quantized", g)
            self.assertIsNone(cfg["config_groups"][g]["input_activations"], g)
        self.assertTrue((fx.dir / "config.json.bak-mixed").exists())

    def test_every_step_runs_inside_the_patched_image_against_the_vendored_scripts(self) -> None:
        self.assertEqual(pc.IMAGE, "docker.io/devai/vllm-devai:latest")
        fx = _Fixture(self.root, shards=7, body_type="int")
        self._run(fx)
        for argv in fx.calls:
            self.assertEqual(argv[0], "podman")
            self.assertIn(pc.IMAGE, argv)
            self.assertIn(f"{fx.dir}:/model:rw", argv)
            self.assertIn(f"{pc.VENDOR_DIR}:/hq:ro", argv)
            self.assertIn("/model", argv)

    # ── failure ──────────────────────────────────────────────────────────
    def test_a_failing_step_stops_the_run_and_writes_no_manifest(self) -> None:
        fx = _Fixture(self.root, shards=7, body_type="int")
        fx.fail_on = "quant_embed.py"
        self.assertNotEqual(self._run(fx), 0)
        self.assertEqual(len(fx.calls), 2)
        self.assertFalse((fx.dir / "PREPARED.json").exists())

    # ── the manifest ─────────────────────────────────────────────────────
    def test_manifest_records_what_is_needed_to_repeat_it(self) -> None:
        fx = _Fixture(self.root, shards=7, body_type="int")
        before = {f.name: (_sha(f), f.stat().st_size) for f in fx.dir.iterdir()}
        self.assertEqual(self._run(fx), 0)
        m = json.loads((fx.dir / "PREPARED.json").read_text())
        self.assertEqual(m["name"], "M")
        self.assertEqual(m["method"], "sharded")
        self.assertEqual(m["hyperqwen"]["commit"], pc.VENDOR_COMMIT)
        self.assertEqual(m["hyperqwen"]["license"], "Apache-2.0")
        self.assertEqual([s["script"] for s in m["steps"]],
                         ["prepare/quant_lm_head.py", "prepare/quant_embed.py",
                          "prepare/quant_mtp.py", "prepare/build_draft_vocab.py"])
        self.assertTrue(all(s["rc"] == 0 for s in m["steps"]))
        # rewritten files carry before AND after; new files only after.
        lm = m["files"]["model-00007-of-00007.safetensors"]
        self.assertEqual(lm["sha256_before"], before["model-00007-of-00007.safetensors"][0])
        self.assertEqual(lm["sha256_after"], _sha(fx.dir / "model-00007-of-00007.safetensors"))
        self.assertNotEqual(lm["sha256_before"], lm["sha256_after"])
        self.assertEqual(lm["backup"], "model-00007-of-00007.safetensors.bak")
        self.assertIsNone(m["files"]["mtp_draft_vocab_ids.pt"]["sha256_before"])
        self.assertNotIn("model-00001-of-00007.safetensors", m["files"], "untouched files are not listed")
        self.assertEqual(m["mixed_quant_groups"], False)

    def test_manifest_flags_the_group_repair_for_a_float_body(self) -> None:
        fx = _Fixture(self.root, shards=1, body_type="float")
        self._run(fx)
        m = json.loads((fx.dir / "PREPARED.json").read_text())
        self.assertEqual(m["method"], "stream")
        self.assertEqual(m["mixed_quant_groups"], True)
        self.assertEqual(m["files"]["model.safetensors"]["backup"], "model.safetensors.bak-orig")

    # ── derive: the source stays as downloaded ───────────────────────────
    def _derive(self, fx: _Fixture, *extra: str, catalog: dict | None = None) -> int:
        catalog = catalog if catalog is not None else {
            "models": [{"name": "M", "source": "hf"},
                       {"name": "M-devai", "source": "derived", "derived_from": "M"}]}
        cat = self.root / "models.yaml"
        cat.write_text(json.dumps(catalog))   # JSON is YAML
        return pc.main(["--dir", str(fx.dir), "--name", "M", "--derive", "--catalog", str(cat), *extra],
                       runner=fx.runner, image_exists=lambda _i: True)

    def test_derive_prepares_a_new_directory_and_leaves_the_source_untouched(self) -> None:
        fx = _Fixture(self.root, shards=1, body_type="float")
        before = {f.name: _sha(f) for f in fx.dir.iterdir() if f.is_file()}
        self.assertEqual(self._derive(fx), 0)
        after = {f.name: _sha(f) for f in fx.dir.iterdir() if f.is_file()}
        self.assertEqual(before, after, "the source directory must be byte-identical")
        self.assertFalse((fx.dir / "PREPARED.json").exists())
        d = self.root / "M-devai"
        self.assertTrue((d / "PREPARED.json").exists())
        self.assertTrue((d / "mtp_draft_vocab_ids.pt").exists())
        self.assertFalse(any(".bak" in p.name for p in d.iterdir()),
                         "no backups in a derived directory: the source holds the originals")
        self.assertFalse((d / ".cache").exists(), "download metadata is not copied")
        # every step ran against the DERIVED directory
        for argv in fx.calls:
            self.assertIn(f"{d}:/model:rw", argv)

    def test_derive_manifest_names_the_source_and_pairs_files_with_it(self) -> None:
        fx = _Fixture(self.root, shards=1, body_type="float")
        self._derive(fx)
        m = json.loads((self.root / "M-devai" / "PREPARED.json").read_text())
        self.assertEqual(m["name"], "M-devai")
        self.assertEqual(m["derived_from"], "M")
        self.assertEqual(m["source_dir"], str(fx.dir))
        f = m["files"]["model.safetensors"]
        self.assertEqual(f["sha256_before"], _sha(fx.dir / "model.safetensors"))
        self.assertIsNone(f["backup"])
        self.assertEqual(f["original_in"], str(fx.dir))

    def test_derive_refuses_a_source_that_is_not_as_downloaded(self) -> None:
        fx = _Fixture(self.root, shards=1, body_type="float")
        (fx.dir / "config.json.bak-quant").write_text("{}")
        self.assertNotEqual(self._derive(fx), 0)
        self.assertFalse((self.root / "M-devai").exists())
        self.assertEqual(fx.calls, [])

    def test_derive_refuses_when_the_derived_directory_exists(self) -> None:
        fx = _Fixture(self.root, shards=1, body_type="float")
        (self.root / "M-devai").mkdir()
        self.assertNotEqual(self._derive(fx), 0)
        self.assertEqual(fx.calls, [])

    def test_derive_needs_a_derived_row_in_the_catalog(self) -> None:
        fx = _Fixture(self.root, shards=1, body_type="float")
        rc = self._derive(fx, catalog={"models": [{"name": "M", "source": "hf"}]})
        self.assertNotEqual(rc, 0)
        self.assertEqual(fx.calls, [])

    def test_reconstruct_for_a_derived_directory_takes_before_from_the_source(self) -> None:
        fx = _Fixture(self.root, shards=1, body_type="float")
        self.assertEqual(self._derive(fx), 0)
        d = self.root / "M-devai"
        (d / "PREPARED.json").unlink()
        rc = pc.main(["--dir", str(d), "--name", "M-devai", "--reconstruct", "--method", "stream",
                      "--source-dir", str(fx.dir)], runner=fx.runner, image_exists=lambda _i: True)
        self.assertEqual(rc, 0)
        m = json.loads((d / "PREPARED.json").read_text())
        self.assertTrue(m["reconstructed"])
        self.assertEqual(m["derived_from"], "M")
        self.assertEqual(m["files"]["model.safetensors"]["sha256_before"], _sha(fx.dir / "model.safetensors"))
        self.assertIsNone(m["files"]["mtp_draft_vocab_ids.pt"]["sha256_before"])
        self.assertNotIn("chat_template.jinja", m["files"], "untouched files are not listed")

    # ── reconstruction for the two checkpoints prepared by hand ──────────
    def test_reconstruct_writes_a_manifest_from_the_backups_without_running_anything(self) -> None:
        fx = _Fixture(self.root, shards=1, body_type="float")
        self.assertEqual(self._run(fx), 0)
        (fx.dir / "PREPARED.json").unlink()
        calls_before = len(fx.calls)
        rc = pc.main(["--dir", str(fx.dir), "--name", "M", "--reconstruct", "--method", "stream"],
                     runner=fx.runner, image_exists=lambda _i: True)
        self.assertEqual(rc, 0)
        self.assertEqual(len(fx.calls), calls_before, "reconstruct must not run any step")
        m = json.loads((fx.dir / "PREPARED.json").read_text())
        self.assertTrue(m["reconstructed"])
        self.assertEqual(m["method"], "stream")
        self.assertEqual(m["files"]["model.safetensors"]["sha256_before"],
                         _sha(fx.dir / "model.safetensors.bak-orig"))
        self.assertEqual(m["files"]["model.safetensors"]["sha256_after"],
                         _sha(fx.dir / "model.safetensors"))

    def test_reconstruct_lists_only_touched_files_and_pairs_each_with_its_original(self) -> None:
        fx = _Fixture(self.root, shards=7, body_type="int")
        self.assertEqual(self._run(fx), 0)
        live = json.loads((fx.dir / "PREPARED.json").read_text())
        (fx.dir / "PREPARED.json").unlink()
        pc.main(["--dir", str(fx.dir), "--name", "M", "--reconstruct", "--method", "sharded"],
                runner=fx.runner, image_exists=lambda _i: True)
        m = json.loads((fx.dir / "PREPARED.json").read_text())
        self.assertEqual(set(m["files"]), set(live["files"]),
                         "reconstruction must name the same files the live run did")
        # config.json was backed up twice (quant_lm_head: .bak-quant, quant_mtp:
        # .bak-mtp); the ORIGINAL is the earlier one.
        self.assertEqual(m["files"]["config.json"]["backup"], "config.json.bak-quant")
        self.assertEqual(m["files"]["config.json"]["sha256_before"],
                         live["files"]["config.json"]["sha256_before"])
        self.assertEqual(m["files"]["model_extra_tensors.safetensors"]["backup"],
                         "model_extra_tensors.safetensors.bak-mtp")

    def test_reconstruct_refuses_an_unprepared_directory(self) -> None:
        fx = _Fixture(self.root, shards=7, body_type="int")
        rc = pc.main(["--dir", str(fx.dir), "--name", "M", "--reconstruct", "--method", "sharded"],
                     runner=fx.runner, image_exists=lambda _i: True)
        self.assertNotEqual(rc, 0)
        self.assertFalse((fx.dir / "PREPARED.json").exists())


class VendorAndWiringTest(unittest.TestCase):
    def test_vendored_scripts_match_their_manifest(self) -> None:
        d = REPO_ROOT / "third_party" / "hyperqwen"
        for line in (d / "MANIFEST.sha256").read_text().splitlines():
            digest, name = line.split(None, 1)
            self.assertEqual(_sha(d / name.strip()), digest, name)

    def test_vendor_readme_states_origin_commit_and_licence(self) -> None:
        readme = (REPO_ROOT / "third_party" / "hyperqwen" / "README.md").read_text()
        self.assertIn("https://github.com/syv-ai/HyperQwen", readme)
        self.assertIn(pc.VENDOR_COMMIT, readme)
        self.assertIn("Apache License 2.0", readme)
        self.assertTrue((REPO_ROOT / "third_party" / "hyperqwen" / "LICENSE").exists())
        self.assertTrue(all(ord(c) < 128 for c in readme), "markdown must be ASCII")

    def test_vendor_commit_matches_the_build_scripts_pin(self) -> None:
        build = (REPO_ROOT / "scripts" / "build-vllm.sh").read_text()
        self.assertIn(f'HYPERQWEN_COMMIT="${{HYPERQWEN_COMMIT:-{pc.VENDOR_COMMIT}}}"', build)

    def test_make_target_exists_and_forwards_name(self) -> None:
        recipe = MAKEFILE.split("\nmodel-prepare:", 1)[1].split("\n\n", 1)[0]
        self.assertIn("scripts/prepare-checkpoint.py", recipe)
        self.assertIn("--name", recipe)
        self.assertIn("$(NAME)", recipe)


if __name__ == "__main__":
    unittest.main()
