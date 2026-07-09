"""Phase C: image-digest drift stamping in the shared probe cache helpers.

Covers scripts/_probe_core.stamp_image_digest (writes the `_meta` block the
router reads to detect a moved backend image) and image_digest_via_cli's
fail-open contract. The stamping mirrors bench/_bench_core.stamp_host_env --
an `image_history` map keyed by digest plus a `current_image_digest` pointer.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _probe_core import (  # noqa: E402
    image_digest_via_cli,
    stamp_image_digest,
)


class TestStampImageDigest(unittest.TestCase):
    def test_creates_meta_block_and_pointer(self) -> None:
        cache: dict = {}
        stamp_image_digest(
            cache, digest="sha256:aaa", image_ref="docker.io/vllm/vllm-openai:v0.22.1")
        self.assertEqual(cache["_meta"]["current_image_digest"], "sha256:aaa")
        self.assertEqual(
            cache["_meta"]["current_image_ref"], "docker.io/vllm/vllm-openai:v0.22.1")
        self.assertIn("sha256:aaa", cache["_meta"]["image_history"])
        entry = cache["_meta"]["image_history"]["sha256:aaa"]
        self.assertEqual(entry["image_ref"], "docker.io/vllm/vllm-openai:v0.22.1")
        self.assertIn("first_seen", entry)

    def test_does_not_disturb_model_rows(self) -> None:
        cache = {"repo@sha": {"schema_version": 2, "fits": True}}
        stamp_image_digest(cache, digest="sha256:aaa", image_ref="img:1")
        self.assertEqual(cache["repo@sha"], {"schema_version": 2, "fits": True})
        self.assertIn("_meta", cache)

    def test_second_digest_accumulates_history_and_moves_pointer(self) -> None:
        cache: dict = {}
        stamp_image_digest(cache, digest="sha256:old", image_ref="img:old")
        first_seen_old = cache["_meta"]["image_history"]["sha256:old"]["first_seen"]
        stamp_image_digest(cache, digest="sha256:new", image_ref="img:new")
        # Pointer moves; both digests retained in history.
        self.assertEqual(cache["_meta"]["current_image_digest"], "sha256:new")
        self.assertIn("sha256:old", cache["_meta"]["image_history"])
        self.assertIn("sha256:new", cache["_meta"]["image_history"])
        # Re-stamping the old digest must not overwrite its first_seen.
        stamp_image_digest(cache, digest="sha256:old", image_ref="img:old")
        self.assertEqual(
            cache["_meta"]["image_history"]["sha256:old"]["first_seen"], first_seen_old)
        self.assertEqual(cache["_meta"]["current_image_digest"], "sha256:old")

    def test_falsy_digest_is_noop(self) -> None:
        # A failed image inspect (None/"") must never corrupt _meta.
        cache: dict = {}
        stamp_image_digest(cache, digest=None, image_ref="img:1")  # type: ignore[arg-type]
        stamp_image_digest(cache, digest="", image_ref="img:1")
        self.assertNotIn("_meta", cache)


class TestImageDigestViaCli(unittest.TestCase):
    def test_missing_runtime_fails_open(self) -> None:
        # A nonexistent runtime binary must return None, not raise -- drift
        # detection degrades to "no baseline" rather than aborting a probe.
        self.assertIsNone(
            image_digest_via_cli("definitely-not-a-real-runtime-xyz", "img:1"))


if __name__ == "__main__":
    unittest.main()
