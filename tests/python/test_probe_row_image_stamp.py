"""Per-row engine-image stamping in the HF probe caches.

Why this exists: `_meta.current_image_digest` is CACHE-WIDE. A partial
re-probe -- which is the normal case, since only models whose weights are
still on disk can be re-probed -- moves that pointer to the new image while
leaving every un-reprobed row exactly as it was. Without a per-row stamp
those rows become indistinguishable from freshly measured ones, and
`make probe-check` (which compares only `_meta` against the running image)
then reports no drift at all.

Concretely: bumping SGLang v0.5.10.post1 -> v0.5.16 with 5 of 21 rows
re-probeable would have re-stamped `_meta` to v0.5.16 and silently
presented 16 rows of v0.5.10 measurements as current.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _probe_core import (  # noqa: E402
    ROW_IMAGE_FIELD,
    backfill_row_images,
    image_stamp_survey,
    row_image_is_stale,
    stamp_image_digest,
    stamp_row_image,
)


class TestStampRowImage(unittest.TestCase):
    def test_stamps_digest_on_row(self) -> None:
        entry: dict = {"schema_version": 2}
        stamp_row_image(entry, "sha256:new")
        self.assertEqual(entry[ROW_IMAGE_FIELD], "sha256:new")

    def test_falsy_digest_is_noop(self) -> None:
        # A failed `podman image inspect` must not write a bogus stamp --
        # an absent stamp is honest ("unknown"), an empty one is a lie.
        entry: dict = {"schema_version": 2}
        stamp_row_image(entry, None)
        stamp_row_image(entry, "")
        self.assertNotIn(ROW_IMAGE_FIELD, entry)

    def test_overwrites_previous_stamp(self) -> None:
        # Re-probing a row under a new image replaces the stamp; there is
        # no history at row level (that lives in _meta.image_history).
        entry = {ROW_IMAGE_FIELD: "sha256:old"}
        stamp_row_image(entry, "sha256:new")
        self.assertEqual(entry[ROW_IMAGE_FIELD], "sha256:new")


class TestImageStampSurvey(unittest.TestCase):
    def _cache(self) -> dict:
        cache = {
            "fresh/model@aaa": {ROW_IMAGE_FIELD: "sha256:new"},
            "stale/model@bbb": {ROW_IMAGE_FIELD: "sha256:old"},
            "legacy/model@ccc": {},
        }
        stamp_image_digest(cache, digest="sha256:new", image_ref="img:new")
        return cache

    def test_partitions_rows_by_stamp(self) -> None:
        survey = image_stamp_survey(self._cache(), "sha256:new")
        self.assertEqual(survey["current"], ["fresh/model@aaa"])
        self.assertEqual(survey["stale"], ["stale/model@bbb"])
        self.assertEqual(survey["unstamped"], ["legacy/model@ccc"])

    def test_skips_meta_block(self) -> None:
        # `_meta` is not a model row; every reader skips `_`-prefixed keys.
        survey = image_stamp_survey(self._cache(), "sha256:new")
        allrows = survey["current"] + survey["stale"] + survey["unstamped"]
        self.assertNotIn("_meta", allrows)

    def test_no_current_digest_cannot_judge(self) -> None:
        # Without a baseline nothing is stale -- guessing would either force
        # a needless full re-probe or hide real drift. Mirrors the same
        # decision in bench-sync's classify().
        survey = image_stamp_survey(self._cache(), None)
        self.assertEqual(survey["stale"], [])
        self.assertEqual(survey["current"], [])
        self.assertEqual(len(survey["unstamped"]), 3)

    def test_the_partial_reprobe_scenario(self) -> None:
        """The exact failure this stamp exists to prevent."""
        # 3 rows measured under the old image...
        cache: dict = {f"m{i}@sha": {} for i in range(3)}
        for entry in cache.values():
            stamp_row_image(entry, "sha256:v0510")
        stamp_image_digest(cache, digest="sha256:v0510", image_ref="sglang:v0.5.10")

        # ...one of which gets re-probed on the new image.
        stamp_row_image(cache["m0@sha"], "sha256:v0516")
        stamp_image_digest(cache, digest="sha256:v0516", image_ref="sglang:v0.5.16")

        # _meta alone now says "current" -- which is why it is not enough.
        self.assertEqual(cache["_meta"]["current_image_digest"], "sha256:v0516")

        survey = image_stamp_survey(cache, "sha256:v0516")
        self.assertEqual(survey["current"], ["m0@sha"])
        self.assertEqual(sorted(survey["stale"]), ["m1@sha", "m2@sha"])


class TestBackfillRowImages(unittest.TestCase):
    def test_attributes_legacy_rows_to_current_meta(self) -> None:
        cache: dict = {"legacy@a": {}, "legacy@b": {}}
        stamp_image_digest(cache, digest="sha256:v0510", image_ref="sglang:v0.5.10")
        self.assertEqual(backfill_row_images(cache), 2)
        self.assertEqual(cache["legacy@a"][ROW_IMAGE_FIELD], "sha256:v0510")
        self.assertEqual(cache["legacy@b"][ROW_IMAGE_FIELD], "sha256:v0510")

    def test_idempotent_and_preserves_existing_stamps(self) -> None:
        cache = {"already@a": {ROW_IMAGE_FIELD: "sha256:other"}, "legacy@b": {}}
        stamp_image_digest(cache, digest="sha256:v0510", image_ref="img")
        self.assertEqual(backfill_row_images(cache), 1)
        # An existing stamp is authoritative and must not be rewritten.
        self.assertEqual(cache["already@a"][ROW_IMAGE_FIELD], "sha256:other")
        # Second call changes nothing.
        self.assertEqual(backfill_row_images(cache), 0)

    def test_noop_without_meta_or_digest(self) -> None:
        # Nothing to attribute to: leave rows honestly unstamped.
        cache: dict = {"row@a": {}}
        self.assertEqual(backfill_row_images(cache), 0)
        self.assertNotIn(ROW_IMAGE_FIELD, cache["row@a"])
        cache2: dict = {"row@a": {}, "_meta": {"image_history": {}}}
        self.assertEqual(backfill_row_images(cache2), 0)
        self.assertNotIn(ROW_IMAGE_FIELD, cache2["row@a"])

    def test_backfill_then_partial_reprobe_reports_correctly(self) -> None:
        """End-to-end: the sequence a real image bump actually runs."""
        cache: dict = {f"m{i}@sha": {} for i in range(3)}
        stamp_image_digest(cache, digest="sha256:v0510", image_ref="sglang:v0.5.10")

        # Bump: backfill legacy rows, THEN move the pointer.
        backfill_row_images(cache)
        stamp_image_digest(cache, digest="sha256:v0516", image_ref="sglang:v0.5.16")
        # Only m0 has weights on disk, so only m0 gets re-probed.
        stamp_row_image(cache["m0@sha"], "sha256:v0516")

        survey = image_stamp_survey(cache, "sha256:v0516")
        self.assertEqual(survey["current"], ["m0@sha"])
        self.assertEqual(sorted(survey["stale"]), ["m1@sha", "m2@sha"])
        self.assertEqual(survey["unstamped"], [])


class TestRowImageIsStale(unittest.TestCase):
    """Auto-invalidation predicate for the probers.

    Demonstrated need: after the v0.5.10.post1 -> v0.5.16 bump,
    `make probe-sglang` ran ZERO probes -- every on-disk model reported
    "band fully cached" and Gemma-4 was skipped as cached
    `unsupported_arch`, even though that verdict was an artefact of the
    old engine and v0.5.16 loads the model. The bump silently rescued
    nothing until this predicate existed.
    """

    def test_different_digest_is_stale(self) -> None:
        self.assertTrue(
            row_image_is_stale({ROW_IMAGE_FIELD: "sha256:old"}, "sha256:new"))

    def test_same_digest_is_fresh(self) -> None:
        self.assertFalse(
            row_image_is_stale({ROW_IMAGE_FIELD: "sha256:new"}, "sha256:new"))

    def test_unknown_either_side_is_not_stale(self) -> None:
        # Nothing to compare -> do not invalidate. Guessing here would
        # re-probe the whole fleet whenever an image inspect failed.
        self.assertFalse(row_image_is_stale({}, "sha256:new"))
        self.assertFalse(row_image_is_stale({ROW_IMAGE_FIELD: "sha256:old"}, None))
        self.assertFalse(row_image_is_stale({}, None))

    def test_dry_run_never_invalidates(self) -> None:
        # --no-cache-write cannot persist a result, so invalidating would
        # re-probe on every dry run and write nothing.
        self.assertFalse(row_image_is_stale(
            {ROW_IMAGE_FIELD: "sha256:old"}, "sha256:new", writable=False))


if __name__ == "__main__":
    unittest.main()
