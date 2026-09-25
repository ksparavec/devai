"""devai's laya trainer: fine-tunes laya "System 1" students behind the router.

See docs/plans/laya-trainer.md. Kept import-light on purpose: scripts/
select-models.py imports `laya_trainer.catalog` on the host, where neither
torch nor laya is installed, so nothing here may import them at module level.
"""
