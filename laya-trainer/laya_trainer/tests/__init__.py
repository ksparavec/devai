"""laya trainer tests. Run inside the image: `make test-laya-trainer` (CPU, no network).

Tests that need torch/laya skip themselves where those are not installed, so the
standard-library parts (job store, controller, contract) also run on a host.
"""
