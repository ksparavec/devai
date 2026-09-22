package main

// Persistent engine caches for the HF backends.
//
// vLLM and SGLang JIT-compile at first use: FlashInfer builds its
// attention kernels with nvcc under ~/.cache/flashinfer/<version>/, vLLM
// keeps torch.compile artifacts under VLLM_CACHE_ROOT (~/.cache/vllm),
// SGLang under ~/.cache/sglang. The router recreates a backend container
// (stop + rm + create) on every model or context switch, and until
// 2026-09-22 bound nothing under ~/.cache, so every launch re-paid the
// whole compile: 260-280 s cold start for Qwen3.8-27B-MTP-devai-NVFP4 on
// vllm-devai, measured that day (18:36:02 -> 18:40:31 = 269 s), most of
// it FlashInfer JIT plus torch.compile rather than the 19 GB weight load.
// With the volumes in place the same launch measured 275 s cold (empty
// volumes) and 51 s warm; the volumes held 28 MB and 188 MB.
//
// Each cache gets its own podman named volume, keyed by backend, so the
// stock and custom vLLM images never share one (their FlashInfer versions
// differ; FlashInfer namespaces its cache by version anyway, and vLLM's is
// hashed, so a stale entry is unused rather than wrong). The volumes are
// narrow -- one per cache directory, not all of ~/.cache -- because podman
// copies an image's existing directory into an empty named volume on
// first use, and the SGLang image carries 877 MB of Hugging Face cache
// there. Named volumes are created on demand by the create call and live
// in the podman graphroot; `make cache-down` leaves them alone. After an
// image bump, `podman volume rm devai-engine-cache-<backend>-*` reclaims
// the old entries (nothing depends on them being gone).
//
// The prober launches the same images through `podman run` and mounts the
// same volumes (scripts/_probe_hf_common.py, engine_cache_volumes), so a
// probe warms the cache a serve-time launch then reuses. The name format
// is pinned on both sides by tests/python/test_engine_cache_volumes.py.

const engineCacheVolumePrefix = "devai-engine-cache-"

// engineCacheVolumeName is the podman named volume holding one cache
// directory of one backend.
func engineCacheVolumeName(backend, cache string) string {
	return engineCacheVolumePrefix + backend + "-" + cache
}

// engineCacheVolumes returns the libpod NamedVolume specs for an HF
// backend: FlashInfer's cache and the engine's own. Ollama has none of
// these (GGUF, no JIT) and its model store is already a rw bind.
func engineCacheVolumes(backend string) []map[string]any {
	engine := engineOf(backend)
	if engine != "vllm" && engine != "sglang" {
		return nil
	}
	return []map[string]any{
		{"Name": engineCacheVolumeName(backend, "flashinfer"), "Dest": "/root/.cache/flashinfer"},
		{"Name": engineCacheVolumeName(backend, engine), "Dest": "/root/.cache/" + engine},
	}
}
