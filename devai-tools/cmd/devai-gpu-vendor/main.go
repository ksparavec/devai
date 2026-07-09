// Command devai-gpu-vendor flips the GPU-vendor knob in a devai .env
// file: DEVAI_GPU_VENDOR plus the 3 vars it derives (DEVAI_GPU_DEVICE,
// VLLM_IMAGE, SGLANG_IMAGE). See docs/gpu-vendors.md.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/sparavec/devai-tools/internal/envfile"
)

// orderedKeys is the write order for a fresh .env with none of these keys
// yet -- keeps a flip's diff readable (vendor first, then its 3 derived
// vars in the same order they're documented in .env.example).
var orderedKeys = []string{"DEVAI_GPU_VENDOR", "DEVAI_GPU_DEVICE", "VLLM_IMAGE", "SGLANG_IMAGE"}

// vendorValues are hardcoded constants, not fetched -- re-verify the AMD
// image tags against Docker Hub before relying on them; ROCm image
// releases move faster than this table gets updated. NVIDIA's values
// match deploy/docker-compose.yaml's own fallback defaults.
var vendorValues = map[string]map[string]string{
	"nvidia": {
		"DEVAI_GPU_VENDOR": "nvidia",
		"DEVAI_GPU_DEVICE": "nvidia.com/gpu=all",
		"VLLM_IMAGE":       "docker.io/vllm/vllm-openai:v0.22.1-x86_64-cu129-ubuntu2404",
		"SGLANG_IMAGE":     "docker.io/lmsysorg/sglang:v0.5.10.post1-cu130",
	},
	"amd": {
		"DEVAI_GPU_VENDOR": "amd",
		"DEVAI_GPU_DEVICE": "amd.com/gpu=all",
		// Placeholder tags -- verify the current vllm-openai-rocm / sglang
		// ROCm-tagged release on Docker Hub before real use.
		"VLLM_IMAGE":   "docker.io/vllm/vllm-openai-rocm:latest",
		"SGLANG_IMAGE": "docker.io/lmsysorg/sglang:latest-rocm",
	},
}

func main() {
	envFilePath := flag.String("env-file", ".env", "path to the .env file to update")
	vendor := flag.String("vendor", "", "nvidia or amd (required)")
	flag.Parse()

	values, ok := vendorValues[*vendor]
	if !ok {
		fmt.Fprintf(os.Stderr, "error: --vendor must be \"nvidia\" or \"amd\", got %q\n", *vendor)
		os.Exit(2)
	}

	if err := envfile.SetKeys(*envFilePath, orderedKeys, values); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}

	fmt.Printf("%s: DEVAI_GPU_VENDOR=%s DEVAI_GPU_DEVICE=%s VLLM_IMAGE=%s SGLANG_IMAGE=%s\n",
		*envFilePath, values["DEVAI_GPU_VENDOR"], values["DEVAI_GPU_DEVICE"], values["VLLM_IMAGE"], values["SGLANG_IMAGE"])
}
