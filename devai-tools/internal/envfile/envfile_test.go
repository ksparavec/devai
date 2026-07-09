package envfile

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func tempEnvFile(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), ".env")
	if content != "" {
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return path
}

func TestSetKeysOnAbsentFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), ".env")
	err := SetKeys(path, []string{"A", "B"}, map[string]string{"A": "1", "B": "2"})
	if err != nil {
		t.Fatalf("SetKeys: %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	want := "A=1\nB=2\n"
	if string(got) != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestSetKeysAddWhenAbsent(t *testing.T) {
	path := tempEnvFile(t, "LAB_PORT=8888\n# a comment\nWEBUI_PORT=8443\n")
	err := SetKeys(path, []string{"DEVAI_GPU_VENDOR", "DEVAI_GPU_DEVICE"}, map[string]string{
		"DEVAI_GPU_VENDOR": "nvidia",
		"DEVAI_GPU_DEVICE": "nvidia.com/gpu=all",
	})
	if err != nil {
		t.Fatalf("SetKeys: %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	want := "LAB_PORT=8888\n# a comment\nWEBUI_PORT=8443\nDEVAI_GPU_VENDOR=nvidia\nDEVAI_GPU_DEVICE=nvidia.com/gpu=all\n"
	if string(got) != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestSetKeysReplaceWhenPresentPreservesOtherLines(t *testing.T) {
	path := tempEnvFile(t, "LAB_PORT=8888\nDEVAI_GPU_DEVICE=nvidia.com/gpu=all\n# a comment about GPUs\nWEBUI_PORT=8443\n")
	err := SetKeys(path, []string{"DEVAI_GPU_DEVICE"}, map[string]string{"DEVAI_GPU_DEVICE": "amd.com/gpu=all"})
	if err != nil {
		t.Fatalf("SetKeys: %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	want := "LAB_PORT=8888\nDEVAI_GPU_DEVICE=amd.com/gpu=all\n# a comment about GPUs\nWEBUI_PORT=8443\n"
	if string(got) != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestSetKeysRoundTripFlipAndFlipBack(t *testing.T) {
	original := "LAB_PORT=8888\nCONTAINER_RUNTIME=podman\n"
	path := tempEnvFile(t, original)

	nvidia := map[string]string{
		"DEVAI_GPU_VENDOR": "nvidia",
		"DEVAI_GPU_DEVICE": "nvidia.com/gpu=all",
		"VLLM_IMAGE":       "docker.io/vllm/vllm-openai:v0.22.1-x86_64-cu129-ubuntu2404",
		"SGLANG_IMAGE":     "docker.io/lmsysorg/sglang:v0.5.10.post1-cu130",
	}
	amd := map[string]string{
		"DEVAI_GPU_VENDOR": "amd",
		"DEVAI_GPU_DEVICE": "amd.com/gpu=all",
		"VLLM_IMAGE":       "docker.io/vllm/vllm-openai-rocm:latest",
		"SGLANG_IMAGE":     "docker.io/lmsysorg/sglang:latest-rocm",
	}
	order := []string{"DEVAI_GPU_VENDOR", "DEVAI_GPU_DEVICE", "VLLM_IMAGE", "SGLANG_IMAGE"}

	if err := SetKeys(path, order, nvidia); err != nil {
		t.Fatalf("SetKeys(nvidia): %v", err)
	}
	if err := SetKeys(path, order, amd); err != nil {
		t.Fatalf("SetKeys(amd): %v", err)
	}
	afterAMD, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for k, v := range amd {
		if !strings.Contains(string(afterAMD), k+"="+v) {
			t.Errorf("after flip to amd, missing %s=%s in:\n%s", k, v, afterAMD)
		}
	}

	if err := SetKeys(path, order, nvidia); err != nil {
		t.Fatalf("SetKeys(nvidia again): %v", err)
	}
	afterFlipBack, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	want := original + "DEVAI_GPU_VENDOR=nvidia\nDEVAI_GPU_DEVICE=nvidia.com/gpu=all\nVLLM_IMAGE=docker.io/vllm/vllm-openai:v0.22.1-x86_64-cu129-ubuntu2404\nSGLANG_IMAGE=docker.io/lmsysorg/sglang:v0.5.10.post1-cu130\n"
	if string(afterFlipBack) != want {
		t.Errorf("after flip back to nvidia, got:\n%s\nwant:\n%s", afterFlipBack, want)
	}
}
