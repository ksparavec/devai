package control

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// Memory reads the card's used memory (C1, "Memory readings"). Present is
// false on a host without a GPU, where every memory check is skipped.
type Memory interface {
	Present() bool
	UsedMiB(ctx context.Context) (int, error)
}

type noGPU struct{}

func (noGPU) Present() bool                        { return false }
func (noGPU) UsedMiB(context.Context) (int, error) { return 0, nil }

// nvidiaSMI reads `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`;
// CDI puts nvidia-smi into every container that gets the GPU.
type nvidiaSMI struct{ path string }

func (n nvidiaSMI) Present() bool { return true }

func (n nvidiaSMI) UsedMiB(ctx context.Context) (int, error) {
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, n.path, "--query-gpu=memory.used", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return 0, fmt.Errorf("nvidia-smi: %w", err)
	}
	first := strings.TrimSpace(strings.SplitN(string(out), "\n", 2)[0])
	v, err := strconv.Atoi(first)
	if err != nil {
		return 0, fmt.Errorf("nvidia-smi: unreadable %q", first)
	}
	return v, nil
}

// NewMemory picks the reader for gpu: auto (nvidia-smi on PATH, else none)
// or none.
func NewMemory(gpu string) Memory {
	if gpu == "none" {
		return noGPU{}
	}
	if p, err := exec.LookPath("nvidia-smi"); err == nil {
		return nvidiaSMI{path: p}
	}
	return noGPU{}
}
