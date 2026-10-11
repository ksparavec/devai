// Command devai-control is the control service of devai-engines,
// devai-laya-trainer and devai-operator (docs/plans/platform-rebuild.md C1,
// C7, I33; docs/slurm.md Sec. 4). supervisord runs it as
//
//	devai-control --container <name> --config /etc/devai/control.yaml
//
// The flag must equal the file's `container:`; the GPU guard finds each
// container's devai-control by this command line (Phase 5).
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/sparavec/devai-tools/internal/control"
)

func main() {
	container := flag.String("container", "", "this container's name, as in control.yaml (required)")
	config := flag.String("config", "/etc/devai/control.yaml", "the control.yaml to read")
	flag.Parse()
	if *container == "" || flag.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "usage: devai-control --container <name> --config /etc/devai/control.yaml")
		os.Exit(2)
	}
	cfg, err := control.LoadConfig(*config, *container)
	if err != nil {
		fmt.Fprintln(os.Stderr, "devai-control:", err)
		os.Exit(1)
	}
	c, err := control.New(cfg, control.Options{})
	if err != nil {
		fmt.Fprintln(os.Stderr, "devai-control:", err)
		os.Exit(1)
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()
	if err := c.Run(ctx); err != nil {
		fmt.Fprintln(os.Stderr, "devai-control:", err)
		os.Exit(1)
	}
}
