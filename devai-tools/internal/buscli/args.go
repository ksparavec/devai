// Package buscli is devai-bus: the bus command-line client job scripts, the
// guard and the operator use (platform-rebuild C3; docs/slurm.md Sec. 4).
// cmd/devai-bus is a thin wrapper; the logic lives here so it can be tested
// in-process against an embedded nats-server.
package buscli

import (
	"fmt"
	"strings"
)

// Exit codes, the same for devai-bus and for every job script, which passes
// them through (C3).
const (
	ExitOK          = 0 // done; a job ended normally, also after its own cancel or stop
	ExitFailed      = 1 // refused or failed
	ExitUnreachable = 2 // the bus could not be reached (hold: only before its first state reply)
	ExitLost        = 3 // the service left its active state on its own
)

// Args holds a command line whose flags may come before, between or after
// its positional arguments (the job scripts write `request <subject> <json>
// --job N --timeout D`).
type Args struct {
	Pos   []string
	Vals  map[string]string
	Bools map[string]bool
}

// ParseArgs splits argv; valueFlags take a value (`--x v` or `--x=v`),
// boolFlags none. An unknown flag is an error.
func ParseArgs(argv []string, valueFlags, boolFlags []string) (*Args, error) {
	isVal := map[string]bool{}
	for _, f := range valueFlags {
		isVal[f] = true
	}
	isBool := map[string]bool{}
	for _, f := range boolFlags {
		isBool[f] = true
	}
	a := &Args{Vals: map[string]string{}, Bools: map[string]bool{}}
	for i := 0; i < len(argv); i++ {
		s := argv[i]
		if s == "--" {
			a.Pos = append(a.Pos, argv[i+1:]...)
			break
		}
		if !strings.HasPrefix(s, "--") || s == "-" {
			a.Pos = append(a.Pos, s)
			continue
		}
		name, val, hasVal := strings.Cut(strings.TrimPrefix(s, "--"), "=")
		switch {
		case isVal[name]:
			if !hasVal {
				if i+1 >= len(argv) {
					return nil, fmt.Errorf("--%s needs a value", name)
				}
				i++
				val = argv[i]
			}
			a.Vals[name] = val
		case isBool[name]:
			if hasVal {
				return nil, fmt.Errorf("--%s takes no value", name)
			}
			a.Bools[name] = true
		default:
			return nil, fmt.Errorf("unknown flag --%s", name)
		}
	}
	return a, nil
}
