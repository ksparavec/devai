package main

// vLLM parser-plugin registry — Go side.
//
// Reads deploy/vllm-plugins.json (the same file consumed by
// scripts/_vllm_plugins.py) and resolves parser names to in-container
// plugin file paths. When a model's tool_parser or reasoning_parser
// name matches a registry entry, the router (a) bind-mounts the
// host plugin directory into the recreated backend container at the
// registry's container_dir, and (b) prepends the corresponding
// `--*-parser-plugin <abs-path>` flag in front of the parser-name
// flag (vLLM resolves parser names at the point the parser-name flag
// is evaluated, so the plugin module must be loaded by then).
//
// Names absent from the registry are treated as built-in vLLM parsers
// and pass through unchanged — the router's behaviour for built-ins
// is identical to pre-plugin builds.
//
// Adding a new plugin: drop the .py file in scripts/vllm_plugins/ and
// add one entry in deploy/vllm-plugins.json. Nothing in this file
// needs to change.

import (
	"encoding/json"
	"log"
	"os"
	"strings"
)

const defaultPluginContainerDir = "/etc/devai/vllm-plugins"

type vllmPluginEntry struct {
	Kind string `json:"kind"` // "tool" | "reasoning"
	File string `json:"file"` // basename inside the plugins dir
}

type vllmPluginRegistry struct {
	ContainerDir string                     `json:"container_dir"`
	Plugins      map[string]vllmPluginEntry `json:"plugins"`
	// HostDir is populated from VLLM_PLUGINS_HOST_DIR (not from the JSON
	// file) so the same registry can be shared across hosts with
	// different repo locations. Empty when the env var is unset; in
	// that mode any model that requires a plugin will fail at
	// containerRecreate time with an actionable error rather than
	// silently launching without the plugin.
	HostDir string `json:"-"`
}

// loadVLLMPluginRegistry reads the JSON registry. A missing file or
// parse error logs a warning and returns an empty (non-nil) registry —
// every parser name then routes through the built-in path, matching
// pre-plugin behaviour.
func loadVLLMPluginRegistry(path, hostDir string) *vllmPluginRegistry {
	r := &vllmPluginRegistry{
		ContainerDir: defaultPluginContainerDir,
		Plugins:      map[string]vllmPluginEntry{},
		HostDir:      hostDir,
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			log.Printf("vllm plugin registry: %s not present — built-in parsers only", path)
		} else {
			log.Printf("warning: vllm plugin registry %s read failed: %v", path, err)
		}
		return r
	}
	// The registry tolerates an `_comment` field at the top level (used
	// for human-readable notes); only `container_dir` and `plugins` are
	// consumed.
	var raw struct {
		ContainerDir string                     `json:"container_dir"`
		Plugins      map[string]vllmPluginEntry `json:"plugins"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		log.Printf("warning: vllm plugin registry %s parse failed: %v", path, err)
		return r
	}
	if strings.TrimSpace(raw.ContainerDir) != "" {
		r.ContainerDir = strings.TrimSpace(raw.ContainerDir)
	}
	for name, entry := range raw.Plugins {
		if entry.Kind != "tool" && entry.Kind != "reasoning" {
			log.Printf("warning: vllm plugin %q has invalid kind=%q; skipping", name, entry.Kind)
			continue
		}
		if strings.TrimSpace(entry.File) == "" {
			log.Printf("warning: vllm plugin %q missing file; skipping", name)
			continue
		}
		r.Plugins[name] = entry
	}
	log.Printf("vllm plugin registry: %s loaded (%d entries; host_dir=%q)",
		path, len(r.Plugins), r.HostDir)
	return r
}

// Lookup returns the registered entry for parserName plus whether it
// matched. A nil receiver and an empty parserName both yield (_, false).
func (r *vllmPluginRegistry) Lookup(parserName string) (vllmPluginEntry, bool) {
	if r == nil || parserName == "" {
		return vllmPluginEntry{}, false
	}
	e, ok := r.Plugins[parserName]
	return e, ok
}

// ContainerPath joins the registry's container_dir with the plugin
// file's basename, producing the absolute path inside the backend
// container that gets passed to `--tool-parser-plugin` /
// `--reasoning-parser-plugin`.
func (r *vllmPluginRegistry) ContainerPath(file string) string {
	if r == nil {
		return ""
	}
	dir := strings.TrimRight(r.ContainerDir, "/")
	if dir == "" {
		dir = strings.TrimRight(defaultPluginContainerDir, "/")
	}
	return dir + "/" + file
}
