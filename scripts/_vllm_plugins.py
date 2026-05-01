"""vLLM parser-plugin registry — Python side.

Reads ``deploy/vllm-plugins.json`` (the single source of truth, also
consumed by ``gpu-arbiter/main.go``) and exposes simple lookup helpers
for the probe driver. The registry maps a parser name (the value passed
to ``--tool-call-parser`` / ``--reasoning-parser``) to the plugin file
that registers it. Names absent from the registry are treated as
built-in vLLM parsers and pass through unchanged.

Adding a new plugin: drop the ``.py`` file in ``scripts/vllm_plugins/``
and add one entry in ``deploy/vllm-plugins.json``. Nothing else needs
to change in this module.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_PATH = REPO_ROOT / "deploy" / "vllm-plugins.json"
DEFAULT_HOST_PLUGIN_DIR = REPO_ROOT / "scripts" / "vllm_plugins"
DEFAULT_CONTAINER_DIR = "/etc/devai/vllm-plugins"


@dataclass(frozen=True)
class PluginEntry:
    name: str
    kind: str  # "tool" | "reasoning"
    file: str  # basename inside the plugins dir

    def container_path(self, container_dir: str) -> str:
        return f"{container_dir.rstrip('/')}/{self.file}"


@dataclass(frozen=True)
class PluginRegistry:
    container_dir: str
    host_dir: Path
    entries: dict[str, PluginEntry]

    def lookup(self, parser_name: str | None) -> PluginEntry | None:
        if not parser_name:
            return None
        return self.entries.get(parser_name)

    def is_plugin(self, parser_name: str | None) -> bool:
        return self.lookup(parser_name) is not None


_CACHED_REGISTRY: PluginRegistry | None = None


def get_registry(
    path: Path | None = None,
    *,
    host_dir: Path | None = None,
    refresh: bool = False,
) -> PluginRegistry:
    """Module-level cache. Within a probe pass the registry file does
    not change; loading it once avoids re-parsing on every cell.
    """
    global _CACHED_REGISTRY
    if refresh or _CACHED_REGISTRY is None:
        _CACHED_REGISTRY = load_registry(path, host_dir=host_dir)
    return _CACHED_REGISTRY


def load_registry(
    path: Path | None = None,
    *,
    host_dir: Path | None = None,
) -> PluginRegistry:
    """Load the JSON registry. Missing file or malformed JSON yields an
    empty registry — every parser name then routes through the built-in
    path (no plugin flag, no bind-mount). This keeps the probe and
    router behaviour identical to pre-plugin builds when the registry
    is absent.
    """
    p = path or DEFAULT_REGISTRY_PATH
    container_dir = DEFAULT_CONTAINER_DIR
    entries: dict[str, PluginEntry] = {}
    try:
        raw = json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        raw = {}
    if isinstance(raw, dict):
        cd = raw.get("container_dir")
        if isinstance(cd, str) and cd.strip():
            container_dir = cd.strip()
        plugins = raw.get("plugins")
        if isinstance(plugins, dict):
            for name, meta in plugins.items():
                if not isinstance(name, str) or not isinstance(meta, dict):
                    continue
                kind = meta.get("kind")
                fname = meta.get("file")
                if kind not in ("tool", "reasoning"):
                    continue
                if not isinstance(fname, str) or not fname.strip():
                    continue
                entries[name] = PluginEntry(
                    name=name, kind=kind, file=fname.strip()
                )
    resolved_host_dir = host_dir or Path(
        os.environ.get("VLLM_PLUGINS_HOST_DIR")
        or DEFAULT_HOST_PLUGIN_DIR
    )
    return PluginRegistry(
        container_dir=container_dir,
        host_dir=resolved_host_dir,
        entries=entries,
    )
