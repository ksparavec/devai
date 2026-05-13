"""Capability classification vocabulary shared by probers, router, picker.

Source of truth for every "structured"/"inline"/... string previously
spread as magic literals across the probers and consumers. StrEnum
(Python 3.11+; this project uses 3.13) makes each member a real `str`,
so `json.dumps`, dict-key lookups, and `==` against legacy literals
continue to work unchanged. Wire format on disk is identical -- the
on-disk JSON still contains lowercase strings like "structured".

The Go router mirrors this list in gpu-arbiter/main.go under a `Cap*`
const block. Keep the two in sync when adding or removing values.
"""

from __future__ import annotations

from enum import StrEnum


class Capability(StrEnum):
    STRUCTURED       = "structured"        # backend has a working --reasoning-parser
    INLINE           = "inline"            # model emits <think> blocks into content
    UNSUPPORTED      = "unsupported"       # no reasoning capability detected
    NONE             = "none"              # clean non-reasoning model (probe gave clean answer)
    UNSUPPORTED_ARCH = "unsupported_arch"  # TERMINAL: backend can't load this arch
    ERROR            = "error"             # TERMINAL: probe failed
    UNKNOWN          = "unknown"           # initial / pre-probe state


# Frozenset of values whose row should be hidden from the picker and
# refused by the router. Centralised so future terminal states (or a
# narrowing of the set) propagate to every reader at once.
TERMINAL: frozenset[Capability] = frozenset({Capability.ERROR, Capability.UNSUPPORTED_ARCH})


def is_terminal(value: "str | Capability | None") -> bool:
    """True when ``value`` should be treated as TERMINAL.

    Accepts plain strings for use by callers that haven't been migrated
    to the enum yet (e.g. reading directly from cache JSON). ``None``
    is treated as non-terminal -- absent capability means "not yet
    probed", not "failed".
    """
    if value is None:
        return False
    return value in TERMINAL
