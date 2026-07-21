"""Source profiles: additive, per-family enrichment layered on the generic parsers.

The parser is still chosen by file extension (``ingest.extract_document`` — the format seam).
A *profile* runs afterwards and only ADDS structured fields it recognises in a particular
document family, so the generic HTML/PDF/DOCX parsing stays one implementation every profile
benefits from. A profile that recognises nothing is a no-op and ingest behaves as before.

Scope rules live on the DOCUMENT, not the chunk. "Instrument: UltraPin1600" or "V11.00.00 and
later" at the top of a page governs the whole page, so all of that page's chunks inherit it by
membership. Only a signal genuinely local to one passage (``restricts``) is returned per chunk.

**This package is the generic seam and is the only tracked file here.** Concrete profiles are
DEPLOYMENT-SPECIFIC (a customer's instrument rules are not part of the shipped product), so
``profiles/*`` is gitignored except this ``__init__``. Drop a ``profiles/<name>.py`` that calls
:func:`register` next to it and it is discovered automatically at import; remove it and the
product forgets that family with no code change. Track one deliberately with a ``!`` rule in
``.gitignore``.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

#: (mcp, service, base_url, config) -> None. Typed loosely so the seam does not import server.
MCPRegistrar = Callable[[object, object, str, object], None]


@dataclass(frozen=True)
class DocFacts:
    """What a profile recognised. Empty fields mean "not stated", never "false"."""

    instrument: str = ""  # comma-joined, normalised instrument names
    applies_from: str = ""  # earliest release a page's rules apply to, e.g. "11.00.00"
    excludes: str = ""  # instruments this page states do NOT support its subject
    restricts: list[int] = field(default_factory=list)  # indexes of chunks stating a limit


class Profile:
    """Recognise structured facts a generic parser cannot know are meaningful."""

    name = "generic"

    def facts(self, title: str, chunk_texts: Sequence[str]) -> DocFacts:  # pragma: no cover
        return DocFacts()


PROFILES: dict[str, Profile] = {}


def register(profile: Profile) -> None:
    """Add a profile to the registry. A profile module calls this at import."""
    PROFILES[profile.name] = profile


def get(name: str) -> Profile | None:
    """The profile for a source, or None for generic parsing (an unknown name is ignored)."""
    return PROFILES.get((name or "").strip().lower()) or None


#: MCP tool registrars contributed by profiles. A profile that adds an MCP tool (e.g. a
#: domain verifier) appends one here at import; ``build_mcp`` calls each so the generic server
#: carries no profile-specific tool of its own. Empty in a product with no profile installed.
MCP_REGISTRARS: list[MCPRegistrar] = []


def register_mcp(fn: MCPRegistrar) -> None:
    """A profile module calls this to contribute MCP tools (given mcp, service, base_url, config)."""
    MCP_REGISTRARS.append(fn)


def apply_mcp(mcp: object, service: object, base_url: str, config: object) -> None:
    """Let every installed profile register its MCP tools on ``mcp``."""
    for fn in MCP_REGISTRARS:
        fn(mcp, service, base_url, config)


def _discover() -> None:
    """Import every sibling module so each self-registers. A missing family is simply absent —
    the generic product ships with no concrete profile and this loop finds nothing."""
    for mod in pkgutil.iter_modules(__path__):
        if not mod.name.startswith("_"):
            importlib.import_module(f"{__name__}.{mod.name}")


_discover()
