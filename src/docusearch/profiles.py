"""Source profiles: additive enrichment layered on top of the generic parsers.

The parser is still chosen by file extension (``ingest.extract_document`` — the format seam).
A *profile* runs afterwards and only ADDS structured fields it recognises in a particular
document family, so the generic HTML/PDF/DOCX parsing stays one implementation that every
profile benefits from. Nothing here re-parses a document; a profile that recognises nothing
is a no-op and ingest behaves exactly as before.

Scope rules live on the DOCUMENT, not the chunk. "Instrument: UltraPin1600" or "V11.00.00
and later" at the top of a page governs the whole page, so all of that page's chunks inherit
it by membership — a hit on chunk 15 is scoped exactly like a hit on chunk 1. Only a signal
that is genuinely local to one passage (``restricts``) is returned per chunk.

Adding a family: write a :class:`Profile`, register it in :data:`PROFILES`, and set
``profile: <name>`` on the source in ``docusearch.yaml``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocFacts:
    """What a profile recognised. Empty fields mean "not stated", never "false"."""

    instrument: str = ""      # comma-joined, normalised instrument names
    applies_from: str = ""    # earliest release a page's rules apply to, e.g. "11.00.00"
    restricts: list[int] = field(default_factory=list)  # indexes of chunks stating a limit


class Profile:
    """Recognise structured facts a generic parser cannot know are meaningful."""

    name = "generic"

    def facts(self, title: str, chunk_texts: Sequence[str]) -> DocFacts:  # pragma: no cover
        return DocFacts()


# A limit is only worth flagging when it is stated as a rule, not merely narrated. Kept
# deliberately tight: "cannot be merged" is a rule, "you cannot see the grid" is prose.
_RESTRICT_RE = re.compile(
    r"\b(?:cannot|can not|must not|may not|is not allowed|not permitted|"
    r"not supported|not available|no longer supported|unsupported)\b",
    re.I,
)

# Instrument names are a closed vocabulary in this doc set. A whitelist (rather than "any
# word after Instrument:") is what keeps "Support" and "HPM" out — those came from headings
# running into the label during an earlier scan of the real catalog.
_IGXL_INSTRUMENTS = (
    "HSD1000", "HSDM", "HSDP", "UltraPin800", "UltraPin1600", "UltraPin2200", "UltraPin4000",
    "UltraVS256", "UltraSerial10G", "SB6G", "DC07", "VHFAC", "UltraFLEX", "UltraFLEXplus",
)
_INSTRUMENT_LINE_RE = re.compile(r"Instrument\s*:\s*([^\n]{0,160})", re.I)

# "V11.00.00 and later", "IG-XL 11.00 or later" -> the release a rule starts applying at.
# Only forward-looking phrasings count: a bare version mention is a reference, not a bound.
_APPLIES_FROM_RE = re.compile(
    r"(?:IG-?XL\s*)?V?(\d{2}\.\d{2}(?:\.\d{2})?)\s+(?:and|or)\s+(?:later|newer|above)", re.I
)


class IgxlProfile(Profile):
    """Teradyne IG-XL help set.

    Three things in these pages are scope, not prose, and flattening them is what makes an
    answer wrong: which instrument a rule governs, which release it starts at, and whether a
    passage states a limit at all. A rule for the UltraPin2200 applied to an UltraPin1600 is
    not a small error — it is the wrong answer stated confidently.
    """

    name = "igxl"

    def facts(self, title: str, chunk_texts: Sequence[str]) -> DocFacts:
        head = " ".join(" ".join(str(t).split()) for t in chunk_texts[:3])[:1200]

        found: list[str] = []
        for m in _INSTRUMENT_LINE_RE.finditer(head):
            for name in _IGXL_INSTRUMENTS:
                if re.search(rf"\b{re.escape(name)}\b", m.group(1), re.I) and name not in found:
                    found.append(name)

        whole = " ".join(" ".join(str(t).split()) for t in chunk_texts)[:20000]
        versions = sorted({m.group(1) for m in _APPLIES_FROM_RE.finditer(whole)})

        restricts = [i for i, t in enumerate(chunk_texts) if _RESTRICT_RE.search(str(t))]
        return DocFacts(
            instrument=", ".join(found),
            applies_from=versions[0] if versions else "",
            restricts=restricts,
        )


PROFILES: dict[str, Profile] = {p.name: p for p in (IgxlProfile(),)}


def get(name: str) -> Profile | None:
    """The profile for a source, or None for generic parsing (an unknown name is ignored)."""
    return PROFILES.get((name or "").strip().lower()) or None
