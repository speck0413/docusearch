"""Generated report files on the server (R-API-1).

A report is written to ``tmp_dir/reports/`` and handed back as a URL the requester can click,
because an MCP client may be nowhere near the machine that rendered it and a binary format
(docx/pptx/xlsx/pdf) has no sane JSON representation — base64 would put the whole artifact
through the model's context, which is what the compact search payload exists to avoid.

Retention follows ``reports.retain_days``: -1 keeps forever, 0 deletes at the next midnight, N
deletes at midnight N days after the day the file was written — midnight in the SERVER's local
timezone. The sweep runs when the server starts and after each write, so no scheduler is needed.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")

# A format whose filename cannot be derived from its name alone: (extension, name discriminator).
# `html-slide` is a deck rendered as one self-contained HTML file, so it must land as .html or a
# browser will not open it — but then it would collide with the plain html report of the same
# spec and silently overwrite it, so the stem carries a marker too.
_FILENAMES = {"html-slide": ("html", "-slides")}
_KEEP_FOREVER = -1


def slug(title: str, *, max_len: int = 60) -> str:
    """A filesystem-safe stem for a report title. ASCII-folded so a title with accents or CJK
    still yields a name every OS (and every browser download) handles identically."""
    folded = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    stem = _SLUG_STRIP.sub("-", folded.lower()).strip("-")[:max_len].strip("-")
    return stem or "report"


def filename(title: str, run_id: str, fmt: str) -> str:
    """The file a report of this title/run/format is written as.

    Two formats that share an extension must never share a name: rendering the same spec to both
    `html` and `html-slide` has to leave two files, not one silently overwritten."""
    ext, marker = _FILENAMES.get(fmt, (fmt, ""))
    return f"{slug(title)}-{run_id}{marker}.{ext}"


def reports_dir(tmp_dir: Path | str) -> Path:
    return Path(tmp_dir) / "reports"


def write(tmp_dir: Path | str, name: str, payload: bytes | str) -> Path:
    """Write one report and return its path. ``name`` is a bare filename — a caller-supplied
    name never escapes the reports directory (it is reduced to its final component)."""
    directory = reports_dir(tmp_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / Path(name).name
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_bytes(payload)
    return path


def resolve(tmp_dir: Path | str, name: str) -> Path | None:
    """The path to serve for ``name``, or None if it is missing or escapes the reports dir.

    Traversal defence in depth: the resolved path must sit inside ``reports/`` — the same
    pattern the images route uses, because this one serves caller-named files."""
    directory = reports_dir(tmp_dir).resolve()
    path = (directory / Path(name).name).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        return None
    return path


def expires_at(written: datetime, retain_days: int) -> datetime | None:
    """When a file written at ``written`` becomes deletable, or None if it is kept forever.

    The file always lives a **minimum** span first, then dies at the next midnight, so a report
    written just before midnight is never swept minutes later. ``0`` is a 12-hour floor; ``N`` is
    N x 24 hours. Both then round up to midnight in the server's local timezone."""
    if retain_days <= _KEEP_FOREVER:
        return None
    floor = timedelta(hours=12) if retain_days == 0 else timedelta(days=retain_days)
    earliest = written + floor
    midnight = datetime.combine(earliest.date(), datetime.min.time())
    return midnight if midnight == earliest else midnight + timedelta(days=1)


def sweep(tmp_dir: Path | str, retain_days: int, *, now: datetime | None = None) -> int:
    """Delete reports past their retention. Returns how many went.

    A file that cannot be read or removed (locked on Windows, permissions) is skipped rather
    than aborting the sweep — this runs on the serve path and must never take the server down.
    """
    if retain_days <= _KEEP_FOREVER:
        return 0
    directory = reports_dir(tmp_dir)
    if not directory.is_dir():
        return 0
    moment = now or datetime.now()
    removed = 0
    for path in sorted(directory.iterdir()):  # sorted => deterministic across platforms
        if not path.is_file():
            continue
        try:
            written = datetime.fromtimestamp(path.stat().st_mtime)
            deadline = expires_at(written, retain_days)
            if deadline is not None and moment >= deadline:
                path.unlink()
                removed += 1
        except OSError:  # locked/vanished — leave it for the next sweep
            continue
    return removed
