"""Acquire the public verification corpora (§15.1) — official downloads only.

This tool downloads real, comparably-large public documentation sets so the agent can
prove docusearch on data Stephen never has to share (R-TEST-5). It fetches published
archives / uses official tooling (rustup, git) — it NEVER crawls live sites. Every set
is recorded in ``corpora/MANIFEST.md`` with its URL, license, file count, and archive
sha256 (§15.1).

It is validation tooling, not part of the shipped package. Nothing here runs during
Phase 0; the Phase 1 online setup step invokes it, then everything else runs offline.

    python corpora/download.py --list
    python corpora/download.py --only php,python --dest corpora/data
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import ssl
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DEST = HERE / "data"
MANIFEST_PATH = HERE / "MANIFEST.md"


@dataclass(frozen=True)
class CorpusSpec:
    """One public corpus and how to obtain it through official channels."""

    name: str
    kind: str  # tarball | zip | rustup | git | manual
    license: str
    fmt: str  # html | md | mixed
    note: str
    url: str = ""
    command: tuple[str, ...] = field(default_factory=tuple)
    file_glob: str = "**/*.html"


# The default set targets >=20k documents combined (§15.1). Exact archive URLs move
# with upstream versions; confirm against the official download page before a run and
# let the manifest record whatever was actually fetched. (§18 Q7 — Stephen confirms.)
CORPORA: tuple[CorpusSpec, ...] = (
    CorpusSpec(
        name="php",
        kind="tarball",
        url="https://www.php.net/distributions/manual/php_manual_en.tar.gz",
        license="CC-BY 3.0",
        fmt="html",
        note="PHP manual 'Many HTML files' (~13k files). Primary vendor analog: "
        "framework chrome, dense cross-links, code everywhere.",
    ),
    CorpusSpec(
        name="python",
        kind="zip",
        url="https://docs.python.org/3/archives/python-3.13-docs-html.zip",
        license="PSF",
        fmt="html",
        note="Official python.org HTML docs zip. Volume top-up. Update the version "
        "in the URL to the current stable release.",
    ),
    CorpusSpec(
        name="rust",
        kind="rustup",
        command=("rustup", "component", "add", "rust-docs"),
        license="MIT OR Apache-2.0",
        fmt="html",
        note="rustup installs 20k+ local HTML files under "
        "$(rustc --print sysroot)/share/doc/rust/html. Size stressor.",
    ),
    CorpusSpec(
        name="mdn",
        kind="git",
        command=("git", "clone", "--depth", "1", "https://github.com/mdn/content"),
        license="CC-BY-SA 2.5+ (content) / MPL-2.0 (code)",
        fmt="md",
        note="Native Markdown corpus for the Phase 4c MD suite.",
        file_glob="**/*.md",
    ),
    CorpusSpec(
        name="wikipedia",
        kind="manual",
        license="CC-BY-SA",
        fmt="html",
        note="Only via official dumps or a Kiwix ZIM extract if the others are "
        "unreachable. Never mass-curl live Wikipedia. Not auto-downloaded.",
    ),
)


def _ssl_context() -> ssl.SSLContext:
    """A TLS context that trusts certifi's CA bundle when available."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:  # pragma: no cover - certifi is normally present
        return ssl.create_default_context()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _count_files(root: Path, glob: str) -> int:
    return sum(1 for p in root.glob(glob) if p.is_file())


@dataclass
class ManifestEntry:
    name: str
    license: str
    source: str
    files: int
    sha256: str
    status: str


def render_manifest(entries: Iterable[ManifestEntry]) -> str:
    """Render the corpora manifest as Markdown (§15.1)."""
    lines = [
        "# Public verification corpora — MANIFEST",
        "",
        "Recorded by `corpora/download.py` (§15.1). Official downloads only; live-site",
        "crawling is never used. Data lives under `corpora/data/` (git-ignored).",
        "",
        "| Corpus | License | Source | Files | SHA-256 (archive) | Status |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for e in entries:
        sha = e.sha256 if e.sha256 else "—"
        lines.append(f"| {e.name} | {e.license} | {e.source} | {e.files} | `{sha}` | {e.status} |")
    lines.append("")
    return "\n".join(lines)


def _skeleton_entries() -> list[ManifestEntry]:
    """Manifest rows before anything is downloaded (Phase 0 state)."""
    entries: list[ManifestEntry] = []
    for spec in CORPORA:
        source = spec.url or (" ".join(spec.command) if spec.command else "(manual)")
        entries.append(
            ManifestEntry(
                name=spec.name,
                license=spec.license,
                source=source,
                files=0,
                sha256="",
                status="not downloaded",
            )
        )
    return entries


def write_manifest(entries: Iterable[ManifestEntry], path: Path = MANIFEST_PATH) -> None:
    path.write_text(render_manifest(entries), encoding="utf-8")


# ------------------------------------------------------------------- fetchers


def _http_fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "docusearch-corpora/0.1"})
    with urllib.request.urlopen(req, context=_ssl_context()) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)
    return dest


def _extract(archive: Path, into: Path) -> None:
    into.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(into, filter="data")  # 'data' filter blocks path traversal
    elif archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(into)
    else:
        raise ValueError(f"unknown archive type: {archive.name}")


def fetch(spec: CorpusSpec, dest: Path) -> ManifestEntry:
    """Fetch one corpus and return its manifest row. Requires network / tooling."""
    target = dest / spec.name
    source = spec.url or " ".join(spec.command)
    if spec.kind in {"tarball", "zip"}:
        suffix = ".tar.gz" if spec.kind == "tarball" else ".zip"
        archive = dest / f"{spec.name}{suffix}"
        _http_fetch(spec.url, archive)
        sha = _sha256_file(archive)
        _extract(archive, target)
        return ManifestEntry(
            spec.name, spec.license, source, _count_files(target, spec.file_glob), sha, "ready"
        )
    if spec.kind in {"rustup", "git"}:
        subprocess.run(spec.command, check=True, cwd=str(dest))
        return ManifestEntry(
            spec.name, spec.license, source, _count_files(target, spec.file_glob), "", "ready"
        )
    return ManifestEntry(spec.name, spec.license, source, 0, "", "manual — see note")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download public verification corpora (§15.1).")
    parser.add_argument("--list", action="store_true", help="list corpora and exit")
    parser.add_argument("--only", default="", help="comma-separated corpus names to fetch")
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help="download destination dir")
    parser.add_argument(
        "--write-skeleton",
        action="store_true",
        help="(re)write MANIFEST.md with 'not downloaded' rows and exit",
    )
    args = parser.parse_args(argv)

    if args.list:
        for spec in CORPORA:
            src = spec.url or " ".join(spec.command) or "(manual)"
            print(f"{spec.name:10} {spec.kind:8} {spec.license:28} {src}")
        return 0

    if args.write_skeleton:
        write_manifest(_skeleton_entries())
        print(f"Wrote skeleton manifest to {MANIFEST_PATH}")
        return 0

    dest = Path(args.dest)
    wanted = {n.strip() for n in args.only.split(",") if n.strip()}
    entries: list[ManifestEntry] = []
    for spec in CORPORA:
        if wanted and spec.name not in wanted:
            continue
        print(f"fetching {spec.name} ({spec.kind}) ...", file=sys.stderr)
        entries.append(fetch(spec, dest))
    write_manifest(entries)
    print(f"Wrote manifest to {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
