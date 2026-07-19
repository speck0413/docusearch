"""Fetch a git / GitHub repo source via the user's own ``git`` (Phase 9 / GATE 9).

A source whose ``location`` is a git URL is cloned to a cache under the store's ``staging_dir`` and the
normal code pipeline runs on the checkout — "GitHub sources treated the same". Authentication is
entirely git's (SSH keys, credential helper), so docusearch never sees or stores a token, and it works
for any git host, not just GitHub. Clones are shallow (only the current tree is needed for symbol
extraction). Hardened against URL/flag injection and against hanging on an auth prompt.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

_SCHEMES = ("https://", "http://", "ssh://", "git://", "file://")
_HOSTS = ("github.com/", "gitlab.com/", "bitbucket.org/")
_SCP_RE = re.compile(r"^[\w.-]+@[\w.-]+:")  # scp-like: git@github.com:owner/repo.git


class GitFetchError(Exception):
    """A git clone/fetch could not be completed — reported as a clean one-line ingest error."""


def is_remote(location: str) -> bool:
    """True when a source ``location`` names a git remote (URL / scp-form / bare known host) rather
    than a local directory path."""
    loc = location.strip()
    if not loc:
        return False
    return loc.startswith(_SCHEMES) or loc.startswith(_HOSTS) or bool(_SCP_RE.match(loc))


def normalize_url(location: str) -> str:
    """Add ``https://`` to a bare ``github.com/owner/repo``; otherwise return the URL unchanged."""
    loc = location.strip()
    return "https://" + loc if loc.startswith(_HOSTS) else loc


def _safe_name(url: str) -> str:
    """A deterministic, filesystem-safe **single-component** cache directory name from the repo URL —
    no path separators and no ``..`` runs, so the clone target can never escape the cache root."""
    s = re.sub(r"^\w+://", "", normalize_url(url))
    s = re.sub(r"\.git$", "", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"\.{2,}", "_", s).strip("._-")  # collapse dot-runs; never a bare . / .. component
    return s[:120] or "repo"


def _validate(url: str, ref: str) -> None:
    # git treats a leading '-' as an option, and ext::/fd:: are transports that can run arbitrary
    # commands — refuse both outright (defence in depth; the URL comes from config, but never trust it
    # to `git` unchecked). The clone itself also passes `--` before positionals.
    for value, what in ((url, "repo URL"), (ref, "ref")):
        if value.startswith("-"):
            raise GitFetchError(f"refusing suspicious {what} (leading dash): {value!r}")
    if re.match(r"^\s*(ext|fd)::", url, re.IGNORECASE):
        raise GitFetchError(f"refusing unsafe git transport: {url!r}")


def fetch_repo(
    location: str, dest_root: str | os.PathLike[str], *, ref: str = "", depth: int = 1,
    refresh: bool = False, timeout: float = 300.0,
) -> Path:
    """Clone ``location`` (a git URL) into ``dest_root/<derived-name>`` and return the checkout path.

    Shallow + single-branch by default; ``ref`` selects a branch or tag. A cached clone is reused
    unless ``refresh`` is set (then it is re-cloned). Auth is delegated to the user's git;
    ``GIT_TERMINAL_PROMPT=0`` means a repo needing credentials fails fast instead of hanging.
    Raises :class:`GitFetchError` on any failure (missing git, bad URL, auth, timeout)."""
    url = normalize_url(location)
    _validate(url, ref)
    root = Path(dest_root)
    root.mkdir(parents=True, exist_ok=True)
    dest = root / _safe_name(url)

    if dest.exists() and not refresh:
        return dest
    if dest.exists():
        shutil.rmtree(dest)

    cmd = ["git", "clone", "--depth", str(int(depth)), "--single-branch"]
    if ref:
        cmd += ["--branch", ref]
    cmd += ["--", url, str(dest)]
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}  # never block on an interactive auth prompt
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    except FileNotFoundError as exc:  # git not installed
        raise GitFetchError("git is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitFetchError(f"git clone {url!r} timed out after {timeout:.0f}s") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        last = detail.splitlines()[-1] if detail else f"exit {proc.returncode}"
        raise GitFetchError(f"git clone {url!r} failed: {last}")
    return dest
