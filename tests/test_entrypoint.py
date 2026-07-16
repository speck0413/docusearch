"""`python -m docusearch` works as a module entry point (R-ARCH-5, cross-platform)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from docusearch import __version__


def test_python_m_docusearch_version(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "docusearch", "--version"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert __version__ in result.stdout


def test_python_m_docusearch_init(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "docusearch", "init"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert (tmp_path / "docusearch.yaml").exists()
    assert "Wrote config" in result.stdout
