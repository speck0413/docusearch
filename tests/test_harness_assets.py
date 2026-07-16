"""Guard the seeded harness data files (§15.2–15.4).

These YAML assets are hand-edited config the later phases depend on. Keep them valid
and structurally sane so a typo can't silently weaken a gate.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_HARNESS = Path(__file__).resolve().parents[1] / "harness"


def test_thresholds_yaml_is_valid_and_complete() -> None:
    data = yaml.safe_load((_HARNESS / "thresholds.yaml").read_text(encoding="utf-8"))
    assert set(data) >= {
        "needle",
        "autoqa",
        "crosslink",
        "absent_negatives",
        "compare",
        "redteam",
    }
    # The hard gates from §15.2–15.4 are probabilities in [0, 1].
    assert data["needle"]["exact_nonce_top1"] == 1.00
    assert 0.0 <= data["autoqa"]["hybrid_recall_at10"] <= 1.0
    assert 0.0 <= data["compare"]["mean_overlap_at10"] <= 1.0
    assert 0.0 <= data["compare"]["top1_logical_doc_match"] <= 1.0
    assert data["absent_negatives"]["fabricated_citations"] == 0
    assert data["redteam"]["token_recovery_min"] >= 0.9


def test_golden_queries_yaml_shape() -> None:
    data = yaml.safe_load((_HARNESS / "golden_queries.yaml").read_text(encoding="utf-8"))
    assert isinstance(data, list) and data
    for entry in data:
        assert {"id", "query", "expect_docs", "notes"} <= set(entry)
