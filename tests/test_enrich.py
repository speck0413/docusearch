"""Phase 5 — pre-flight classification (R-ING-7): sample docs stratified by folder → Claude temp 0
→ proposed chunk rules + gotcha regexes in preflight_rules.yaml → Stephen approves before they run."""

from __future__ import annotations

from pathlib import Path

from docusearch import enrich


def test_stratified_sample_covers_every_folder(tmp_path: Path) -> None:
    # 3 folders of different sizes; a stratified sample must draw from EVERY folder (not just the
    # biggest), be capped at n, and be deterministic for a given seed.
    paths: list[Path] = []
    for folder, count in (("guide", 20), ("api", 8), ("faq", 2)):
        d = tmp_path / folder
        d.mkdir()
        for i in range(count):
            p = d / f"{i}.html"
            p.write_text("x", encoding="utf-8")
            paths.append(p)

    sample = enrich.stratified_sample(paths, n=9, seed=7)
    assert len(sample) == 9
    folders = {p.parent.name for p in sample}
    assert folders == {"guide", "api", "faq"}  # every folder represented, incl. the 2-doc one
    # deterministic
    assert enrich.stratified_sample(paths, n=9, seed=7) == sample
    # a different seed may reorder within folders but still covers all + stays capped
    other = enrich.stratified_sample(paths, n=9, seed=99)
    assert len(other) == 9 and {p.parent.name for p in other} == {"guide", "api", "faq"}


def test_stratified_sample_returns_all_when_n_exceeds_corpus(tmp_path: Path) -> None:
    paths = []
    d = tmp_path / "only"
    d.mkdir()
    for i in range(3):
        p = d / f"{i}.html"
        p.write_text("x", encoding="utf-8")
        paths.append(p)
    assert set(enrich.stratified_sample(paths, n=50, seed=1)) == set(paths)


def test_preflight_rules_roundtrip_and_approval_gate(tmp_path: Path) -> None:
    rules = enrich.PreflightRules(
        approved=False,
        gotcha_patterns=[
            enrich.GotchaPattern(pattern=r"\bdo NOT\b", label="warning"),
            enrich.GotchaPattern(pattern=r"deprecated", label="deprecation"),
        ],
        notes="Headings are H1>H2; code fenced; watch 'do NOT' cautions.",
        sampled=12,
    )
    path = tmp_path / "preflight_rules.yaml"
    enrich.write_preflight_rules(rules, path)
    text = path.read_text(encoding="utf-8")
    assert "approved: false" in text and "do NOT" in text  # human-readable, editable

    loaded = enrich.load_preflight_rules(path)
    assert loaded.gotcha_patterns == rules.gotcha_patterns and loaded.approved is False
    # the gate: unapproved rules must not apply
    assert enrich.active_gotcha_patterns(path) == []  # nothing until approved

    # Stephen approves by editing the file
    path.write_text(text.replace("approved: false", "approved: true"), encoding="utf-8")
    active = enrich.active_gotcha_patterns(path)
    assert [g.label for g in active] == ["warning", "deprecation"]


def test_active_gotcha_patterns_missing_file_is_empty(tmp_path: Path) -> None:
    assert enrich.active_gotcha_patterns(tmp_path / "nope.yaml") == []


def test_match_gotcha_returns_first_label_or_none() -> None:
    pats = [
        enrich.GotchaPattern(pattern=r"do NOT", label="warning"),
        enrich.GotchaPattern(pattern=r"deprecated", label="deprecation"),
    ]
    assert enrich.match_gotcha("You must do NOT power off", pats) == "warning"
    assert enrich.match_gotcha("this API is deprecated", pats) == "deprecation"
    assert enrich.match_gotcha("ordinary prose", pats) is None
    # a broken regex is skipped, not crash-inducing
    assert enrich.match_gotcha("x", [enrich.GotchaPattern(pattern=r"(", label="bad")]) is None


def test_gotcha_tag_prefixes_text_bm25_visible() -> None:
    tagged = enrich.gotcha_tag_text("do NOT power off mid-write")
    assert tagged.startswith("[GOTCHA]") and "power off" in tagged  # prefix + original text
    assert enrich.gotcha_tag_text("[GOTCHA] already tagged") == "[GOTCHA] already tagged"  # idempotent
