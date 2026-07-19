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


def test_propose_rules_parses_claude_json() -> None:
    import json

    def fake_runner(argv: list[str]) -> tuple[int, str, str]:
        assert "-p" in argv and "--output-format" in argv  # a headless JSON claude call
        reply = json.dumps(
            {"gotcha_patterns": [{"pattern": r"do NOT", "label": "warning"}], "notes": "H1>H2."}
        )
        return 0, json.dumps({"result": reply, "is_error": False}), ""

    rules = enrich.propose_rules(["doc one", "doc two"], model="m", runner=fake_runner)
    assert rules.approved is False and rules.sampled == 2
    assert rules.gotcha_patterns == [enrich.GotchaPattern(r"do NOT", "warning")]
    assert "H1>H2" in rules.notes


def test_propose_rules_raises_on_cli_failure() -> None:
    import pytest

    def failing(argv: list[str]) -> tuple[int, str, str]:
        return 1, "", "claude: not found"

    with pytest.raises(enrich.EnrichError):
        enrich.propose_rules(["doc"], model="m", runner=failing)


def test_run_preflight_writes_unapproved_rules_from_sampled_source(tmp_path: Path) -> None:
    import json

    from docusearch import config as cfg

    root = tmp_path / "docs"
    for folder in ("a", "b"):
        d = root / folder
        d.mkdir(parents=True)
        for i in range(4):
            (d / f"{i}.html").write_text(
                f"<body><h1>Doc {folder}{i}</h1><p>Do NOT power off during a write.</p></body>",
                encoding="utf-8",
            )
    config_path = tmp_path / "d.yaml"
    config_path.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: d\n    location: "{root.as_posix()}"\n    min_content_chars: 1\n'
        'embed:\n  model: "none"\nenrich:\n  preflight_sample: 4\n',
        encoding="utf-8",
    )
    config = cfg.load(config_path)

    seen_prompt: dict[str, str] = {}

    def fake_runner(argv: list[str]) -> tuple[int, str, str]:
        seen_prompt["p"] = argv[argv.index("-p") + 1]
        reply = json.dumps(
            {"gotcha_patterns": [{"pattern": r"do NOT", "label": "warning"}], "notes": "H1 headings."}
        )
        return 0, json.dumps({"result": reply, "is_error": False}), ""

    out = tmp_path / "preflight_rules.yaml"
    rules = enrich.run_preflight(config, out_path=out, model="m", runner=fake_runner, seed=1)
    assert out.is_file()
    assert rules.approved is False and rules.sampled == 4  # capped at preflight_sample
    assert rules.gotcha_patterns == [enrich.GotchaPattern(r"do NOT", "warning")]
    assert "Do NOT power off" in seen_prompt["p"]  # real sampled text reached the model
    assert enrich.active_gotcha_patterns(out) == []  # still needs approval
