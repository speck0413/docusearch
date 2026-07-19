"""STDF v4 support (Phase 6 / GATE 6): the DTR **condition engine** + (later) file ingest and the
agent-facing analytics tools.

The condition engine (R-STDF-1) reads Datalog Text Records (DTRs) in file order and maintains the
set of **active conditions** that apply to every following test until changed or cleared:

- ``COND: KEY=VALUE, KEY2=VALUE2, …`` — set a comma-delimited list of sticky conditions. Every key
  and value is **trimmed** of surrounding whitespace.
- ``COND_OFF: KEY`` — clear one key.
- ``COND_OFF`` (bare) **or** ``COND_OFF: *`` — clear **all** conditions.

It is driven by YAML ``stdf.datalog_rules`` (``match`` regex → ``action`` set/clear with an
``extract`` strategy), so other conventions (shmoo, corner, temperature …) plug in without code.
``scope`` is a config knob: ``part`` (default — reset at each part boundary, PRR) or ``run``.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatalogRule:
    """One DTR-text rule. ``match`` is a regex; a ``set`` rule feeds its ``body`` group through
    ``extract`` (``kv_csv`` → ``"K=V, K=V"``); a ``clear`` rule reads its ``key`` group (empty or
    ``*`` clears all). ``scope`` and ``store`` describe where/how long the result lives."""

    name: str
    match: str
    action: str = "set"  # set | clear
    extract: str = "kv_csv"
    scope: str = "sticky"
    store: str = "conditions"


# The built-in COND / COND_OFF convention. `COND_OFF` is matched with a trailing word boundary so
# `COND_OFFICER` (or any longer token) never reads as a clear. Order-independent: the two regexes
# are mutually exclusive (`COND:` vs `COND_OFF`).
DEFAULT_DATALOG_RULES: tuple[DatalogRule, ...] = (
    DatalogRule(name="conditions_off", match=r"^\s*COND_OFF\b\s*:?\s*(?P<key>\S*)\s*$", action="clear"),
    DatalogRule(name="conditions", match=r"^\s*COND:\s*(?P<body>.+?)\s*$", action="set", extract="kv_csv"),
)


def parse_kv_csv(body: str) -> dict[str, str]:
    """Parse ``"K = V , K2=V2"`` into ``{K: V, K2: V2}`` with every key and value trimmed. Pairs
    without ``=`` or with an empty key are skipped."""
    out: dict[str, str] = {}
    for pair in body.split(","):
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        key, value = key.strip(), value.strip()
        if key:
            out[key] = value
    return out


class ConditionTracker:
    """Applies datalog rules to a stream of DTR texts, maintaining the active condition set. Feed
    each DTR's text to :meth:`apply` in file order; call :meth:`reset` at a part boundary when the
    scope is per-part; read :meth:`snapshot` to attach the active conditions to a test."""

    def __init__(self, rules: tuple[DatalogRule, ...] | None = None) -> None:
        self._rules = rules if rules is not None else DEFAULT_DATALOG_RULES
        self._compiled = [(r, re.compile(r.match)) for r in self._rules]
        self._active: dict[str, str] = {}

    def apply(self, text: str) -> None:
        """Apply the first matching rule to one DTR text (no-op if none match)."""
        for rule, rx in self._compiled:
            m = rx.match(text)
            if m is None:
                continue
            if rule.action == "set" and rule.extract == "kv_csv":
                self._active.update(parse_kv_csv(m.group("body")))
            elif rule.action == "clear":
                key = (m.groupdict().get("key") or "").strip()
                if key in ("", "*"):
                    self._active.clear()
                else:
                    self._active.pop(key, None)
            return  # first matching rule wins

    def reset(self) -> None:
        """Clear all active conditions — called at a part boundary when scope is per-part."""
        self._active.clear()

    def snapshot(self) -> dict[str, str]:
        """A **copy** of the active conditions to attach to a test (later applies won't mutate it)."""
        return dict(self._active)


# --------------------------------------------------------------- STDF file parse


@dataclass
class StdfTest:
    """One parametric/functional test result with the conditions active when it ran."""

    test_num: int
    test_txt: str
    result: float | None
    head: int
    site: int
    passed: bool
    part_id: str
    conditions: dict[str, str]


@dataclass
class StdfRun:
    """The tests parsed from one STDF file, plus lot/program identity from the MIR."""

    lot_id: str = ""
    part_typ: str = ""
    job_nam: str = ""
    tests: list[StdfTest] = field(default_factory=list)


def _as_int(v: object, default: int = 0) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
        return int(v)
    return default


def _as_float(v: object) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def parse_stdf_tests(
    data: bytes,
    *,
    rules: tuple[DatalogRule, ...] | None = None,
    scope: str = "part",
) -> StdfRun:
    """Parse an STDF v4 byte stream into per-test records, each carrying a snapshot of the conditions
    active when it ran (R-STDF-1). DTRs drive a :class:`ConditionTracker`; PTR/FTR/MPR snapshot it;
    part_id is filled from the PRR; ``scope='part'`` resets conditions at each part boundary.
    pystdf is the ``[stdf]`` extra, imported lazily."""
    from pystdf.IO import Parser  # lazy: only when an STDF file is parsed

    collected: list[tuple[str, dict[str, object]]] = []

    class _Sink:
        def before_begin(self, s: object) -> None: ...
        def after_begin(self, s: object) -> None: ...
        def before_complete(self, s: object) -> None: ...
        def after_complete(self, s: object) -> None: ...
        def before_send(self, s: object, d: object) -> None: ...
        def after_send(self, s: object, d: tuple) -> None:  # type: ignore[type-arg]
            rectype, values = d
            collected.append(
                (rectype.__class__.__name__, dict(zip(rectype.fieldNames, values, strict=False)))
            )
        def before_cancel(self, s: object, e: object) -> None: ...
        def after_cancel(self, s: object, e: object) -> None: ...

    parser = Parser(inp=io.BytesIO(data))
    parser.addSink(_Sink())
    parser.parse()

    tracker = ConditionTracker(rules)
    run = StdfRun()
    part_index = 0
    pending: list[StdfTest] = []
    for name, f in collected:
        if name == "Mir":
            run.lot_id = str(f.get("LOT_ID") or "")
            run.part_typ = str(f.get("PART_TYP") or "")
            run.job_nam = str(f.get("JOB_NAM") or "")
        elif name == "Dtr":
            tracker.apply(str(f.get("TEXT_DAT") or ""))
        elif name == "Pir":
            part_index += 1
        elif name in ("Ptr", "Ftr", "Mpr"):
            flg = _as_int(f.get("TEST_FLG"))
            pending.append(
                StdfTest(
                    test_num=_as_int(f.get("TEST_NUM")),
                    test_txt=str(f.get("TEST_TXT") or "").strip(),
                    result=_as_float(f.get("RESULT")),
                    head=_as_int(f.get("HEAD_NUM")),
                    site=_as_int(f.get("SITE_NUM")),
                    passed=not (flg & 0x80),
                    part_id="",
                    conditions=tracker.snapshot(),
                )
            )
        elif name == "Prr":
            part_id = str(f.get("PART_ID") or part_index)
            for t in pending:
                t.part_id = part_id
            run.tests.extend(pending)
            pending = []
            if scope == "part":
                tracker.reset()
    run.tests.extend(pending)  # tests with no closing PRR
    return run


def stdf_test_text(t: StdfTest) -> str:
    """The searchable text for one test chunk: name, number, result, pass/fail, part/head/site, and
    ``COND k=v`` tokens (BM25-visible) for its active conditions."""
    parts = [t.test_txt or f"test {t.test_num}", f"test {t.test_num}"]
    if t.result is not None:
        parts.append(f"result {t.result:g}")
    parts.append("PASS" if t.passed else "FAIL")
    parts.append(f"part {t.part_id} head {t.head} site {t.site}")
    if t.conditions:
        parts.append("COND " + " ".join(f"{k}={v}" for k, v in sorted(t.conditions.items())))
    return " ".join(parts)
