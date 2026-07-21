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
from collections.abc import Sequence
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
    """One parametric/functional test result with its limits + the conditions active when it ran.
    ``rec_type`` is the STDF record it came from — ``PTR`` (single parametric), ``MPR`` (one pin of a
    multiple-result parametric, ``pin`` set), or ``FTR`` (functional, ``result`` is None)."""

    test_num: int
    test_txt: str
    result: float | None
    head: int
    site: int
    passed: bool
    part_id: str
    conditions: dict[str, str]
    lo_limit: float | None = None
    hi_limit: float | None = None
    units: str = ""
    rec_type: str = "PTR"
    pin: int | None = None


@dataclass
class StdfPart:
    """One part touchdown (a PRR) with its identity + bin result. The tuple returned by
    :meth:`key` is the **unique part identifier** used to trace a part across insertions."""

    lot_id: str
    sublot_id: str
    wafer_id: str
    x: int | None
    y: int | None
    part_id: str
    head: int
    site: int
    hard_bin: int
    soft_bin: int
    passed: bool
    insertion: str
    test_time_ms: int = 0  # PRR TEST_T: how long this part took to test

    def key(self, fields: Sequence[str] = ("lot", "sublot", "wafer", "x", "y")) -> tuple[str, ...]:
        """The unique part id from the chosen ``fields`` (default wafer x/y). Falls back to part_id
        if the chosen fields carry no coordinate/wafer identity."""
        m = {
            "lot": self.lot_id, "sublot": self.sublot_id, "wafer": self.wafer_id,
            "x": "" if self.x is None else str(self.x), "y": "" if self.y is None else str(self.y),
            "part_id": self.part_id, "site": str(self.site),
        }
        parts = tuple(m.get(f, "") for f in fields)
        if not any(parts):  # no identity from the chosen fields → fall back to part_id
            return (self.part_id,)
        return parts


@dataclass
class StdfRun:
    """The tests + part touchdowns parsed from one STDF file, plus lot/program/insertion identity."""

    lot_id: str = ""
    sublot_id: str = ""
    part_typ: str = ""
    job_nam: str = ""
    insertion: str = ""
    tests: list[StdfTest] = field(default_factory=list)
    parts: list[StdfPart] = field(default_factory=list)


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


def _as_seq(v: object) -> list[object]:
    """A pystdf array field comes back as a tuple/list; normalise (scalar/None → []/[v])."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


@dataclass
class _StaticFields:
    """The descriptive fields the STDF spec puts on only the FIRST P/M/F-TR for a test number."""

    test_txt: str
    units: str
    lo: float | None
    hi: float | None


def _resolve_static(
    f: dict[str, object], tnum: int, defaults: dict[int, _StaticFields]
) -> _StaticFields:
    """Resolve TEST_TXT / LO_LIMIT / HI_LIMIT / UNITS for a P/M/FTR honoring the STDF v4 rule that a
    test's descriptive fields are carried only by its **first** record (looked up **by test number**);
    later records omit them, or flag them invalid via ``OPT_FLAG``, and inherit the first record's
    values (R-STDF-1). ``OPT_FLAG`` bits (PTR/MPR): ``0x40`` = no low limit, ``0x10`` = LO_LIMIT
    invalid→inherit; ``0x80`` = no high limit, ``0x20`` = HI_LIMIT invalid→inherit. A missing
    ``OPT_FLAG`` (trailing field omitted) means inherit any absent limit."""
    d = defaults.get(tnum)
    # The FIRST record for a test number is authoritative for its name/units — prefer the cached
    # value so a later record carrying a *different* name doesn't split one test into two in the
    # name-keyed audit/diff join downstream (red-team phase6b #5).
    txt = (d.test_txt if (d and d.test_txt) else str(f.get("TEST_TXT") or "").strip())
    units = (d.units if (d and d.units) else str(f.get("UNITS") or "").strip())
    lo_rec, hi_rec = _as_float(f.get("LO_LIMIT")), _as_float(f.get("HI_LIMIT"))
    opt = _as_int(f.get("OPT_FLAG"), -1)  # -1 ⇒ OPT_FLAG absent (limits omitted → inherit)
    if opt < 0:
        lo = lo_rec if lo_rec is not None else (d.lo if d else None)
        hi = hi_rec if hi_rec is not None else (d.hi if d else None)
    else:
        lo = None if opt & 0x40 else ((d.lo if d else None) if opt & 0x10 else lo_rec)
        hi = None if opt & 0x80 else ((d.hi if d else None) if opt & 0x20 else hi_rec)
    resolved = _StaticFields(txt, units, lo, hi)
    if d is None:  # first record for this test number sets the defaults every later one inherits
        defaults[tnum] = resolved
    return resolved


def _tests_from_record(
    name: str, f: dict[str, object], conditions: dict[str, str],
    defaults: dict[int, _StaticFields],
) -> list[StdfTest]:
    """Turn one PTR / FTR / MPR record into StdfTests. PTR → one; FTR → one functional (result None);
    MPR → **one per pin** (``TEST_TXT[pin]``), so each pin is analyzed as its own distribution.
    Static fields (name/limits/units) are resolved via first-record lookup (:func:`_resolve_static`)."""
    flg = _as_int(f.get("TEST_FLG"))
    passed = not (flg & 0x80)
    tnum = _as_int(f.get("TEST_NUM"))
    static = _resolve_static(f, tnum, defaults)
    txt, units, lo, hi = static.test_txt, static.units, static.lo, static.hi
    head, site = _as_int(f.get("HEAD_NUM")), _as_int(f.get("SITE_NUM"))

    def make(result: float | None, *, rec: str, pin: int | None, suffix: str = "") -> StdfTest:
        return StdfTest(
            test_num=tnum, test_txt=(txt or f"test {tnum}") + suffix, result=result, head=head,
            site=site, passed=passed, part_id="", conditions=dict(conditions),
            lo_limit=lo, hi_limit=hi, units=units, rec_type=rec, pin=pin,
        )

    if name == "Mpr":
        results = _as_seq(f.get("RTN_RSLT"))
        pins = _as_seq(f.get("RTN_INDX"))
        out = []
        for i, r in enumerate(results):
            pin = _as_int(pins[i]) if i < len(pins) else i
            out.append(make(_as_float(r), rec="MPR", pin=pin, suffix=f"[{pin}]"))
        return out
    if name == "Ftr":
        return [make(None, rec="FTR", pin=None)]
    return [make(_as_float(f.get("RESULT")), rec="PTR", pin=None)]


def parse_stdf_tests(
    data: bytes,
    *,
    rules: tuple[DatalogRule, ...] | None = None,
    scope: str = "part",
    insertion: str = "",
) -> StdfRun:
    """Parse an STDF v4 byte stream into per-test records + per-part touchdowns, each carrying a
    snapshot of the conditions active when it ran (R-STDF-1). DTRs drive a :class:`ConditionTracker`;
    PTR/FTR/MPR snapshot it; the PRR yields an :class:`StdfPart` (lot/sublot/wafer/x/y/bin) for part
    traceability + yield. ``insertion`` labels the run (else MIR ``TEST_COD``/``JOB_NAM``).
    ``scope='part'`` resets conditions at each part boundary. pystdf is the ``[stdf]`` extra."""
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
    wafer_id = ""
    part_index = 0
    pending: list[StdfTest] = []
    defaults: dict[int, _StaticFields] = {}  # first-record static fields per test number (run-scoped)
    for name, f in collected:
        if name == "Mir":
            run.lot_id = str(f.get("LOT_ID") or "")
            run.sublot_id = str(f.get("SBLOT_ID") or "")
            run.part_typ = str(f.get("PART_TYP") or "")
            run.job_nam = str(f.get("JOB_NAM") or "")
            run.insertion = insertion or str(f.get("TEST_COD") or "") or run.job_nam
        elif name == "Wir":
            wafer_id = str(f.get("WAFER_ID") or "")
        elif name == "Dtr":
            tracker.apply(str(f.get("TEXT_DAT") or ""))
        elif name == "Pir":
            part_index += 1
        elif name in ("Ptr", "Ftr", "Mpr"):
            pending.extend(_tests_from_record(name, f, tracker.snapshot(), defaults))
        elif name == "Prr":
            part_id = str(f.get("PART_ID") or part_index)
            part_flg = _as_int(f.get("PART_FLG"))
            xc, yc = _as_int(f.get("X_COORD"), -32768), _as_int(f.get("Y_COORD"), -32768)
            run.parts.append(
                StdfPart(
                    lot_id=run.lot_id, sublot_id=run.sublot_id, wafer_id=wafer_id,
                    x=None if xc == -32768 else xc, y=None if yc == -32768 else yc,
                    part_id=part_id, head=_as_int(f.get("HEAD_NUM")), site=_as_int(f.get("SITE_NUM")),
                    hard_bin=_as_int(f.get("HARD_BIN")), soft_bin=_as_int(f.get("SOFT_BIN")),
                    passed=not (part_flg & 0x08), insertion=run.insertion,
                    test_time_ms=_as_int(f.get("TEST_T")),
                )
            )
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
