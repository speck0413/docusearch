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

import re
from dataclasses import dataclass


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
