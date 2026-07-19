"""Configuration: one source of truth for the template, the defaults, and validation.

External YAML is the *only* configuration mechanism (R-CFG-1); secrets and roles come
from environment variables instead. The commented template that ``docusearch init``
writes is generated from the ``SCHEMA`` record list below (R-CFG-3) — never a
hand-copied string — so the docs, the defaults, and the validator can never drift
apart. ``embed.model: none`` (BM25-only) and ``auto`` are first-class (R-CFG-4).

Public surface:
    Config, and the nested *Config dataclasses  -- typed, frozen configuration
    ConfigError                                 -- raised on invalid enum values
    DEFAULT_CONFIG_PATH                         -- Path("docusearch.yaml")
    render_template() -> str                    -- the fully-commented YAML template
    write_template(path, *, force=False) -> bool
    load(path=DEFAULT_CONFIG_PATH) -> Config    -- auto-creates a missing file (R-CFG-2)
    default() -> Config                         -- the built-in defaults
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from ._version import __version__

DEFAULT_CONFIG_PATH = Path("docusearch.yaml")

_Value = str | int | float | bool | list[str]


class ConfigError(Exception):
    """A config value is invalid (e.g. a bad enum). Carries the accepted options."""


# --------------------------------------------------------------------------- schema
# The single source of truth. Each record carries the key, its default, a comment
# (what it does + accepted options), and — for closed enums — the allowed choices.


@dataclass(frozen=True)
class _Field:
    key: str
    default: _Value
    comment: str = ""  # block comment rendered above the field (may be multi-line)
    inline: str = ""  # short trailing comment after the value
    choices: tuple[str, ...] | None = None  # closed enum => validated on load


@dataclass(frozen=True)
class _Section:
    """A nested mapping, e.g. ``embed:`` with scalar children."""

    key: str
    fields: tuple[_Field, ...]
    comment: str = ""


@dataclass(frozen=True)
class _ListSection:
    """A list of mappings documented by one example entry, e.g. ``sources:``."""

    key: str
    fields: tuple[_Field, ...]
    comment: str = ""


_Node = _Field | _Section | _ListSection


SCHEMA: tuple[_Node, ...] = (
    _Field(
        "mode",
        "standalone",
        comment=(
            "Run mode.\n"
            "  standalone : index + search + serve on this machine\n"
            "  server     : ingestion/index/API host (does the heavy work)\n"
            "  client     : thin client talking to server_url below"
        ),
        choices=("standalone", "server", "client"),
    ),
    _Field(
        "server_url",
        "http://docs-server.local:8321",
        comment="Only used when mode: client.",
    ),
    _Section(
        "paths",
        (
            _Field(
                "staging_dir",
                "./staging",
                inline="mirrored sources + extracted images live here",
            ),
            _Field("db_path", "./catalog.db", inline="SQLite database (one per index)"),
            _Field("tmp_dir", "./tmp", inline="ALL generated output (reports, logs, gates)"),
        ),
    ),
    _ListSection(
        "sources",
        (
            _Field("type", "fs", inline="fs = filesystem folder (git/sharepoint later)"),
            _Field("name", "vendor-html", inline="a short label for this source"),
            _Field("version", "", inline='doc release/version, e.g. "2024.3" (blank = untracked)'),
            _Field("location", "D:/docs/vendor-html", inline="folder to ingest (Windows or POSIX)"),
            _Field("include", ["**/*.html"], inline="glob whitelist"),
            _Field("exclude", ["**/nav/**"], inline="glob blacklist (framework/nav noise)"),
            _Field(
                "content_selector",
                "",
                comment=(
                    'CSS selector for the real article body, e.g. "main.article".\n'
                    "Empty = keep the whole page."
                ),
            ),
            _Field(
                "strip_selectors",
                [],
                comment=(
                    "CSS selectors removed before extraction,\n"
                    'e.g. ["header", "footer", ".sidebar"].'
                ),
            ),
            _Field(
                "min_content_chars",
                200,
                inline="below this after stripping => skipped and reported in the audit",
            ),
            _Field(
                "audience",
                ["engineering"],
                comment=(
                    "Who may see these docs (cooperative filter, not cryptographic).\n"
                    "Documented values: company | engineering | test-eng | finance."
                ),
            ),
        ),
        comment="One entry per source folder. Copy the block to add more sources.",
    ),
    _Section(
        "embed",
        (
            _Field(
                "model",
                "sentence-transformers/all-MiniLM-L6-v2",
                comment=(
                    "Embedding model. Options:\n"
                    "  none  -> BM25-only. Fastest, smallest, no model download. Good baseline.\n"
                    "  auto  -> negotiate: ask the server what it uses; if that model fits\n"
                    "           locally (see auto_max_mb) use it, else send plain text.\n"
                    "  ...or a model id. Curated choices below (any Hugging Face\n"
                    "  sentence-transformers id also works):\n"
                    "    sentence-transformers/all-MiniLM-L6-v2   # 384d, ~90MB  - laptop default\n"
                    "    BAAI/bge-small-en-v1.5                   # 384d, ~130MB - better quality\n"
                    "    BAAI/bge-base-en-v1.5                    # 768d, ~440MB - server mid\n"
                    "    BAAI/bge-large-en-v1.5                   # 1024d, ~1.3GB - server best\n"
                    "    nomic-ai/nomic-embed-text-v1.5           # 768d, needs trust_remote_code"
                ),
            ),
            _Field(
                "device",
                "auto",
                comment=(
                    "Compute device for embedding:\n"
                    "  auto -> best available (CUDA GPU, else Apple-Silicon Metal 'mps', else cpu)\n"
                    "  cpu  -> most reproducible; cuda -> NVIDIA GPU; mps -> macOS GPU (Metal)"
                ),
                choices=("auto", "cpu", "cuda", "mps"),
            ),
            _Field("batch_size", 128, inline="chunks embedded per batch"),
            _Field(
                "auto_max_mb",
                200,
                inline="'auto' loads the server's model locally only if smaller than this",
            ),
            _Field("trust_remote_code", False, inline="required true for some models (see list)"),
        ),
    ),
    _Section(
        "index",
        (
            _Field("chunk_tokens", 350, inline="target chunk size (code blocks never split)"),
            _Field("chunk_overlap", 40, inline="token overlap between adjacent chunks"),
            _Field("ann", True, inline="build hnswlib ANN when embeddings exist"),
            _Field("ann_m", 16, inline="hnswlib graph connectivity"),
            _Field("ann_ef_construction", 200, inline="hnswlib build-time accuracy"),
        ),
    ),
    _Section(
        "search",
        (
            _Field("top_k_default", 10, inline="results returned when not otherwise specified"),
            _Field("rrf_k", 60, inline="RRF constant for hybrid fusion"),
            _Field("bm25_only", False, inline="force-skip vectors at query time"),
        ),
    ),
    _Section(
        "serve",
        (
            _Field("host", "0.0.0.0", inline="bind address"),
            _Field("port", 8321, inline="HTTP port for REST + MCP"),
            _Field("mcp_path", "/mcp", inline="MCP over streamable HTTP mounts here"),
            _Field(
                "self_heal_minutes",
                60,
                inline="auto-prune orphaned docs every N min while serving (0 = off)",
            ),
        ),
        comment=(
            "Roles come from the DOCUSEARCH_ROLES env var on the client/agent process,\n"
            "e.g. DOCUSEARCH_ROLES=engineering,test-eng"
        ),
    ),
    _Section(
        "access",
        (
            _Field(
                "visibility",
                "public",
                comment=(
                    "Who may search this document store:\n"
                    "  public  : anyone on the server (the default)\n"
                    "  private : only allowed_users / allowed_groups below, verified from the\n"
                    "            X-Docusearch-User (and X-Docusearch-Groups) request headers"
                ),
                choices=("public", "private"),
            ),
            _Field("allowed_users", [], inline="usernames allowed when visibility: private"),
            _Field(
                "allowed_groups", [], inline="groups allowed when visibility: private"
            ),
        ),
        comment="Access control for this store. Defaults to public (nothing defined = public).",
    ),
    _Section(
        "enrich",
        (
            _Field("preflight_sample", 150, inline="docs sampled for rule proposal (temp 0)"),
            _Field(
                "preflight_rules",
                "preflight_rules.yaml",
                comment=(
                    "Where `docusearch preflight` writes proposed gotcha rules and where ingest\n"
                    "reads them. Rules apply ONLY after you set `approved: true` in that file."
                ),
            ),
            _Field("ai_summaries", False, inline="Phase 5+ — off by default"),
            _Field(
                "vision_images",
                False,
                comment=(
                    "Cloud image OCR + description. When true, `docusearch vision` sends each\n"
                    "retained image ONCE to vision_model and stores the result (searchable +\n"
                    "report-embeddable). Off by default: it calls a paid cloud API. Auth comes\n"
                    "from the ANTHROPIC_API_KEY env var or an `ant auth login` profile — never\n"
                    "put a key in this file."
                ),
            ),
            _Field(
                "vision_provider",
                "claude-cli",
                comment=(
                    "Which vision backend `docusearch vision` uses:\n"
                    "  claude-cli : shell out to the `claude` CLI — uses your Claude Code login,\n"
                    "               NO API key (the natural path if you already run Claude Code)\n"
                    "  anthropic  : Anthropic Messages API — needs ANTHROPIC_API_KEY or an `ant`\n"
                    "               profile ([vision] extra)\n"
                    "  local      : a local transformers model — offline, no key ([vision-local]\n"
                    "               extra); set vision_model to a HF id, e.g. google/gemma-3-4b-it"
                ),
                choices=("claude-cli", "anthropic", "local"),
            ),
            _Field(
                "vision_model",
                "claude-opus-4-8",
                inline="model id — claude-cli/anthropic: claude-opus-4-8 | sonnet; local: HF id",
            ),
        ),
        comment="Enrichment — all off by default.",
    ),
    _Section(
        "logging",
        (
            _Field(
                "level",
                "info",
                inline="debug | info | warning",
                choices=("debug", "info", "warning"),
            ),
            _Field("jsonl", True, inline="tmp/logs/<date>.jsonl (async, non-blocking)"),
        ),
    ),
    _ListSection(
        "federation",
        (
            _Field("name", "", inline="label for this member store, e.g. python | rust | acme"),
            _Field("config", "", inline="path to that store's docusearch.yaml"),
        ),
        comment=(
            "OPTIONAL — federate several independent stores into one searchable set (R-TEST-3).\n"
            "List each member's name + its own config file. `serve` and `search` then fan out\n"
            "across all members (signal-level RRF merge, deduped by content hash) and rank as if\n"
            "it were one store. Scope a query to a subset with --stores (CLI) or the search_docs\n"
            "`stores` argument (MCP/AI), e.g. --stores acme to search ONLY the ACME store.\n"
            "Leave this out entirely for a normal single-store setup."
        ),
    ),
)


# ----------------------------------------------------------------- template rendering


def _fmt(value: _Value) -> str:
    """Render a Python default as a YAML scalar/flow value."""
    if isinstance(value, bool):  # bool is a subclass of int — check it first
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_fmt(item) for item in value) + "]"
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _comment(text: str, pad: str) -> Iterator[str]:
    for line in text.split("\n"):
        yield f"{pad}#" + (f" {line}" if line else "")


def _value_lines(key: str, default: _Value, pad: str, inline: str) -> list[str]:
    """Field lines. A non-empty list renders as a block sequence (easier to edit)::

        include:
          - "**/*.html"

    An empty list stays inline (``[]``); scalars stay ``key: value``.
    """
    tail = f"  # {inline}" if inline else ""
    if isinstance(default, list) and default:
        return [f"{pad}{key}:{tail}"] + [f"{pad}  - {_fmt(item)}" for item in default]
    return [f"{pad}{key}: {_fmt(default)}{tail}"]


def _field_lines(field: _Field, pad: str) -> Iterator[str]:
    if field.comment:
        yield from _comment(field.comment, pad)
    yield from _value_lines(field.key, field.default, pad, field.inline)


def _list_item_lines(fields: Sequence[_Field], pad: str) -> Iterator[str]:
    """Render one list entry: first field gets the ``- `` dash, rest align under it."""
    inner = pad + "  "
    for i, field in enumerate(fields):
        if field.comment:
            yield from _comment(field.comment, inner)
        if i == 0:  # first field carries the "- " dash (always a scalar here, e.g. type)
            tail = f"  # {field.inline}" if field.inline else ""
            yield f"{pad}- {field.key}: {_fmt(field.default)}{tail}"
        else:
            yield from _value_lines(field.key, field.default, inner, field.inline)


def render_template(version: str = __version__, generated_on: date | None = None) -> str:
    """Return the fully-commented YAML template (R-CFG-2/3). Doubles as the config spec."""
    on = generated_on or date.today()
    out: list[str] = [
        "# " + "=" * 60,
        "# docusearch configuration",
        f"# Generated {on.isoformat()} by docusearch v{version} — edit freely.",
        "# Every field is documented. Options are copy-paste ready.",
        "# " + "=" * 60,
        "",
    ]
    for node in SCHEMA:
        if isinstance(node, _Field):
            out.extend(_field_lines(node, ""))
        elif isinstance(node, _Section):
            if node.comment:
                out.extend(_comment(node.comment, ""))
            out.append(f"{node.key}:")
            for field in node.fields:
                out.extend(_field_lines(field, "  "))
        else:  # _ListSection
            if node.comment:
                out.extend(_comment(node.comment, ""))
            out.append(f"{node.key}:")
            out.extend(_list_item_lines(node.fields, "  "))
        out.append("")
    return "\n".join(out).rstrip() + "\n"


# ----------------------------------------------------------------- defaults + merge


def _default_mapping() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for node in SCHEMA:
        if isinstance(node, _Field):
            out[node.key] = copy.deepcopy(node.default)
        elif isinstance(node, _Section):
            out[node.key] = {f.key: copy.deepcopy(f.default) for f in node.fields}
        else:  # _ListSection
            out[node.key] = [{f.key: copy.deepcopy(f.default) for f in node.fields}]
    return out


def _merged(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay a validated raw mapping onto the defaults (partial entries inherit)."""
    base = _default_mapping()
    for node in SCHEMA:
        if node.key not in raw:
            continue
        val = raw[node.key]
        if isinstance(node, _Field):
            base[node.key] = val
        elif isinstance(node, _Section):
            if isinstance(val, Mapping):
                section = base[node.key]
                for field in node.fields:
                    if field.key in val:
                        section[field.key] = val[field.key]
        else:  # _ListSection
            if isinstance(val, list):
                field_defaults = {f.key: copy.deepcopy(f.default) for f in node.fields}
                merged_list: list[dict[str, Any]] = []
                for item in val:
                    entry = copy.deepcopy(field_defaults)
                    if isinstance(item, Mapping):
                        for field in node.fields:
                            if field.key in item:
                                entry[field.key] = item[field.key]
                    merged_list.append(entry)
                base[node.key] = merged_list
    return base


# ----------------------------------------------------------------- validation


def _validate(raw: Mapping[str, Any], nodes: Sequence[_Node], prefix: str = "") -> None:
    known = {node.key: node for node in nodes}
    for key, val in raw.items():
        full = f"{prefix}{key}"
        node = known.get(key)
        if node is None:
            warnings.warn(f"unknown config key {full!r} (ignored)", UserWarning, stacklevel=3)
            continue
        if isinstance(node, _Field):
            if node.choices is not None and val not in node.choices:
                accepted = ", ".join(node.choices)
                raise ConfigError(f"Invalid value {val!r} for {full!r}. Accepted: {accepted}.")
        elif isinstance(node, _Section):
            if isinstance(val, Mapping):
                _validate(val, node.fields, prefix=f"{full}.")
        else:  # _ListSection
            if isinstance(val, list):
                for i, item in enumerate(val):
                    if isinstance(item, Mapping):
                        _validate(item, node.fields, prefix=f"{full}[{i}].")


# ----------------------------------------------------------------- typed config


@dataclass(frozen=True)
class PathsConfig:
    staging_dir: str
    db_path: str
    tmp_dir: str


@dataclass(frozen=True)
class SourceConfig:
    type: str
    name: str
    version: str
    location: str
    include: list[str]
    exclude: list[str]
    content_selector: str
    strip_selectors: list[str]
    min_content_chars: int
    audience: list[str]


@dataclass(frozen=True)
class FederationMemberConfig:
    """One member store of a federation: a label plus the path to that store's own config."""

    name: str
    config: str


@dataclass(frozen=True)
class AccessConfig:
    """Who may search this document store. ``public`` = anyone on the server. ``private`` = only
    the whitelisted usernames / groups, verified from the request's username (the
    ``X-Docusearch-User`` / ``X-Docusearch-Groups`` HTTP headers). Defaults to public."""

    visibility: str  # "public" | "private"
    allowed_users: list[str]
    allowed_groups: list[str]

    def permits(self, *, user: str | None, groups: set[str]) -> bool:
        """True if a request from ``user`` (in ``groups``) may search this store."""
        if self.visibility != "private":
            return True
        if user is not None and user in self.allowed_users:
            return True
        return bool(groups & set(self.allowed_groups))


@dataclass(frozen=True)
class EmbedConfig:
    model: str
    device: str
    batch_size: int
    auto_max_mb: int
    trust_remote_code: bool


@dataclass(frozen=True)
class IndexConfig:
    chunk_tokens: int
    chunk_overlap: int
    ann: bool
    ann_m: int
    ann_ef_construction: int


@dataclass(frozen=True)
class SearchConfig:
    top_k_default: int
    rrf_k: int
    bm25_only: bool


@dataclass(frozen=True)
class ServeConfig:
    host: str
    port: int
    mcp_path: str
    self_heal_minutes: int


@dataclass(frozen=True)
class EnrichConfig:
    preflight_sample: int
    preflight_rules: str
    ai_summaries: bool
    vision_images: bool
    vision_provider: str
    vision_model: str


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    jsonl: bool


def _strs(value: Any) -> list[str]:
    return [str(item) for item in value]


@dataclass(frozen=True)
class Config:
    """The fully-resolved, validated configuration (R-CFG-1). Immutable after load."""

    mode: str
    server_url: str
    paths: PathsConfig
    sources: list[SourceConfig]
    embed: EmbedConfig
    index: IndexConfig
    search: SearchConfig
    serve: ServeConfig
    access: AccessConfig
    enrich: EnrichConfig
    logging: LoggingConfig
    federation: list[FederationMemberConfig]

    @classmethod
    def _from_mapping(cls, m: Mapping[str, Any]) -> Config:
        p, e, ix = m["paths"], m["embed"], m["index"]
        se, sv, en, lg = m["search"], m["serve"], m["enrich"], m["logging"]
        return cls(
            mode=str(m["mode"]),
            server_url=str(m["server_url"]),
            paths=PathsConfig(
                staging_dir=str(p["staging_dir"]),
                db_path=str(p["db_path"]),
                tmp_dir=str(p["tmp_dir"]),
            ),
            sources=[
                SourceConfig(
                    type=str(s["type"]),
                    name=str(s["name"]),
                    version=str(s["version"]),
                    location=str(s["location"]),
                    include=_strs(s["include"]),
                    exclude=_strs(s["exclude"]),
                    content_selector=str(s["content_selector"]),
                    strip_selectors=_strs(s["strip_selectors"]),
                    min_content_chars=int(s["min_content_chars"]),
                    audience=_strs(s["audience"]),
                )
                for s in m["sources"]
            ],
            embed=EmbedConfig(
                model=str(e["model"]),
                device=str(e["device"]),
                batch_size=int(e["batch_size"]),
                auto_max_mb=int(e["auto_max_mb"]),
                trust_remote_code=bool(e["trust_remote_code"]),
            ),
            index=IndexConfig(
                chunk_tokens=int(ix["chunk_tokens"]),
                chunk_overlap=int(ix["chunk_overlap"]),
                ann=bool(ix["ann"]),
                ann_m=int(ix["ann_m"]),
                ann_ef_construction=int(ix["ann_ef_construction"]),
            ),
            search=SearchConfig(
                top_k_default=int(se["top_k_default"]),
                rrf_k=int(se["rrf_k"]),
                bm25_only=bool(se["bm25_only"]),
            ),
            serve=ServeConfig(
                host=str(sv["host"]),
                port=int(sv["port"]),
                mcp_path=str(sv["mcp_path"]),
                self_heal_minutes=int(sv["self_heal_minutes"]),
            ),
            access=AccessConfig(
                visibility=str(m["access"]["visibility"]),
                allowed_users=_strs(m["access"]["allowed_users"]),
                allowed_groups=_strs(m["access"]["allowed_groups"]),
            ),
            enrich=EnrichConfig(
                preflight_sample=int(en["preflight_sample"]),
                preflight_rules=str(en["preflight_rules"]),
                ai_summaries=bool(en["ai_summaries"]),
                vision_images=bool(en["vision_images"]),
                vision_provider=str(en["vision_provider"]),
                vision_model=str(en["vision_model"]),
            ),
            logging=LoggingConfig(
                level=str(lg["level"]),
                jsonl=bool(lg["jsonl"]),
            ),
            federation=[
                FederationMemberConfig(name=str(f["name"]), config=str(f["config"]))
                for f in m.get("federation", [])
                if isinstance(f, Mapping) and str(f.get("name", "")).strip()
            ],
        )


# ----------------------------------------------------------------- public entry points


def default() -> Config:
    """The built-in defaults, exactly as the generated template describes them."""
    return Config._from_mapping(_default_mapping())


def write_template(path: Path | str, *, force: bool = False) -> bool:
    """Write the commented template to ``path``. Returns False if it existed (no force)."""
    path = Path(path)
    if path.exists() and not force:
        return False
    if path.parent != Path():
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_template(), encoding="utf-8")
    return True


def load(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Load config from ``path``, writing the template first if it is missing (R-CFG-2)."""
    path = Path(path)
    if not path.exists():
        write_template(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Config at {path} must be a YAML mapping, got {type(raw).__name__}.")
    _validate(raw, SCHEMA)
    return Config._from_mapping(_merged(raw))
