"""Async, structured JSONL logging that never blocks the hot path (§13, R-LOG-1..3).

Callers do ``log("event", **fields)``; the record is dropped on an in-memory queue
and a background daemon thread appends it as one JSON line to ``<log_dir>/<date>.jsonl``.
Query and ingest paths therefore never wait on disk (R-LOG-2). Everything queued is
flushed on ``close()`` and on process exit (``atexit``).

Public surface:
    RUN_ID                       -- stable id for this process
    new_run_id() -> str          -- a fresh unique run id
    RunLog                       -- explicit-lifecycle logger (context manager)
    configure(dir, ...) -> RunLog / log(event, **fields) / flush() / shutdown()
                                 -- process-wide convenience over a single RunLog
"""

from __future__ import annotations

import atexit
import json
import queue
import sys
import threading
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

_LEVELS = {"debug": 10, "info": 20, "warning": 30}


def new_run_id() -> str:
    """A unique, roughly-sortable run id, e.g. ``20260716T151500-1a2b3c4d``."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


RUN_ID = new_run_id()

_STOP = object()  # sentinel put on the queue to stop the writer thread


class RunLog:
    """A background JSONL logger. Cheap to call; the writer thread does the I/O."""

    def __init__(
        self,
        log_dir: Path | str,
        *,
        level: str = "info",
        enabled: bool = True,
        run_id: str = RUN_ID,
    ) -> None:
        self.run_id = run_id
        self.enabled = enabled
        self._dir = Path(log_dir)
        self._threshold = _LEVELS.get(level, _LEVELS["info"])
        self._queue: queue.Queue[Any] = queue.Queue()
        self._fh: TextIO | None = None
        self._closed = False
        self._thread: threading.Thread | None = None
        if enabled:
            self._thread = threading.Thread(target=self._run, name=f"runlog-{run_id}", daemon=True)
            self._thread.start()

    # -- public API --------------------------------------------------------------

    def log(self, event: str, *, level: str = "info", **fields: Any) -> None:
        """Queue one structured record. Returns immediately (R-LOG-2)."""
        if not self.enabled or self._closed:
            return
        if _LEVELS.get(level, _LEVELS["info"]) < self._threshold:
            return
        record: dict[str, Any] = {
            "run_id": self.run_id,
            "ts": datetime.now(UTC).isoformat(),
            "level": level,
            "event": event,
        }
        record.update(fields)
        self._queue.put(record)

    def flush(self) -> None:
        """Block until everything queued so far has been written."""
        if self.enabled and not self._closed:
            self._queue.join()

    def close(self) -> None:
        """Flush, stop the writer thread, and close the file."""
        if self._closed or not self.enabled:
            self._closed = True
            return
        self._closed = True
        self._queue.join()
        self._queue.put(_STOP)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def __enter__(self) -> RunLog:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- writer thread -----------------------------------------------------------

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    break
                self._write(item)
            except Exception as err:  # never let the writer thread die silently
                print(f"[runlog] write failed: {err}", file=sys.stderr)
            finally:
                self._queue.task_done()
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _write(self, record: dict[str, Any]) -> None:
        if self._fh is None:
            self._dir.mkdir(parents=True, exist_ok=True)
            path = self._dir / f"{date.today().isoformat()}.jsonl"
            self._fh = path.open("a", encoding="utf-8")
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()


# --------------------------------------------------------------- process-wide layer

_active: RunLog | None = None
_active_lock = threading.Lock()


def configure(log_dir: Path | str, *, level: str = "info", enabled: bool = True) -> RunLog:
    """Install a process-wide logger, replacing any previous one."""
    global _active
    with _active_lock:
        if _active is not None:
            _active.close()
        _active = RunLog(log_dir, level=level, enabled=enabled, run_id=RUN_ID)
        return _active


def log(event: str, *, level: str = "info", **fields: Any) -> None:
    """Log to the process-wide logger; a no-op if none is configured."""
    logger = _active
    if logger is not None:
        logger.log(event, level=level, **fields)


def flush() -> None:
    if _active is not None:
        _active.flush()


def shutdown() -> None:
    """Flush and stop the process-wide logger (also runs at exit)."""
    global _active
    with _active_lock:
        if _active is not None:
            _active.close()
            _active = None


atexit.register(shutdown)
