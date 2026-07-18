#!/usr/bin/env python3
r"""
run_benchmark.py — sweep the embedding ``batch_size`` and find where the GPU
saturates.

For each batch size it: rewrites ``batch_size:`` in the config with ``sed``,
runs the command, starts a timer when a trigger string appears, samples GPU
utilization for a fixed window, records how many chunks were embedded in that
window, then shuts the command down. Results are appended to a CSV whose header
is ``time, batch_size, embed_count, embed_per_sec, gpu_peak%, gpu_avg%,
gpu_min%, model`` (``embed_per_sec`` is ``embed_count / time``; the trailing
``model`` column records the ``embed.model:`` configured in the YAML; columns
are space-padded so rows line up). By
default new rows are appended to any existing CSV so older runs are retained
(clean them up yourself); pass ``--overwrite-file`` to truncate first. The GPU
minimum ignores a warmup window (``--gpu-warmup``, default 3s) because the GPU
reads near-zero until the workload spins up.

By default it walks 1,2,4,8,16,32,... and auto-stops when any of these hits:
average GPU% >= 95, or embed_count is static/declining for 5 checks in a row.
A SIGINT/SIGTERM (Ctrl-C) cancels gracefully after the current collection is
recorded; a second one aborts immediately. Both the batch-size list
(``--batch-sizes``) and auto-stop (``--no-auto-stop``) are configurable.

Defaults reproduce the requested benchmark:
    command      : docusearch ingest --force
    config       : docusearch.yaml            (its batch_size: line is swept)
    start text   : "  embed:"                 (starts the timer)
    regex        : embed:\s*(\d+)/(\d+)        (e.g. "embed: 996/341769")
    window       : 30 seconds after the start text
    batch sizes  : 1,2,4,8,16,32,64,128,256   (auto-stop trims the tail)

Notes on design:
  * The child runs under a pseudo-terminal so its isatty() is true; progress
    tools that throttle when piped (docusearch drops to one line per 10%) keep
    emitting the frequent \r-style updates this reader parses. We split the
    stream on BOTH '\r' and '\n' and keep only the LAST regex match.
  * GPU sampling auto-detects the backend: nvidia-smi, else Apple-Silicon
    ioreg (no sudo), else none.
"""

import argparse
import contextlib
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import pty  # POSIX only; lets the child think it's on a terminal
except ImportError:  # pragma: no cover - Windows
    pty = None

GPU_SAMPLE_INTERVAL = 0.5  # seconds between GPU samples / status updates

# CSV columns, in order. ``model`` is last and left-aligned (variable length);
# the rest are right-justified to the widths below so rows line up under their
# headers. Numeric formats: time %4.1f, batch_size %4d, embed_count %7d,
# percentages %.1f — the widths here are the max of that and the header label.
_CSV_HEADERS = [
    "time",
    "batch_size",
    "embed_count",
    "embed_per_sec",
    "gpu_peak%",
    "gpu_avg%",
    "gpu_min%",
    "model",
]
_CSV_WIDTHS = [4, 10, 11, 13, 9, 8, 8, 0]  # 0 -> no padding (last column)


def _fmt_csv_row(cells):
    """Join pre-stringified ``cells`` with ", ", right-justified per column."""
    return ", ".join(
        cell.rjust(w) if w else cell for cell, w in zip(cells, _CSV_WIDTHS, strict=True)
    )


def _csv_cells(elapsed, batch_size, embed_count, peak, avg, lo, model):
    """Format one data row's raw values into aligned cell strings.

    ``peak``/``avg``/``lo`` may be None (rendered as empty cells); percentages
    all print with one decimal.
    """

    def pct(v):
        return "" if v is None else f"{float(v):.1f}"

    eps = embed_count / elapsed if elapsed else 0.0
    return [
        f"{elapsed:.1f}",
        f"{batch_size:d}",
        f"{embed_count:d}",
        f"{eps:.1f}",
        pct(peak),
        pct(avg),
        pct(lo),
        model,
    ]


_GPU_BACKEND = None  # resolved once on first sample: "nvidia" | "apple" | "none"


def _gpu_nvidia():
    """Max GPU utilization (%) from nvidia-smi, or None if it's absent."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    vals = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    return max(vals) if vals else None


def _gpu_apple():
    """Apple-Silicon GPU busy % from ioreg (no sudo), or None if unavailable.

    The Metal GPU (e.g. AGXAcceleratorG17X) publishes a live counter under
    IOAccelerator -> PerformanceStatistics -> "Device Utilization %".
    """
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator", "-w", "0"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    vals = [int(m) for m in re.findall(r'"Device Utilization %"=(\d+)', out.stdout)]
    return max(vals) if vals else None


def sample_gpu_percent():
    """Max GPU utilization (%) via whichever backend this box has, else None."""
    global _GPU_BACKEND
    if _GPU_BACKEND is None:
        for name, fn in (("nvidia", _gpu_nvidia), ("apple", _gpu_apple)):
            if fn() is not None:
                _GPU_BACKEND = name
                break
        else:
            _GPU_BACKEND = "none"
    if _GPU_BACKEND == "nvidia":
        return _gpu_nvidia()
    if _GPU_BACKEND == "apple":
        return _gpu_apple()
    return None


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_match = None  # matched text, group(0)
        self.last_groups = None  # capture groups of the last match
        self.trigger_time = None  # monotonic time the start text was seen
        self.started = threading.Event()
        self.proc_done = threading.Event()


def reader_thread(read_fd, state, start_text, pattern):
    """Read the child output, note the start trigger and the latest regex match."""
    fd = read_fd
    buf = b""

    def handle(text):
        if not text:
            return
        if not state.started.is_set() and start_text in text:
            with state.lock:
                if state.trigger_time is None:
                    state.trigger_time = time.monotonic()
            state.started.set()
        m = pattern.search(text)
        if m:
            with state.lock:
                state.last_match = m.group(0).strip()
                state.last_groups = m.groups()

    while True:
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            break
        if not chunk:  # EOF
            break
        buf += chunk
        parts = re.split(rb"[\r\n]", buf)  # handle \r-style progress reprints
        buf = parts.pop()  # last element is an incomplete line
        for p in parts:
            handle(p.decode("utf-8", "replace"))
    if buf:  # flush trailing partial line
        handle(buf.decode("utf-8", "replace"))
    state.proc_done.set()


def shutdown(proc):
    """Stop the child (and its group): SIGINT, then SIGTERM, then SIGKILL."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=5)


def run_window(argv, start_text, pattern, window, warmup=3.0):
    """Run ``argv``, wait for ``start_text``, sample GPU for ``window`` seconds.

    ``peak``/``avg`` cover every sample; ``min`` ignores the first ``warmup``
    seconds, since the GPU reads near-zero until the workload spins up and that
    would otherwise dominate the minimum.

    Returns a dict: ``saw_start`` (bool), ``elapsed`` (s), ``peak``/``avg``/``min``
    GPU% (None if no backend / no post-warmup sample), ``n_samples``,
    ``embed_count`` (last matched current count, None if unmatched), ``total``,
    ``exited_early`` (bool).
    """
    # A pty makes the child's isatty() true; both fds share the slave, so the
    # child's stdout/stderr are merged for us.
    master_fd = slave_fd = None
    if pty is not None:
        master_fd, slave_fd = pty.openpty()
        popen_kwargs = dict(stdout=slave_fd, stderr=slave_fd)
    else:  # pragma: no cover - Windows fallback
        popen_kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)

    proc = subprocess.Popen(
        argv,
        start_new_session=True,  # own process group -> clean group shutdown
        **popen_kwargs,
    )
    if slave_fd is not None:
        os.close(slave_fd)  # parent keeps only the master (read) end
    read_fd = master_fd if master_fd is not None else proc.stdout.fileno()

    state = State()
    reader = threading.Thread(
        target=reader_thread,
        args=(read_fd, state, start_text, pattern),
        daemon=True,
    )
    reader.start()

    samples = []  # every GPU sample -> peak, avg
    warm_samples = []  # samples after `warmup` s -> min (cold GPU excluded)
    saw_start = False
    exited_early = False
    t0 = time.monotonic()
    try:
        while not state.started.is_set():
            if state.proc_done.is_set():
                proc.wait()
                break
            time.sleep(0.05)
        else:
            saw_start = True

        if saw_start:
            sample_gpu_percent()  # resolve the GPU backend
            t0 = state.trigger_time or time.monotonic()
            while True:
                elapsed = time.monotonic() - t0
                if elapsed >= window:
                    break
                if state.proc_done.is_set():
                    exited_early = True
                    break
                g = sample_gpu_percent()
                if g is not None:
                    samples.append(g)
                    if elapsed >= warmup:
                        warm_samples.append(g)
                peak = max(samples) if samples else None
                lo = min(warm_samples) if warm_samples else None
                with state.lock:
                    last = state.last_match
                gpu_str = f"{g}%" if g is not None else "n/a"
                peak_str = f"{peak}%" if peak is not None else "n/a"
                min_str = f"{lo}%" if lo is not None else "warmup" if elapsed < warmup else "n/a"
                sys.stderr.write(
                    f"\r[bench] t={elapsed:5.1f}s  gpu={gpu_str:>4}  "
                    f"peak={peak_str:>4}  min={min_str:>6}  last={last or '-'}   "
                )
                sys.stderr.flush()
                time.sleep(GPU_SAMPLE_INTERVAL)
            sys.stderr.write("\n")
    finally:
        shutdown(proc)

    with state.lock:
        groups = state.last_groups
    cur = tot = None
    if groups and len(groups) >= 2 and groups[0] and groups[1]:
        try:
            cur, tot = int(groups[0]), int(groups[1])
        except ValueError:
            cur = tot = None
    return {
        "saw_start": saw_start,
        "elapsed": min(time.monotonic() - t0, window) if saw_start else 0.0,
        "peak": max(samples) if samples else None,
        "avg": (sum(samples) / len(samples)) if samples else None,
        "min": min(warm_samples) if warm_samples else None,
        "n_samples": len(samples),
        "embed_count": cur,
        "total": tot,
        "exited_early": exited_early,
    }


def read_model(config_path):
    """Return the configured ``embed.model:`` string, or "" if not found.

    Matches the first non-comment ``model:`` line (comment lines start with
    ``#`` and are skipped) and strips surrounding quotes plus any trailing
    inline comment.
    """
    text = Path(config_path).read_text(encoding="utf-8")
    m = re.search(r"^[ \t]*model:[ \t]*(.+?)[ \t]*$", text, re.MULTILINE)
    if not m:
        return ""
    value = m.group(1)
    # drop an inline comment on an unquoted value ("model: none  # ...")
    if not value.startswith(("'", '"')):
        value = value.split("#", 1)[0].strip()
    return value.strip().strip("'\"")


def set_batch_size(config_path, value):
    """Rewrite the ``batch_size:`` line of the YAML in place, via sed.

    Preserves indentation and any trailing comment; only the integer changes.
    """
    subprocess.run(
        [
            "sed",
            "-i.bak",
            "-E",
            rf"s/^([[:space:]]*batch_size:[[:space:]]*)[0-9]+/\1{value}/",
            config_path,
        ],
        check=True,
    )
    bak = Path(config_path + ".bak")  # -i.bak is the portable (BSD+GNU) in-place form
    if bak.exists():
        bak.unlink()


def main():
    ap = argparse.ArgumentParser(
        description="Sweep embedding batch_size and record throughput vs GPU%%; "
        "auto-stop at GPU saturation. CSV: time, batch_size, "
        "embed_count, embed_per_sec, gpu_peak%%, gpu_avg%%, "
        "gpu_min%%, model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "-t",
        "--time",
        type=float,
        default=30.0,
        help="seconds to sample per batch size, after the start text",
    )
    ap.add_argument(
        "--gpu-warmup",
        type=float,
        default=3.0,
        help="seconds to ignore before accumulating the GPU minimum "
        "(the GPU reads near-zero until the workload spins up)",
    )
    ap.add_argument(
        "-s", "--start-text", default="  embed:", help="substring that starts the timer"
    )
    ap.add_argument(
        "-r",
        "--regex",
        default=r"embed:\s*(\d+)/(\d+)",
        help="regex to extract the embed count (last match is used)",
    )
    ap.add_argument(
        "-c",
        "--command",
        default="docusearch ingest --force",
        help="command to run for each batch size",
    )
    ap.add_argument(
        "--config", default="docusearch.yaml", help="YAML file whose batch_size: line is swept"
    )
    ap.add_argument(
        "-b",
        "--batch-sizes",
        default="1,2,4,8,16,32,64,128,256",
        help="comma-separated batch sizes to try, in order",
    )
    ap.add_argument("-o", "--csv", default="tmp/benchmark-batchsize.csv", help="CSV output path")
    ap.add_argument(
        "--overwrite-file",
        action="store_true",
        help="truncate the CSV before writing; default appends so older "
        "entries are retained (clean them up yourself)",
    )
    ap.add_argument(
        "--auto-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop early at avg-GPU saturation or flat/declining throughput",
    )
    ap.add_argument(
        "--gpu-avg-stop", type=float, default=99.0, help="auto-stop once average GPU%% reaches this"
    )
    ap.add_argument(
        "--flat-streak",
        type=int,
        default=5,
        help="auto-stop after this many checks in a row where embed_count is static or declining",
    )
    ap.add_argument(
        "--flat-ratio",
        type=float,
        default=1.05,
        help="a check counts as improving only if embed_count beats the "
        "previous by this factor (1.05 = over 5%% gain); otherwise it "
        "is static/declining toward the --flat-streak limit",
    )
    args = ap.parse_args()

    pattern = re.compile(args.regex)
    argv = shlex.split(args.command)
    config_path = args.config

    try:
        batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    except ValueError:
        print(f"[bench] bad --batch-sizes: {args.batch_sizes!r}", file=sys.stderr)
        return 2
    if not batch_sizes:
        print("[bench] no batch sizes given", file=sys.stderr)
        return 2
    if not Path(config_path).exists():
        print(f"[bench] config not found: {config_path!r}", file=sys.stderr)
        return 2

    original = Path(config_path).read_bytes()  # restored verbatim afterwards
    model = read_model(config_path)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Append by default (retain older entries); --overwrite-file truncates. Write
    # the header only when starting a fresh/empty file.
    append = not args.overwrite_file and csv_path.exists() and csv_path.stat().st_size > 0
    open_mode = "a" if append else "w"

    print(
        f"[bench] sweeping batch_size {batch_sizes} in {config_path} (model={model or 'n/a'})",
        file=sys.stderr,
    )
    print(
        f"[bench] auto-stop={'on' if args.auto_stop else 'off'}  "
        f"-> {csv_path} ({'append' if append else 'overwrite'})",
        file=sys.stderr,
    )

    # First SIGINT/SIGTERM: finish the current collection, then stop cleanly.
    # A second one aborts immediately (raises into the loop below).
    cancel = {"requested": False, "signame": ""}

    def _on_signal(signum, _frame):
        if cancel["requested"]:
            raise KeyboardInterrupt  # second signal -> abort now
        cancel["requested"] = True
        cancel["signame"] = signal.Signals(signum).name
        sys.stderr.write(
            f"\n[bench] {cancel['signame']} received — will cancel after the current "
            f"benchmark collection (signal again to abort now).\n"
        )
        sys.stderr.flush()

    prev_int = signal.signal(signal.SIGINT, _on_signal)
    prev_term = signal.signal(signal.SIGTERM, _on_signal)

    prev_embed = None
    flat_streak = 0
    stop_reason = None
    try:
        with csv_path.open(open_mode, encoding="utf-8", newline="") as fh:
            if not append:
                fh.write(_fmt_csv_row(_CSV_HEADERS) + "\n")
                fh.flush()
            for bs in batch_sizes:
                set_batch_size(config_path, bs)
                print(f"\n[bench] === batch_size={bs} ===", file=sys.stderr)
                res = run_window(argv, args.start_text, pattern, args.time, warmup=args.gpu_warmup)

                if not res["saw_start"]:
                    print(
                        f"[bench] batch_size={bs}: start text never appeared "
                        f"(command exited or wrong --start-text); stopping.",
                        file=sys.stderr,
                    )
                    stop_reason = "start text never appeared"
                    break

                embed = res["embed_count"] or 0
                avg = res["avg"]
                fh.write(
                    _fmt_csv_row(
                        _csv_cells(res["elapsed"], bs, embed, res["peak"], avg, res["min"], model)
                    )
                    + "\n"
                )
                fh.flush()

                peak_cell = "" if res["peak"] is None else str(res["peak"])
                avg_cell = "" if avg is None else f"{avg:.1f}"
                min_cell = "" if res["min"] is None else str(res["min"])

                # A check "improves" only if it beats the previous by --flat-ratio;
                # otherwise it's static/declining and grows the streak.
                if prev_embed is None or embed > prev_embed * args.flat_ratio:
                    flat_streak = 0
                else:
                    flat_streak += 1
                prev_embed = embed

                print(
                    f"[bench] batch_size={bs}: embed_count={embed}  "
                    f"gpu peak={peak_cell or 'n/a'}% avg={avg_cell or 'n/a'}% "
                    f"min={min_cell or 'n/a'}%  flat_streak={flat_streak}/"
                    f"{args.flat_streak}  ({res['elapsed']:.1f}s"
                    f"{', exited early' if res['exited_early'] else ''})",
                    file=sys.stderr,
                )

                # User cancel wins, and only after the current collection is recorded.
                if cancel["requested"]:
                    stop_reason = f"cancelled by user ({cancel['signame']})"
                    break
                if args.auto_stop:
                    if avg is not None and avg >= args.gpu_avg_stop:
                        stop_reason = f"avg GPU {avg:.1f}% >= {args.gpu_avg_stop:g}%"
                        break
                    if flat_streak >= args.flat_streak:
                        stop_reason = (
                            f"embed_count static/declining for {flat_streak} checks in a row"
                        )
                        break
    except KeyboardInterrupt:
        stop_reason = f"aborted by user ({cancel['signame'] or 'interrupt'})"
        print("\n[bench] aborted.", file=sys.stderr)
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)
        Path(config_path).write_bytes(original)
        print(f"[bench] restored {config_path}", file=sys.stderr)

    if stop_reason:
        print(f"[bench] stopped: {stop_reason}", file=sys.stderr)
    print(f"[bench] done -> {csv_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
