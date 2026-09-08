#!/usr/bin/env python3
"""
===============================================================================
 screen_trace_suite.py -- rank every benchmark in a trace suite by HBF fitness
===============================================================================

WHY THIS EXISTS
    H3 only does something interesting when a workload has a large buffer that
    is read many times and never written. Most GPU benchmarks do not. Finding
    out which ones do, one benchmark at a time with run_real_trace.sh, means
    dozens of manual invocations and no side-by-side view.

    This walks a downloaded suite, classifies every benchmark in it, and prints
    one ranked table. Screening is cheap; simulating is not. Decide what to
    simulate from this table.

WHAT IT REPORTS, PER BENCHMARK
    kernels        how many kernel traces were scanned
    touched MiB    distinct bytes the trace accessed (NOT the trace file size --
                   a small buffer read a million times makes a huge trace)
    HBF MiB        bytes in buffers that are read-only and >= --hbf-min-bytes
    HBF share      percentage of all accessed bytes that land in HBF
    split          the same share with --split-on-write, i.e. an UPPER BOUND

    'HBF share' is the number that matters. Below ~1% the benchmark cannot
    exercise H3 and simulating it wastes hours.

    'split' is the diagnostic. classify_trace.py fuses touched pages that are
    within --merge-gap (1 MiB by default) of each other into one buffer; if a
    read-only allocation sits that close to a written one, the read-only half
    loses its eligibility. When 'split' is much larger than 'HBF share', that
    fusion is hiding real read-only data and the placement heuristic -- not the
    benchmark -- is what needs fixing.

WHAT IT CONNECTS TO
    tools/classify_trace.py     imported directly (scan_kernel, merge_pages)
    scripts/run_real_trace.sh   what you run on the benchmarks this ranks well

USAGE
    python3 tools/screen_trace_suite.py /data/shivansh/traces/hw_run
    python3 tools/screen_trace_suite.py <root> --csv screen.csv --min-share 1.0
    python3 tools/screen_trace_suite.py <root> --max-kernels 20   # fast pass

CONFIGURABLE PARAMETERS
    --hbf-min-bytes   smallest buffer that may go to HBF      (default 64 KiB)
    --page-bytes      tally granularity                       (default 64 KiB)
    --merge-gap       allocation-boundary guess                (default 1 MiB)
    --max-kernels     scan only the first N kernels per benchmark. FAST BUT
                      UNSAFE: a buffer written by a later kernel looks
                      read-only, so HBF share is over-reported. Use to triage,
                      never to conclude.
    --min-share       highlight benchmarks at or above this HBF share
    --csv             also write the table as CSV

NO DOWNLOADS AND NO WRITES TO THE TRACES. Read-only; allocation maps are not
produced here (run_real_trace.sh does that for the benchmarks you select).
===============================================================================
"""

import argparse
import contextlib
import csv
import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict

from classify_trace import merge_pages, scan_kernel


def find_benchmarks(root):
    """Every directory under root holding a kernelslist(.g).

    Accel-Sim's layout is
        hw_run/<suite>/<cuda>/<app>/<args>/traces/kernelslist.g
    but nothing here depends on that shape -- we just look for the file.
    """
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in ("kernelslist.g", "kernelslist"):
            if name in filenames:
                found.append((dirpath, os.path.join(dirpath, name)))
                break
    return sorted(found)


def label_for(path, root):
    """Readable name: the path relative to root, minus a trailing /traces."""
    rel = os.path.relpath(path, root)
    parts = [p for p in rel.split(os.sep) if p != "traces"]
    return "/".join(parts) or rel


def tally(trace_dir, kernelslist, page_bytes, max_kernels):
    """Page-level read/write tallies for one benchmark. Returns (pages, n)."""
    kernels = []
    for line in open(kernelslist, errors="replace"):
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("MemcpyHtoD"):
            kernels.append(line)
    if max_kernels:
        kernels = kernels[:max_kernels]

    pages = defaultdict(lambda: [0, 0])
    scanned = 0
    for k in kernels:
        p = os.path.join(trace_dir, k)
        if not os.path.exists(p):
            continue
        # scan_kernel logs per-kernel progress to stderr; we print our own.
        with contextlib.redirect_stderr(io.StringIO()):
            scan_kernel(p, page_bytes, pages)
        scanned += 1
    return pages, scanned


def summarise(pages, page_bytes, merge_gap, hbf_min_bytes, split_on_write):
    ranges = merge_pages(pages, page_bytes, merge_gap,
                         split_on_write=split_on_write)
    if not ranges:
        return 0.0, 0.0, 0, 0.0
    hbf_bytes = hbf_read = 0
    total_read = 0
    touched = 0
    n_hbf = 0
    for r in ranges:
        size = r["end"] - r["start"]
        touched += size
        total_read += r["read"] + r["write"]
        if r["write"] == 0 and size >= hbf_min_bytes:
            n_hbf += 1
            hbf_bytes += size
            hbf_read += r["read"]
    share = (hbf_read / total_read * 100.0) if total_read else 0.0
    return share, hbf_bytes / 2 ** 20, n_hbf, touched / 2 ** 20


def main():
    ap = argparse.ArgumentParser(
        description="Rank every benchmark in a trace suite by HBF fitness.")
    ap.add_argument("root", help="directory to walk (e.g. .../hw_run)")
    ap.add_argument("--hbf-min-bytes", type=int, default=64 * 1024)
    ap.add_argument("--page-bytes", type=int, default=64 * 1024)
    ap.add_argument("--merge-gap", type=int, default=1024 * 1024)
    ap.add_argument("--max-kernels", type=int, default=0,
                    help="scan only the first N kernels per benchmark "
                         "(triage only -- over-reports HBF share)")
    ap.add_argument("--min-share", type=float, default=1.0,
                    help="highlight benchmarks at or above this HBF share")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"ERROR: not a directory: {args.root}", file=sys.stderr)
        return 1

    benches = find_benchmarks(args.root)
    if not benches:
        print(f"ERROR: no kernelslist found anywhere under {args.root}\n"
              f"       Extract a suite first: tar -xzf <suite>.tgz",
              file=sys.stderr)
        return 1

    print(f"Screening {len(benches)} benchmarks under {args.root}")
    if args.max_kernels:
        print(f"  FAST PASS: first {args.max_kernels} kernels only. HBF share is")
        print(f"  an over-estimate -- a later kernel may write a buffer that")
        print(f"  looks read-only here. Re-run without --max-kernels to confirm.")
    print()

    rows = []
    for i, (d, kl) in enumerate(benches, 1):
        name = label_for(d, args.root)
        t0 = time.time()
        print(f"  [{i}/{len(benches)}] {name} ...", end="", flush=True)
        try:
            pages, n_kernels = tally(d, kl, args.page_bytes, args.max_kernels)
        except (OSError, MemoryError) as e:
            print(f" SKIPPED ({e.__class__.__name__}: {e})")
            continue
        if not pages:
            print(" no memory accesses")
            continue
        share, hbf_mib, n_hbf, touched = summarise(
            pages, args.page_bytes, args.merge_gap, args.hbf_min_bytes, False)
        split, _, _, _ = summarise(
            pages, args.page_bytes, args.merge_gap, args.hbf_min_bytes, True)
        rows.append({"benchmark": name, "kernels": n_kernels,
                     "touched_mib": touched, "hbf_buffers": n_hbf,
                     "hbf_mib": hbf_mib, "hbf_share_pct": share,
                     "hbf_share_split_pct": split})
        print(f" {share:5.1f}%  ({time.time() - t0:.0f}s)")

    if not rows:
        print("\nNothing scanned successfully.")
        return 1

    rows.sort(key=lambda r: r["hbf_share_pct"], reverse=True)

    print()
    print("=" * 92)
    print("  HBF fitness -- ranked. 'share' is the fraction of accessed bytes")
    print("  in read-only buffers >= %d KiB. 'split' is the upper bound with"
          % (args.hbf_min_bytes // 1024))
    print("  range-fusion disabled (see --split-on-write in classify_trace.py).")
    print("=" * 92)
    print(f"  {'benchmark':<46} {'kern':>5} {'touched':>10} {'HBF MiB':>9} "
          f"{'share':>7} {'split':>7}")
    print("  " + "-" * 88)
    for r in rows:
        flag = " *" if r["hbf_share_pct"] >= args.min_share else "  "
        print(f"  {r['benchmark'][:46]:<46} {r['kernels']:>5} "
              f"{r['touched_mib']:>9.1f}M {r['hbf_mib']:>8.1f}M "
              f"{r['hbf_share_pct']:>6.1f}% {r['hbf_share_split_pct']:>6.1f}%{flag}")

    good = [r for r in rows if r["hbf_share_pct"] >= args.min_share]
    hidden = [r for r in rows
              if r["hbf_share_split_pct"] - r["hbf_share_pct"] >= 5.0]

    print()
    print(f"  {len(good)} of {len(rows)} benchmarks reach {args.min_share}% HBF share.")
    if good:
        print("  Simulate these:")
        for r in good[:10]:
            print(f"    ./scripts/run_real_trace.sh {os.path.join(args.root, r['benchmark'])} --cycles 200000")
    else:
        print("  None. No benchmark here can exercise H3's HBF path, so none is")
        print("  worth simulating. That is a result worth reporting, not a failure.")

    if hidden:
        print()
        print(f"  {len(hidden)} benchmarks show a much higher upper bound than actual")
        print("  share, meaning --merge-gap is fusing read-only buffers into written")
        print("  neighbours. Re-check these with a smaller --merge-gap:")
        for r in hidden[:10]:
            print(f"    {r['benchmark']}  {r['hbf_share_pct']:.1f}% -> "
                  f"{r['hbf_share_split_pct']:.1f}%")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  wrote {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
