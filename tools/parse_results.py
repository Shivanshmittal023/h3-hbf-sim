#!/usr/bin/env python3
"""
===============================================================================
 parse_results.py -- compare H3 and HBM-only simulation results
===============================================================================

WHAT THIS IS
    Parses the logs produced by scripts/run_smoke_test.sh and
    scripts/run_smoke_test_hbm_only.sh and prints a side-by-side comparison
    table: metric | HBM-only | H3 | difference.

WHAT IT CONNECTS TO
    * results/smoke_test/sim.log              (H3)
    * results/smoke_test_hbm_only/sim.log     (baseline)
    * results/smoke_test_no_lhb/sim.log       (optional LHB ablation)
    * traces/<dir>/manifest.json              expected access counts, used to
      verify the simulator actually saw the traffic the generator emitted

PARSING
    GPGPU-Sim and the H3 components both emit statistics as `key = value`
    lines, which is why h3_memory_backend.cpp deliberately uses that style:
    one parser handles both, and util/job_launching/get_stats.py picks the H3
    counters up with no changes.

USAGE
    python3 tools/parse_results.py
    python3 tools/parse_results.py --results-dir results --json out.json
===============================================================================
"""

import argparse
import json
import os
import re
import sys

STAT_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.\[\]]*)\s*=\s*(-?[0-9.eE+-]+)\s*$")


def parse_log(path):
    """Collect every `key = value` statistic. Later values win (final totals)."""
    stats, meta = {}, {"path": path, "exists": os.path.exists(path)}
    if not meta["exists"]:
        return stats, meta
    text_flags = {"exit_detected": False, "assertion": False}
    with open(path, errors="replace") as f:
        for line in f:
            if "exit detected" in line:
                text_flags["exit_detected"] = True
            if re.search(r"assert|Segmentation fault|terminate called", line, re.I):
                text_flags["assertion"] = True
            m = STAT_RE.match(line)
            if m:
                key, val = m.group(1), m.group(2)
                try:
                    stats[key] = float(val)
                except ValueError:
                    pass
    meta.update(text_flags)
    return stats, meta


def read_aux(d):
    out = {}
    for name, fn in (("exit_code", int), ("wall_seconds", float)):
        p = os.path.join(d, name)
        if os.path.exists(p):
            try:
                out[name] = fn(open(p).read().strip())
            except ValueError:
                pass
    return out


def fmt(v, nd=2):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v != v:                      # NaN
            return "n/a"
        if abs(v) >= 1e6:
            return f"{v:,.0f}"
        if abs(v) >= 1000 or v == int(v):
            return f"{int(v):,}"
        return f"{v:.{nd}f}"
    return str(v)


def delta(base, new):
    """Human-readable difference, with a ratio when both are meaningful."""
    if base is None or new is None:
        return "n/a"
    if base == 0:
        return "+inf" if new else "0"
    ratio = new / base
    return f"{ratio:.3f}x  ({(ratio - 1) * 100:+.1f}%)"


# Rows: (label, key, kind)
#   kind 'lower_better' annotates which direction is good.
ROWS = [
    ("--- Execution ---", None, None),
    ("Simulated cycles", "gpu_tot_sim_cycle", "lower_better"),
    ("Instructions", "gpu_tot_sim_insn", None),
    ("IPC", "gpu_tot_ipc", "higher_better"),
    ("--- Requests served ---", None, None),
    ("Total memory requests", "gpgpu_n_mem_read_global", None),
    ("L2 accesses", "L2_total_cache_accesses", None),
    ("L2 misses", "L2_total_cache_misses", None),
    ("--- H3 routing ---", None, None),
    ("Requests -> HBM", "h3_router_requests_hbm", None),
    ("Requests -> HBF", "h3_router_requests_hbf", None),
    ("HBF request fraction", "h3_router_hbf_request_fraction", None),
    ("D2D hops", "h3_router_d2d_hops", None),
    ("--- Bandwidth ---", None, None),
    ("HBM bandwidth (GB/s)", "h3_router_hbm_bandwidth_gbps", "higher_better"),
    ("HBF bandwidth (GB/s)", "h3_router_hbf_bandwidth_gbps", "higher_better"),
    ("HBM utilisation", "h3_router_hbm_utilization", None),
    ("HBF utilisation", "h3_router_hbf_utilization", None),
    ("--- Latency Hiding Buffer ---", None, None),
    ("LHB hit rate", "lhb_hit_rate", "higher_better"),
    ("LHB misses (late)", "lhb_misses_late", "lower_better"),
    ("LHB misses (absent)", "lhb_misses_absent", "lower_better"),
    ("LHB avg stall/miss (ns)", "lhb_avg_stall_ns_per_miss", "lower_better"),
    ("LHB utilisation", "lhb_avg_utilization", None),
    ("LHB prefetch efficiency", "lhb_prefetch_efficiency", "higher_better"),
    ("--- Correctness (must be 0) ---", None, None),
    ("Writes to read-only HBF", "h3_router_write_attempts_to_hbf", "must_be_zero"),
    ("Unmapped requests", "h3_router_unmapped_requests", "must_be_zero"),
    ("Backend-full rejects", "h3_router_backend_full_rejects", "must_be_zero"),
    ("Backend avg latency (ns)", "h3_backend_avg_latency_ns", None),
]


def main():
    ap = argparse.ArgumentParser(
        description="Compare H3 and HBM-only smoke test results.")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--h3", default=None, help="override the H3 results dir")
    ap.add_argument("--baseline", default=None, help="override the baseline results dir")
    ap.add_argument("--trace-dir", default="traces/synthetic_h3_smoke_test")
    ap.add_argument("--json", default=None, help="also write the comparison as JSON")
    args = ap.parse_args()

    h3_dir = args.h3 or os.path.join(args.results_dir, "smoke_test")
    bl_dir = args.baseline or os.path.join(args.results_dir, "smoke_test_hbm_only")
    ab_dir = os.path.join(args.results_dir, "smoke_test_no_lhb")

    h3_stats, h3_meta = parse_log(os.path.join(h3_dir, "sim.log"))
    bl_stats, bl_meta = parse_log(os.path.join(bl_dir, "sim.log"))
    ab_stats, ab_meta = parse_log(os.path.join(ab_dir, "sim.log"))
    h3_meta.update(read_aux(h3_dir))
    bl_meta.update(read_aux(bl_dir))

    if not h3_meta["exists"] and not bl_meta["exists"]:
        print("No results found.", file=sys.stderr)
        print(f"  looked for {h3_dir}/sim.log and {bl_dir}/sim.log", file=sys.stderr)
        print("Run ./scripts/run_smoke_test.sh and "
              "./scripts/run_smoke_test_hbm_only.sh first.", file=sys.stderr)
        return 1

    W = 62
    print("=" * W)
    print("  H3 vs HBM-only -- smoke test comparison")
    print("=" * W)

    # ---- Run health -------------------------------------------------------
    for label, meta in (("HBM-only", bl_meta), ("H3", h3_meta)):
        if not meta["exists"]:
            print(f"  {label:10s}: NOT RUN ({meta['path']})")
            continue
        status = []
        rc = meta.get("exit_code")
        status.append("exit=0" if rc == 0 else f"exit={rc}")
        if meta.get("exit_detected"):
            status.append("clean-exit")
        elif rc == 0:
            status.append("hit cycle limit")
        else:
            status.append("FAILED before completion")
        if meta.get("assertion"):
            status.append("ASSERTION/CRASH")
        wall = meta.get("wall_seconds")
        if wall is not None:
            status.append(f"{wall:.0f}s wall")
        print(f"  {label:10s}: {', '.join(status)}")

    # ---- Trace expectations ----------------------------------------------
    man_path = os.path.join(args.trace_dir, "manifest.json")
    if os.path.exists(man_path):
        man = json.load(open(man_path))
        exp = man.get("expected_accesses", {})
        print()
        print(f"  Trace: {man.get('num_layers')} layers, "
              f"{man.get('kernels')} kernels, "
              f"expected HBF access fraction "
              f"{man.get('expected_hbf_access_fraction', 0):.3f}")
        got = h3_stats.get("h3_router_hbf_request_fraction")
        if got is not None:
            want = man.get("expected_hbf_access_fraction", 0)
            ok = abs(got - want) < 0.15
            print(f"  Observed HBF fraction: {got:.3f} "
                  f"({'matches the trace' if ok else 'DIVERGES from the trace'})")

    # ---- Comparison table -------------------------------------------------
    cyc_b, cyc_h = bl_stats.get("gpu_tot_sim_cycle"), h3_stats.get("gpu_tot_sim_cycle")
    if cyc_b and cyc_h and abs(cyc_b - cyc_h) / max(cyc_b, 1) < 0.01:
        print()
        print("  NOTE: both runs stopped at the same cycle limit, so 'Simulated")
        print("        cycles' is equal by construction. Read INSTRUCTIONS and IPC")
        print("        instead -- they measure work completed in a fixed budget.")
    print()
    print(f"  {'Metric':<28}{'HBM-only':>12}{'H3':>14}{'Difference':>20}")
    print("  " + "-" * 72)
    rows_out = {}
    for label, key, kind in ROWS:
        if key is None:
            print(f"  {label}")
            continue
        b = bl_stats.get(key)
        h = h3_stats.get(key)
        if b is None and h is None:
            continue
        note = ""
        if kind == "must_be_zero" and h:
            note = "  <-- ERROR"
        d = delta(b, h) if (b is not None and h is not None) else "-"
        print(f"  {label:<28}{fmt(b):>12}{fmt(h):>14}{d:>20}{note}")
        rows_out[key] = {"baseline": b, "h3": h}

    # ---- Optional ablation -----------------------------------------------
    if ab_meta["exists"]:
        print()
        print("  LHB ablation (H3 with the buffer disabled)")
        print("  " + "-" * 72)
        for label, key in (("Simulated cycles", "gpu_tot_sim_cycle"),
                           ("Instructions", "gpu_tot_sim_insn"),
                           ("IPC", "gpu_tot_ipc"),
                           ("LHB hit rate", "lhb_hit_rate"),
                           ("LHB avg stall/miss (ns)", "lhb_avg_stall_ns_per_miss")):
            a, h = ab_stats.get(key), h3_stats.get(key)
            print(f"  {label:<28}{fmt(a):>12}{fmt(h):>14}{delta(a, h):>20}")

        # CHOOSE THE RIGHT METRIC. When both runs stop at the same cycle cap,
        # comparing cycles is meaningless -- they are equal by construction.
        # The honest comparison is then WORK DONE in that fixed budget.
        cyc_ab = ab_stats.get("gpu_tot_sim_cycle")
        cyc_h3 = h3_stats.get("gpu_tot_sim_cycle")
        ins_ab = ab_stats.get("gpu_tot_sim_insn")
        ins_h3 = h3_stats.get("gpu_tot_sim_insn")
        capped = (cyc_ab and cyc_h3 and abs(cyc_ab - cyc_h3) / max(cyc_ab, 1) < 0.01)
        print()
        if capped and ins_ab and ins_h3:
            print(f"  Both runs stopped at the same cycle limit ({fmt(cyc_h3)}), so"
                  f" cycles cannot")
            print(f"  distinguish them. Comparing work completed in that budget:")
            print(f"    -> the LHB delivers {ins_h3 / ins_ab:.1f}x more instructions"
                  f" ({fmt(ins_ab)} -> {fmt(ins_h3)})")
        elif cyc_ab and cyc_h3:
            print(f"  -> the LHB accounts for a {cyc_ab / cyc_h3:.2f}x speedup"
                  f" on this trace")

    # ---- Verdict ----------------------------------------------------------
    print()
    print("=" * W)
    problems = []
    if h3_stats.get("h3_router_write_attempts_to_hbf"):
        problems.append("writes reached the read-only HBF region")
    if h3_stats.get("h3_router_unmapped_requests"):
        problems.append("addresses fell outside the H3 address map")
    if h3_meta.get("assertion"):
        problems.append("assertion or crash in the H3 run")
    if h3_meta.get("exit_code") not in (0, None):
        problems.append(f"H3 run exited with code {h3_meta['exit_code']}")

    if problems:
        print("  VERDICT: PROBLEMS FOUND")
        for p in problems:
            print(f"    - {p}")
    else:
        print("  VERDICT: pipeline healthy (no routing or placement errors)")
    print("=" * W)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"baseline": bl_stats, "h3": h3_stats,
                       "ablation": ab_stats, "rows": rows_out,
                       "problems": problems}, f, indent=2)
        print(f"\n  wrote {args.json}")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
