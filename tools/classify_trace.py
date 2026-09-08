#!/usr/bin/env python3
"""
===============================================================================
 classify_trace.py -- decide which parts of a real trace belong in HBF vs HBM
===============================================================================

WHY THIS EXISTS
    A real Accel-Sim trace uses whatever virtual addresses CUDA happened to
    allocate (typically around 0x7f35ab700000). The H3 router expects data to
    live in fixed regions:

        HBM  0x0          .. 0x3000000000
        HBF  0x3000000000 .. 0x33000000000

    So every request from a real trace falls outside both regions, the router
    reports 100% unmapped, and the HBF path is never exercised. That is why the
    270 MB of rodinia traces already on disk cannot be used with H3 as-is.

    This tool removes that blocker WITHOUT rewriting the traces. It works out
    which buffers a trace actually uses, decides where each one belongs, and
    writes a small address map the router loads at run time.

THE CLASSIFICATION RULE
    H3 places data by what it IS, not where it sits:

        large AND never written   -> HBF   (model weights, shared KV cache)
        everything else           -> HBM   (generated KV cache, activations)

    That is exactly the decision H3 makes in reality. Here it is DERIVED from
    the trace -- a buffer goes to HBF only if the trace never stores to it --
    rather than assumed.

HOW BUFFERS ARE FOUND
    Not from the MemcpyHtoD lines alone: those only cover host-to-device
    copies, so output-only buffers (allocated and written, never copied in)
    would be missed entirely. Instead every address the trace touches is
    bucketed by page, adjacent pages are merged into ranges, and each range is
    tallied for reads and writes. MemcpyHtoD entries are used as corroborating
    evidence and for naming.

WHAT IT CONNECTS TO
    * Reads   : any Accel-Sim trace directory (kernelslist.g + kernel-*.traceg)
    * Writes  : an address-map YAML consumed by h3_address_router.cpp
                (H3RouterConfig::allocation_map)
    * Mirrors : tools/validate_trace.py, which shares the line grammar

USAGE
    python3 tools/classify_trace.py hw_run/rodinia_2.0-ft/9.1/backprop-*/*/traces \
        -o configs/generated/rodinia_backprop_map.yaml

    # then point the backend at it:
    #   configs/h3_router_config.yaml -> address_map.allocation_map: <that file>

CONFIGURABLE PARAMETERS
    --hbf-min-bytes   smallest buffer that may go to HBF   (default 64 KiB)
    --page-bytes      granularity for merging addresses    (default 64 KiB)
    --merge-gap       gap below which ranges are merged     (default 1 MiB)
    --hbm-base/--hbf-base   must match configs/h3_router_config.yaml
===============================================================================
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

WARP_SIZE = 32
# A store opcode means the buffer is written, which disqualifies it from HBF.
STORE_RE = re.compile(r"^(STG|STL|STS|ST|RED|ATOM|ATOMG|ATOMS)\b")


def datawidth_from_opcode(op):
    """Mirror of inst_trace_t::get_datawidth_from_opcode()."""
    for tok in op.split("."):
        if tok.isdigit():
            return int(tok) // 8
        if tok.startswith("U") and tok[1:].isdigit():
            return int(tok[1:]) // 8
    return 4


def scan_kernel(path, page_bytes, pages):
    """Tally read and write bytes per page for one kernel trace."""
    version, lineinfo = 1, 0
    with open(path, errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s[0] == "-":
                if "accelsim tracer version" in s:
                    version = int(s.split("=")[1])
                elif "enable_lineinfo" in s:
                    lineinfo = int(s.split("=")[1])
                continue
            if s[0] == "#" or s.startswith(("thread", "warp", "insts", "cluster")):
                continue

            t = s.split()
            try:
                i = 4 if version < 3 else 0
                if lineinfo:
                    i += 1
                i += 1                                  # PC
                mask = int(t[i], 16); i += 1
                n_dst = int(t[i]); i += 1 + n_dst
                opcode = t[i]; i += 1
                n_src = int(t[i]); i += 1 + n_src
                mem_width = int(t[i]); i += 1
                if mem_width <= 0:
                    continue
                mode = int(t[i]); i += 1
                n = bin(mask).count("1")
                if mode == 1:
                    base = int(t[i], 16); i += 1
                    stride = int(t[i]); i += 1
                    addrs = (base + k * stride for k in range(n))
                elif mode == 0:
                    addrs = [int(t[i + k], 16) for k in range(n)]
                    i += n
                else:
                    base = int(t[i], 16); i += 1
                    acc, out = base, [base]
                    for k in range(n - 1):
                        acc += int(t[i]); i += 1
                        out.append(acc)
                    addrs = out
                w = datawidth_from_opcode(opcode)
                is_write = bool(STORE_RE.match(opcode))
                for a in addrs:
                    if a == 0:
                        continue
                    e = pages[a // page_bytes]
                    if is_write:
                        e[1] += w
                    else:
                        e[0] += w
            except (IndexError, ValueError):
                continue    # malformed line: validate_trace.py reports these


def merge_pages(pages, page_bytes, merge_gap):
    """Merge adjacent/nearby touched pages into buffer-sized ranges."""
    if not pages:
        return []
    ranges = []
    cur = None
    for pg in sorted(pages):
        start, end = pg * page_bytes, (pg + 1) * page_bytes
        r, w = pages[pg]
        if cur and start - cur["end"] <= merge_gap:
            cur["end"] = end
            cur["read"] += r
            cur["write"] += w
        else:
            if cur:
                ranges.append(cur)
            cur = {"start": start, "end": end, "read": r, "write": w}
    ranges.append(cur)
    return ranges


def main():
    ap = argparse.ArgumentParser(
        description="Classify a real trace's buffers into H3's HBM and HBF tiers.")
    ap.add_argument("trace_dir")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--hbf-min-bytes", type=int, default=64 * 1024)
    ap.add_argument("--hbf-traffic-target", type=float, default=None,
                    help="ROUTER TEST MODE: send the most-read buffers to HBF "
                         "until this share of traffic (0-1) lands there. This "
                         "is NOT H3's physical placement rule -- it exists so "
                         "both router paths carry real traffic on traces that "
                         "have no large read-only data.")
    ap.add_argument("--page-bytes", type=int, default=64 * 1024)
    ap.add_argument("--merge-gap", type=int, default=1024 * 1024)
    ap.add_argument("--hbm-base", default="0x0")
    ap.add_argument("--hbf-base", default="0x3000000000")
    ap.add_argument("--align", type=int, default=4096)
    args = ap.parse_args()

    kl = os.path.join(args.trace_dir, "kernelslist.g")
    if not os.path.exists(kl):
        kl = os.path.join(args.trace_dir, "kernelslist")
    if not os.path.exists(kl):
        print(f"ERROR: no kernelslist in {args.trace_dir}", file=sys.stderr)
        return 1

    declared = []
    kernels = []
    for line in open(kl):
        line = line.strip()
        if line.startswith("MemcpyHtoD"):
            p = line.split(",")
            declared.append((int(p[1], 16), int(p[2])))
        elif line and not line.startswith("#"):
            kernels.append(line)

    pages = defaultdict(lambda: [0, 0])
    for i, k in enumerate(kernels, 1):
        p = os.path.join(args.trace_dir, k)
        if not os.path.exists(p):
            continue
        print(f"  scanning [{i}/{len(kernels)}] {k}", file=sys.stderr)
        scan_kernel(p, args.page_bytes, pages)

    ranges = merge_pages(pages, args.page_bytes, args.merge_gap)
    if not ranges:
        print("ERROR: no memory accesses found in the trace", file=sys.stderr)
        return 1

    hbm_base = int(args.hbm_base, 0)
    hbf_base = int(args.hbf_base, 0)
    align = args.align

    def bump(cur, size):
        return ((cur + size + align - 1) // align) * align

    # ---- Decide which buffers belong in HBF --------------------------------
    # Default (physical): the trace never writes the buffer, and it is large.
    # --hbf-traffic-target overrides both. It picks the most-read buffers until
    #   the requested share of traffic lands in HBF, so that BOTH router paths
    #   carry real traffic. That is for TESTING THE ROUTER on traces that lack
    #   read-only bulk data -- it is not a placement H3 would actually make.
    forced = set()
    if args.hbf_traffic_target is not None:
        total = sum(r["read"] + r["write"] for r in ranges) or 1
        want = args.hbf_traffic_target * total
        got = 0.0
        for i in sorted(range(len(ranges)), key=lambda k: -ranges[k]["read"]):
            if got >= want:
                break
            forced.add(i)
            got += ranges[i]["read"] + ranges[i]["write"]

    hbm_cur = hbf_cur = 0
    out = []
    for idx, r in enumerate(ranges):
        size = r["end"] - r["start"]
        traffic = r["read"] + r["write"]
        read_only = r["write"] == 0
        to_hbf = read_only and size >= args.hbf_min_bytes
        if args.hbf_traffic_target is not None:
            to_hbf = idx in forced
        # Was this range declared as a host-to-device copy? Corroborating
        # evidence that it holds input data, and useful for naming.
        dec = any(r["start"] <= b < r["end"] for b, _ in declared)
        if to_hbf:
            local = hbf_cur
            hbf_cur = bump(hbf_cur, size)
            region, h3 = "hbf", hbf_base + local
        else:
            local = hbm_cur
            hbm_cur = bump(hbm_cur, size)
            region, h3 = "hbm", hbm_base + local
        out.append({"orig": r["start"], "size": size, "region": region,
                    "h3": h3, "read": r["read"], "write": r["write"],
                    "declared": dec})

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("# H3 allocation map -- GENERATED by tools/classify_trace.py\n"
                "#\n"
                f"# source trace : {args.trace_dir}\n"
                "#\n"
                "# Each entry maps a buffer the trace actually touches into the H3\n"
                "# address space. A buffer goes to HBF only if the trace NEVER stores\n"
                "# to it and it is at least --hbf-min-bytes, which is the placement\n"
                "# rule H3 uses in reality: read-only bulk data in flash, everything\n"
                "# mutable in HBM.\n"
                "#\n"
                "# Fields: orig_addr size region h3_addr   (all sizes in bytes)\n"
                "allocations:\n")
        for e in out:
            f.write(f"  - orig_addr: 0x{e['orig']:x}\n"
                    f"    size: {e['size']}\n"
                    f"    region: {e['region']}\n"
                    f"    h3_addr: 0x{e['h3']:x}\n"
                    f"    bytes_read: {e['read']}\n"
                    f"    bytes_written: {e['write']}\n")

    n_hbf = sum(1 for e in out if e["region"] == "hbf")
    b_hbf = sum(e["size"] for e in out if e["region"] == "hbf")
    b_hbm = sum(e["size"] for e in out if e["region"] == "hbm")
    r_hbf = sum(e["read"] for e in out if e["region"] == "hbf")
    r_all = sum(e["read"] + e["write"] for e in out)

    print(f"\nClassified {len(out)} buffers from {len(kernels)} kernels")
    print(f"  -> HBF : {n_hbf:3d} buffers, {b_hbf/2**20:8.2f} MiB   (read-only, >= "
          f"{args.hbf_min_bytes//1024} KiB)")
    print(f"  -> HBM : {len(out)-n_hbf:3d} buffers, {b_hbm/2**20:8.2f} MiB   "
          f"(written, or small)")
    if r_all:
        print(f"  traffic share to HBF : {r_hbf/r_all*100:5.1f}% of all bytes accessed")
    print(f"  wrote {args.output}")
    if args.hbf_traffic_target is not None:
        print("\n  NOTE: --hbf-traffic-target was used, so placement was chosen to")
        print("        exercise both router paths, NOT by H3's physical rule.")
        print("        Use this to TEST the router; do not quote its performance.")
    if n_hbf == 0:
        print("\n  WARNING: no buffer qualified for HBF. Every access will go to HBM,")
        print("           so the H3 path will not be exercised. Try lowering")
        print("           --hbf-min-bytes, or use a trace with large read-only data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
