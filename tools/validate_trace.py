#!/usr/bin/env python3
"""
===============================================================================
 validate_trace.py -- verify an Accel-Sim trace before the simulator sees it
===============================================================================

WHY THIS EXISTS
    A malformed trace does not produce a helpful error. It produces an
    assertion deep inside the simulator, thousands of lines into a run, e.g.

        accel-sim.out: trace_parser.cc:211:
        Assertion `reg_srcs_num <= MAX_SRC' failed.

    That failure was caused by emitting the pre-version-3 line layout while
    declaring tracer version 4 -- every field shifted by four tokens. This
    script re-implements the parser's grammar and its limits, so the same
    mistake fails in one second with a precise message.

WHAT IT MIRRORS  (gpu-simulator/trace-parser/)
    * trace_parser.h:17-18   MAX_DST = 1, MAX_SRC = 4
    * trace_parser.cc:165    the tb/warp prefix exists ONLY for version < 3
    * trace_parser.cc:179    an extra line_num field when enable_lineinfo is set
    * trace_parser.cc:287    address modes: 0 list_all, 1 base_stride, 2 base_delta
    * trace_parser.cc:281    access width comes from the OPCODE, not mem_width
    * trace_parser.cc:606    `insts = N` must match the lines that follow

    Also checks, using the H3 address map, that every generated address falls
    inside a region the H3 router decodes -- so a layout disagreement between
    the trace generator and h3_address_router shows up here, not as
    h3_router_unmapped_requests after a long run.

USAGE
    python3 tools/validate_trace.py traces/synthetic_h3_smoke_test
    python3 tools/validate_trace.py <dir> --hbm-base 0 --hbm-size-gb 192 \
                                          --hbf-base 0x3000000000 --hbf-size-tb 3

EXIT CODE
    0 = valid, 1 = problems found (each reported with file and line).
===============================================================================
"""

import argparse
import glob
import os
import sys

MAX_DST = 1
MAX_SRC = 4
WARP_SIZE = 32


def datawidth_from_opcode(op):
    """Mirror of inst_trace_t::get_datawidth_from_opcode()."""
    for tok in op.split("."):
        if tok.isdigit():
            return int(tok) // 8
        if tok.startswith("U") and tok[1:].isdigit():
            return int(tok[1:]) // 8
    return 4


class Problems:
    def __init__(self):
        self.items = []

    def add(self, path, lineno, msg):
        self.items.append(f"{os.path.basename(path)}:{lineno}: {msg}")

    def __len__(self):
        return len(self.items)


def parse_kernel(path, probs, regions, stats):
    version, lineinfo = 1, 0
    declared, seen, warp = None, 0, None
    in_tb = False

    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue

            if line.startswith("-"):
                if "accelsim tracer version" in line:
                    version = int(line.split("=")[1])
                elif "enable_lineinfo" in line:
                    lineinfo = int(line.split("=")[1])
                continue
            if line.startswith("#traces format"):
                continue
            if line == "#BEGIN_TB":
                in_tb = True
                continue
            if line == "#END_TB":
                if declared is not None and seen != declared:
                    probs.add(path, lineno, f"warp {warp}: declared insts={declared} but saw {seen}")
                declared, seen = None, 0
                in_tb = False
                continue
            if line.startswith("thread block") or line.startswith("cluster"):
                continue
            if line.startswith("warp ="):
                if declared is not None and seen != declared:
                    probs.add(path, lineno, f"warp {warp}: declared insts={declared} but saw {seen}")
                warp = int(line.split("=")[1])
                declared, seen = None, 0
                continue
            if line.startswith("insts ="):
                declared, seen = int(line.split("=")[1]), 0
                continue

            # ---- instruction line ----
            if not in_tb:
                probs.add(path, lineno, "instruction outside a #BEGIN_TB block")
                continue

            t = line.split()
            i = 0
            try:
                if version < 3:
                    i += 4                      # threadblock x,y,z + warpid_tb
                if lineinfo:
                    i += 1                      # line_num
                int(t[i], 16); i += 1           # PC
                mask = int(t[i], 16); i += 1    # active mask

                n_dst = int(t[i]); i += 1
                if n_dst > MAX_DST:
                    probs.add(path, lineno,
                              f"reg_dsts_num={n_dst} exceeds MAX_DST={MAX_DST} "
                              f"(is the tracer-version line layout correct?)")
                    continue
                i += n_dst

                opcode = t[i]; i += 1
                n_src = int(t[i]); i += 1
                if n_src > MAX_SRC:
                    probs.add(path, lineno,
                              f"reg_srcs_num={n_src} exceeds MAX_SRC={MAX_SRC} "
                              f"(version {version} expects "
                              f"{'a tb/warp prefix' if version < 3 else 'NO tb/warp prefix'})")
                    continue
                i += n_src

                mem_width = int(t[i]); i += 1
                stats["insts"] += 1
                seen += 1

                if mem_width > 0:
                    mode = int(t[i]); i += 1
                    if mode not in (0, 1, 2):
                        probs.add(path, lineno, f"unsupported address mode {mode}")
                        continue
                    n_lanes = bin(mask).count("1")
                    if mode == 1:
                        base = int(t[i], 16); i += 1
                        stride = int(t[i]); i += 1
                        addrs = [base + k * stride for k in range(n_lanes)]
                    elif mode == 0:
                        addrs = [int(t[i + k], 16) for k in range(n_lanes)]
                        i += n_lanes
                    else:
                        base = int(t[i], 16); i += 1
                        addrs = [base]
                        for k in range(n_lanes - 1):
                            addrs.append(addrs[-1] + int(t[i])); i += 1

                    w = datawidth_from_opcode(opcode)
                    stats["mem"] += 1
                    stats["bytes"] += n_lanes * w
                    for a in addrs:
                        placed = False
                        for name, lo, hi in regions:
                            if lo <= a < hi:
                                stats[name] += 1
                                placed = True
                                break
                        if not placed:
                            stats["unmapped"] += 1
                            if stats["unmapped"] <= 5:
                                probs.add(path, lineno,
                                          f"address 0x{a:x} is outside every H3 region")
                    i += 1  # imm (optional on non-memory instructions)
            except (IndexError, ValueError) as e:
                probs.add(path, lineno, f"malformed instruction line ({e}): {line[:70]}")

    if declared is not None and seen != declared:
        probs.add(path, "EOF", f"warp {warp}: declared insts={declared} but saw {seen}")
    return version


def main():
    ap = argparse.ArgumentParser(description="Validate an Accel-Sim trace directory.")
    ap.add_argument("trace_dir")
    ap.add_argument("--hbm-base", default="0")
    ap.add_argument("--hbm-size-gb", type=float, default=192.0)
    ap.add_argument("--hbf-base", default="0x3000000000")
    ap.add_argument("--hbf-size-tb", type=float, default=3.0)
    args = ap.parse_args()

    hbm_base = int(args.hbm_base, 0)
    hbf_base = int(args.hbf_base, 0)
    regions = [
        ("hbm", hbm_base, hbm_base + int(args.hbm_size_gb * (1 << 30))),
        ("hbf", hbf_base, hbf_base + int(args.hbf_size_tb * (1 << 40))),
    ]

    kl = os.path.join(args.trace_dir, "kernelslist.g")
    if not os.path.exists(kl):
        print(f"ERROR: no kernelslist.g in {args.trace_dir}", file=sys.stderr)
        return 1

    probs = Problems()
    stats = dict(insts=0, mem=0, hbm=0, hbf=0, unmapped=0, bytes=0, kernels=0)

    # kernelslist.g must reference files that exist.
    listed = []
    for line in open(kl):
        line = line.strip()
        if not line or line.startswith("MemcpyHtoD"):
            continue
        listed.append(line)
        if not os.path.exists(os.path.join(args.trace_dir, line)):
            probs.add(kl, "-", f"kernelslist references missing file: {line}")

    version = None
    for name in listed:
        p = os.path.join(args.trace_dir, name)
        if os.path.exists(p):
            version = parse_kernel(p, probs, regions, stats)
            stats["kernels"] += 1

    print(f"Validating {args.trace_dir}")
    print(f"  tracer version : {version}"
          f"  ({'tb/warp prefix expected' if (version or 0) < 3 else 'no tb/warp prefix'})")
    print(f"  kernels        : {stats['kernels']}")
    print(f"  instructions   : {stats['insts']:,}  ({stats['mem']:,} memory)")
    print(f"  lane accesses  : HBM {stats['hbm']:,}   HBF {stats['hbf']:,}")
    if stats["hbm"] + stats["hbf"]:
        print(f"  HBF fraction   : {stats['hbf'] / (stats['hbm'] + stats['hbf']):.4f}")
    print(f"  data touched   : {stats['bytes'] / 2**20:.2f} MiB")
    print(f"  unmapped addrs : {stats['unmapped']}")

    if len(probs):
        print(f"\n  {len(probs)} PROBLEM(S):")
        for p in probs.items[:25]:
            print(f"    {p}")
        if len(probs) > 25:
            print(f"    ... and {len(probs) - 25} more")
        print("\n  RESULT: INVALID")
        return 1

    print("\n  RESULT: VALID -- grammar, limits and address map all check out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
