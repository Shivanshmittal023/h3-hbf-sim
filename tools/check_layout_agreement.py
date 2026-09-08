#!/usr/bin/env python3
"""
===============================================================================
 check_layout_agreement.py -- the generator and the scheduler must agree
===============================================================================

WHY THIS MATTERS
    tools/generate_llm_trace.py computes where every tensor lives so it can emit
    accesses to it. h3-components/llm_prefetch_scheduler.cpp computes the same
    layout so it can issue prefetch hints for it. They are separate
    implementations of the same formulas, in different languages.

    If they ever disagree -- a changed alignment, a reordered tensor, a dropped
    /tp division -- every prefetch hint would point at bytes the trace never
    reads. The hit rate would collapse to zero and the experiment would measure
    nothing, with no error message anywhere.

    This check compares them tensor by tensor: name, address, size and region.

HOW IT WORKS
    Builds a small C++ program against the real scheduler, dumps its layout,
    and diffs it against the Python generator's. Needs only a C++17 compiler --
    no Ramulator, no Accel-Sim.

USAGE
    python3 tools/check_layout_agreement.py [--config configs/llama_smoke.yaml]
                                            [--layers 4]
    exit 0 = layouts agree, 1 = they diverged
===============================================================================
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile

DUMP_CPP = r'''
#include <cstdio>
#include "llm_prefetch_scheduler.h"
int main(int argc, char** argv) {
  h3::ModelConfig c = h3::ModelConfig::from_yaml(argv[1]);
  c.num_layers = (unsigned long long)atoll(argv[2]);
  h3::LLMPrefetchScheduler s(c);
  for (const auto& t : s.tensors())
    printf("%s %llu %llu %s\n", t.name.c_str(),
           (unsigned long long)t.global_addr, (unsigned long long)t.size_bytes,
           t.region == h3::TensorRegion::Hbf ? "hbf" : "hbm");
}
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/llama_smoke.yaml")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "dump.cpp")
        exe = os.path.join(tmp, "dump")
        open(src, "w").write(DUMP_CPP)
        cmd = [a.cxx, "-std=c++17", "-O1", "-I", os.path.join(root, "h3-components"),
               src,
               os.path.join(root, "h3-components", "llm_prefetch_scheduler.cpp"),
               os.path.join(root, "h3-components", "latency_hiding_buffer.cpp"),
               "-o", exe]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("  build of the layout dumper failed:", file=sys.stderr)
            print(r.stderr[:1200], file=sys.stderr)
            return 1
        r = subprocess.run([exe, a.config, str(a.layers)],
                           capture_output=True, text=True, cwd=root)
        if r.returncode != 0:
            print("  layout dumper failed:", r.stderr[:600], file=sys.stderr)
            return 1
        cpp = [l.split() for l in r.stdout.splitlines() if l.strip()]

    spec = importlib.util.spec_from_file_location(
        "gen", os.path.join(root, "tools", "generate_llm_trace.py"))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    cfg = gen.load_config(os.path.join(root, a.config))
    py, _, _ = gen.build_layout(cfg, a.layers)

    print(f"  scheduler (C++) : {len(cpp)} tensors")
    print(f"  generator (py)  : {len(py)} tensors")

    bad = 0
    if len(cpp) != len(py):
        print(f"  COUNT MISMATCH")
        bad += 1
    for p, cl in zip(py, cpp):
        if (p["name"] != cl[0] or p["addr"] != int(cl[1])
                or p["size"] != int(cl[2]) or p["region"] != cl[3]):
            bad += 1
            if bad <= 6:
                print(f"  MISMATCH {p['name']} / {cl[0]}: "
                      f"py 0x{p['addr']:x}/{p['size']}/{p['region']}  vs  "
                      f"cpp 0x{int(cl[1]):x}/{cl[2]}/{cl[3]}")
    if bad:
        print(f"\n  FAIL: {bad} divergence(s). Prefetch hints would point at "
              f"bytes the trace never reads.")
        return 1
    print("\n  PASS: layouts agree exactly (name, address, size, region)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
