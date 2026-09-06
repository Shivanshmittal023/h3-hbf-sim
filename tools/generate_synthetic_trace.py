#!/usr/bin/env python3
"""
===============================================================================
 generate_synthetic_trace.py -- small synthetic Accel-Sim trace for H3
===============================================================================

WHAT THIS IS
    Generates a SMALL Accel-Sim SASS trace that exercises the H3 memory system
    end to end: it touches both the HBM and the HBF address regions, in the
    ratio a real LLM decode step would, with the sequential locality the
    Latency Hiding Buffer depends on.

    It is a SMOKE TEST, not a workload. It mimics 2-4 transformer layers, not
    126, and a 1K sequence, not 1M. Full-scale Llama traces are generated on a
    machine with real GPUs via NVBit (see docs/large_machine_requirements.md).

NO WEIGHTS, NO DOWNLOADS
    Nothing is fetched. Tensor sizes come from the same analytic formulas as
    h3-components/llm_prefetch_scheduler.cpp, and the trace contains addresses
    only -- never data.

WHAT IT CONNECTS TO
    * configs/llama_405b_config.yaml  -- architecture parameters and the
      HBM/HBF base addresses. The layout produced here MUST match the one
      LLMPrefetchScheduler computes, or the router will report unmapped
      requests. Both read the same YAML.
    * scripts/run_smoke_test.sh       -- consumes the generated directory.
    * Accel-Sim's trace parser        -- gpu-simulator/trace-parser/

TRACE FORMAT (verified against gpu-simulator/trace-parser/trace_parser.cc)
    Per-instruction line (tracer version >= 3):
        PC mask n_dst [dsts] opcode n_src [srcs]
        mem_width [addr_mode base_addr stride] imm

    VERSION MATTERS. inst_trace_t::parse_from_string() reads a leading
    `threadblock_x threadblock_y threadblock_z warpid_tb` prefix ONLY when
    `trace_version < 3`. From version 3 onward the CTA id comes from the
    `thread block = x,y,z` header and the warp id from `warp = N`, so the
    instruction line starts directly at the PC. Emitting the old prefix while
    declaring version 4 shifts every field and trips
    `assert(reg_srcs_num <= MAX_SRC)` deep inside the parser.
    We declare version 4 (matching SM90 configs) and use the modern layout.

    Addresses use addr_mode = 1 (base_stride): one base plus a stride, which
    base_stride_decompress() expands across the active lanes. With
    mask = ffffffff and stride = 16 a single LDG.E.128 covers 32 x 16 = 512
    contiguous bytes -- compact in the file, realistic on the memory system.

    NOTE: the parser derives access width from the OPCODE, not the mem_width
    field (get_datawidth_from_opcode), so LDG.E.128 is what makes it 16 B/lane.

WHY MemcpyHtoD SIZES ARE SMALL
    gpgpu_sim::perf_memcpy_to_gpu() loops over the region in 32-byte steps.
    Declaring the real 3 TB HBF region would mean ~10^11 iterations and the
    simulation would never start. Only the footprint actually touched by this
    trace is declared. Use --no-memcpy (with -gpgpu_perf_sim_memcpy 0) to skip
    the declarations entirely.

USAGE
    python3 tools/generate_synthetic_trace.py \
        --num_layers 4 --batch_size 1 --seq_length 1024 \
        --output_dir ./traces/synthetic_h3_smoke_test

CONFIGURABLE PARAMETERS
    --num_layers            transformer layers to emit      (default 4)
    --batch_size            batch size                      (default 1)
    --seq_length            sequence length in tokens       (default 1024)
    --output_dir            destination directory
    --config                model YAML (default configs/llama_405b_config.yaml)
    --target_trace_mb       hard cap on total trace size    (default 64)
    --hbf_read_fraction     override the HBM/HBF access mix (default: computed)
    --warps_per_block       warps per threadblock           (default 8)
    --no-memcpy             omit MemcpyHtoD declarations
    --seed                  RNG seed for reproducibility    (default 0)
===============================================================================
"""

import argparse
import json
import os
import random
import sys

# --------------------------------------------------------------------------
# Model description
# --------------------------------------------------------------------------

DEFAULTS = {
    "num_layers": 126,
    "hidden_dim": 16384,
    "num_attention_heads": 128,
    "num_kv_heads": 8,
    "intermediate_dim": 53248,
    "vocab_size": 128256,
    "head_dim": 128,
    "weight_dtype_bytes": 1,
    "num_cache_heads": 128,
    "kv_dtype_bytes": 2,
    "input_seq_len": 1024,
    "output_seq_len": 1024,
    "tensor_parallel_size": 8,
    "num_gpus": 8,
    "hbm_base_addr": 0x0,
    "hbf_base_addr": 0x3000000000,
    "tensor_alignment_bytes": 4096,
}


def load_config(path):
    """Read the flat model YAML. Falls back to DEFAULTS if PyYAML is absent."""
    cfg = dict(DEFAULTS)
    if not path or not os.path.exists(path):
        return cfg
    try:
        import yaml
    except ImportError:
        print(f"note: PyYAML not available; using built-in defaults", file=sys.stderr)
        return cfg
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    for section in ("model", "kv_cache", "serving", "address_map"):
        for k, v in (doc.get(section) or {}).items():
            if k in cfg:
                cfg[k] = int(v, 0) if isinstance(v, str) and v.startswith("0x") else v
    return cfg


def align_up(v, a):
    return ((v + a - 1) // a) * a


def layer_weight_bytes(c, tp):
    """Per-layer weight bytes for ONE GPU's tensor-parallel shard.

    Mirrors ModelConfig::weight_bytes_per_layer() in the C++ scheduler:
        q + k + v + o + 3 x mlp + 2 norms
    """
    H, A, KV, hd = c["hidden_dim"], c["num_attention_heads"], c["num_kv_heads"], c["head_dim"]
    I, dt = c["intermediate_dim"], c["weight_dtype_bytes"]
    q = H * A * hd
    k = H * KV * hd
    o = A * hd * H
    mlp = 3 * H * I
    norm = 2 * H
    return (q + 2 * k + o + mlp + norm) * dt // tp


def kv_bytes_per_token(c):
    """2 (K,V) x layers x cache_heads x head_dim x dtype -- whole model."""
    return 2 * c["num_layers"] * c["num_cache_heads"] * c["head_dim"] * c["kv_dtype_bytes"]


# --------------------------------------------------------------------------
# Trace emission
# --------------------------------------------------------------------------

WARP_SIZE = 32
BYTES_PER_LANE = 16           # LDG.E.128 / STG.E.128
BYTES_PER_WARP_ACCESS = WARP_SIZE * BYTES_PER_LANE   # 512 B

# Tracer version 4: no threadblock/warp prefix on instruction lines.
TRACE_VERSION = 4
TRACE_HEADER_FORMAT = (
    "#traces format = PC mask dest_num reg_dests opcode src_num reg_srcs "
    "mem_width mem_addresses"
)


def kernel_header(name, kid, grid, block):
    return (
        f"-kernel name = {name}\n"
        f"-kernel id = {kid}\n"
        f"-grid dim = ({grid[0]},{grid[1]},{grid[2]})\n"
        f"-block dim = ({block[0]},{block[1]},{block[2]})\n"
        f"-shmem = 0\n"
        f"-nregs = 32\n"
        f"-binary version = 90\n"          # SM90, matching SM90_H100 configs
        f"-cuda stream id = 0\n"
        f"-shmem base_addr = 0x0\n"
        f"-local mem base_addr = 0x0\n"
        f"-nvbit version = 1.5.5\n"
        f"-accelsim tracer version = {TRACE_VERSION}\n"
        f"\n"
        f"{TRACE_HEADER_FORMAT}\n"
        f"\n"
    )


def load_inst(pc, base_addr, dst_reg="R4", src_reg="R2"):
    """One warp-wide 512-byte coalesced global load, base_stride encoded."""
    return (
        f"{pc:04x} ffffffff 1 {dst_reg} LDG.E.128 1 {src_reg} "
        f"{BYTES_PER_LANE} 1 0x{base_addr:x} {BYTES_PER_LANE} 0"
    )


def store_inst(pc, base_addr, src_reg="R6"):
    """One warp-wide 512-byte coalesced global store (HBM only -- HBF is RO)."""
    return (
        f"{pc:04x} ffffffff 0 STG.E.128 2 R2 {src_reg} "
        f"{BYTES_PER_LANE} 1 0x{base_addr:x} {BYTES_PER_LANE} 0"
    )


def alu_inst(pc):
    """A compute instruction, so the trace is not purely memory.
    MAX_SRC is 4 and MAX_DST is 1 (trace_parser.h:17-18) -- stay within both."""
    return f"{pc:04x} ffffffff 1 R8 FFMA 3 R4 R6 R8 0"


def emit_kernel(path, name, kid, blocks, warps_per_block, insts_per_warp, addr_fn):
    """Write one kernel trace file.

    addr_fn(block, warp, i) -> (kind, address)
        kind: 'load_hbf' | 'load_hbm' | 'store_hbm' | 'alu'
    """
    counts = {"load_hbf": 0, "load_hbm": 0, "store_hbm": 0, "alu": 0}
    with open(path, "w") as f:
        f.write(kernel_header(name, kid, (blocks, 1, 1), (warps_per_block * WARP_SIZE, 1, 1)))
        for b in range(blocks):
            f.write("#BEGIN_TB\n\n")
            f.write(f"thread block = {b},0,0\n\n")
            for w in range(warps_per_block):
                f.write(f"warp = {w}\n")
                f.write(f"insts = {insts_per_warp}\n")
                pc = 0
                for i in range(insts_per_warp):
                    kind, addr = addr_fn(b, w, i)
                    if kind == "load_hbf":
                        f.write(load_inst(pc, addr) + "\n")
                    elif kind == "load_hbm":
                        f.write(load_inst(pc, addr, dst_reg="R10", src_reg="R12") + "\n")
                    elif kind == "store_hbm":
                        f.write(store_inst(pc, addr) + "\n")
                    else:
                        f.write(alu_inst(pc) + "\n")
                    counts[kind] += 1
                    pc += 16
                f.write("\n")
            f.write("#END_TB\n\n")
    return counts


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Generate a small synthetic Accel-Sim trace for the H3 memory system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--num_layers", type=int, default=4,
                    help="transformer layers to emit (keep small)")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seq_length", type=int, default=1024,
                    help="sequence length in tokens (NOT 1M -- this is a smoke test)")
    ap.add_argument("--output_dir", default="./traces/synthetic_h3_smoke_test")
    ap.add_argument("--config", default="configs/llama_405b_config.yaml")
    ap.add_argument("--target_trace_mb", type=float, default=64.0,
                    help="hard cap on total generated trace size")
    ap.add_argument("--hbf_read_fraction", type=float, default=None,
                    help="override the HBF share of reads (default: computed from the model)")
    ap.add_argument("--warps_per_block", type=int, default=8)
    ap.add_argument("--blocks_per_kernel", type=int, default=16)
    ap.add_argument("--no-memcpy", dest="memcpy", action="store_false",
                    help="omit MemcpyHtoD declarations")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    c = load_config(args.config)
    c["num_layers"] = args.num_layers
    tp = c["tensor_parallel_size"]

    # ---- Footprint --------------------------------------------------------
    w_per_layer = layer_weight_bytes(c, tp)
    kv_tok = kv_bytes_per_token(c) * args.num_layers // DEFAULTS["num_layers"]
    generated_tokens = min(args.seq_length, c["input_seq_len"] + c["output_seq_len"])
    shared_tokens = max(0, args.seq_length - generated_tokens)
    kv_shared_per_layer = kv_tok * shared_tokens // max(1, args.num_layers) // c["num_gpus"]
    kv_gen_per_layer = kv_tok * generated_tokens // max(1, args.num_layers) // c["num_gpus"]

    # HBF traffic = weights + shared KV; HBM traffic = generated KV.
    hbf_bytes = (w_per_layer + kv_shared_per_layer) * args.num_layers
    hbm_bytes = kv_gen_per_layer * args.num_layers
    computed_frac = hbf_bytes / max(1, hbf_bytes + hbm_bytes)
    hbf_frac = args.hbf_read_fraction if args.hbf_read_fraction is not None else computed_frac
    hbf_frac = min(max(hbf_frac, 0.0), 1.0)

    # ---- Size the trace to the byte budget --------------------------------
    # ~72 bytes per instruction line; 2 kernels per layer.
    kernels = args.num_layers * 2
    budget_bytes = int(args.target_trace_mb * 1024 * 1024)
    est_line = 72
    total_insts = budget_bytes // est_line
    per_kernel = max(1, total_insts // kernels)
    insts_per_warp = max(8, per_kernel // (args.blocks_per_kernel * args.warps_per_block))

    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)

    hbm_base = c["hbm_base_addr"]
    hbf_base = c["hbf_base_addr"]
    align = c["tensor_alignment_bytes"]

    # Sequential cursors: the LHB's whole premise is sequential streaming, so
    # the trace must stream, not random-access. Any randomness here would
    # silently destroy the prefetch hit rate we are trying to measure.
    hbf_cursor = [hbf_base]
    hbm_cursor = [hbm_base]

    kernel_files = []
    totals = {"load_hbf": 0, "load_hbm": 0, "store_hbm": 0, "alu": 0}
    kid = 1

    for layer in range(args.num_layers):
        for phase in ("attention", "mlp"):
            def addr_fn(b, w, i, _hbf=hbf_cursor, _hbm=hbm_cursor):
                r = random.random()
                if r < 0.15:
                    return ("alu", 0)
                if random.random() < hbf_frac:
                    a = _hbf[0]
                    _hbf[0] += BYTES_PER_WARP_ACCESS
                    return ("load_hbf", a)
                # HBM: generated KV is read and written
                a = _hbm[0]
                _hbm[0] += BYTES_PER_WARP_ACCESS
                if random.random() < 0.25:
                    return ("store_hbm", a)
                return ("load_hbm", a)

            fname = f"kernel-{kid}.traceg"
            counts = emit_kernel(
                os.path.join(outdir, fname),
                f"_Z{len(phase)}h3_{phase}_layer{layer}",
                kid, args.blocks_per_kernel, args.warps_per_block,
                insts_per_warp, addr_fn,
            )
            for k, v in counts.items():
                totals[k] += v
            kernel_files.append(fname)
            kid += 1

    hbf_touched = align_up(hbf_cursor[0] - hbf_base, align)
    hbm_touched = align_up(hbm_cursor[0] - hbm_base, align)

    # ---- kernelslist.g ----------------------------------------------------
    # MemcpyHtoD declares only the footprint actually touched. See the module
    # docstring: perf_memcpy_to_gpu walks the region in 32-byte steps.
    with open(os.path.join(outdir, "kernelslist.g"), "w") as f:
        if args.memcpy:
            f.write(f"MemcpyHtoD,0x{hbm_base:016x},{hbm_touched}\n")
            f.write(f"MemcpyHtoD,0x{hbf_base:016x},{hbf_touched}\n")
        for kf in kernel_files:
            f.write(kf + "\n")

    # ---- manifest ---------------------------------------------------------
    # tools/parse_results.py checks simulator counters against these expectations.
    manifest = {
        "generator": "generate_synthetic_trace.py",
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "kernels": len(kernel_files),
        "insts_per_warp": insts_per_warp,
        "blocks_per_kernel": args.blocks_per_kernel,
        "warps_per_block": args.warps_per_block,
        "hbm_base_addr": hex(hbm_base),
        "hbf_base_addr": hex(hbf_base),
        "hbm_bytes_touched": hbm_touched,
        "hbf_bytes_touched": hbf_touched,
        "hbf_read_fraction_target": hbf_frac,
        "expected_accesses": totals,
        "expected_hbf_access_fraction": (
            totals["load_hbf"] /
            max(1, totals["load_hbf"] + totals["load_hbm"] + totals["store_hbm"])
        ),
        "bytes_per_warp_access": BYTES_PER_WARP_ACCESS,
    }
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ---- report -----------------------------------------------------------
    total_size = sum(
        os.path.getsize(os.path.join(outdir, f)) for f in os.listdir(outdir)
    )
    print(f"Synthetic H3 trace written to {outdir}")
    print(f"  layers / kernels        : {args.num_layers} / {len(kernel_files)}")
    print(f"  insts per warp          : {insts_per_warp}")
    print(f"  HBF loads               : {totals['load_hbf']:,}")
    print(f"  HBM loads / stores      : {totals['load_hbm']:,} / {totals['store_hbm']:,}")
    print(f"  ALU insts               : {totals['alu']:,}")
    print(f"  HBF access fraction     : {manifest['expected_hbf_access_fraction']:.3f} "
          f"(target {hbf_frac:.3f})")
    print(f"  HBM footprint touched   : {hbm_touched/2**20:.2f} MiB @ 0x{hbm_base:x}")
    print(f"  HBF footprint touched   : {hbf_touched/2**20:.2f} MiB @ 0x{hbf_base:x}")
    print(f"  total trace size        : {total_size/2**20:.2f} MiB "
          f"(cap {args.target_trace_mb:.0f} MiB)")
    if total_size > budget_bytes * 1.25:
        print("  WARNING: trace exceeded its size budget; lower --target_trace_mb",
              file=sys.stderr)


if __name__ == "__main__":
    main()
