#!/usr/bin/env python3
"""
===============================================================================
 generate_llm_trace.py -- a realistic LLM decode workload for the H3 simulator
===============================================================================

WHY THIS EXISTS
    We screened every freely downloadable Accel-Sim trace suite:

        rodinia_2.0-ft   11 benchmarks   0.0 - 0.5% of traffic read-only
        ubench           29 benchmarks   0.0%  -- every single one

    Not one has what H3 targets: a large buffer that is read many times and
    never written. rodinia is HPC kernels that read AND write their working
    sets; ubench is bandwidth/latency microbenchmarks that write by design. The
    only suite that would work is CUTLASS, whose GEMM weights are genuinely
    read-only -- and it is 387 GB compressed, 3.3 TB extracted.

    So no public trace can exercise H3. This generator produces the workload
    instead.

WHY GENERATING IT IS LEGITIMATE, NOT A SHORTCUT
    LLM inference has a COMPLETELY DETERMINISTIC memory access pattern. Every
    tensor read is known before the first token is generated -- there are no
    data-dependent addresses. That is the H3 paper's own premise for why
    prefetching works at all (Section III-C). A pattern that is knowable in
    advance can be generated exactly; there is nothing to discover by tracing
    it. What tracing would add is the surrounding kernel code, which is not
    what the memory system study is about.

WHAT IT MODELS -- one decode step, layer by layer, in execution order

    for each layer:
        input_norm            read weights            HBF
        q_proj, k_proj, v_proj  read weights          HBF
        attention             read shared KV cache    HBF   <- the bulk
                              read+write generated KV HBM
        o_proj                read weights            HBF
        post_norm             read weights            HBF
        gate_proj, up_proj, down_proj  read weights   HBF

    Addresses are laid out with the SAME formulas and the same 4 KiB alignment
    as LLMPrefetchScheduler::build_schedule(), in the same order, so a prefetch
    hint for a tensor points at exactly the bytes this trace then reads. If the
    two disagreed, every prefetch would miss and the experiment would measure
    nothing.

    Compute instructions are emitted between memory accesses. Decode at batch 1
    is memory bound -- each weight byte is used once, giving ~2 FLOPs/byte at
    FP8. A warp access moves 512 B (2 x 512 = 1024 FLOPs) and a warp FFMA does
    32 lanes x 2 = 64 FLOPs, so ~16 FFMA per memory access. That is the default;
    --compute-per-access overrides it.

WHAT IT CONNECTS TO
    * configs/llama_smoke.yaml or configs/llama_405b_config.yaml (architecture)
    * h3-components/llm_prefetch_scheduler.cpp (identical layout formulas)
    * tools/validate_trace.py (output is checked against Accel-Sim's grammar)
    * scripts/run_smoke_test.sh (consumes the generated directory)

    NO WEIGHTS ARE DOWNLOADED OR LOADED. Only sizes are computed; the trace
    contains addresses, never data.

USAGE
    python3 tools/generate_llm_trace.py \
        --config configs/llama_smoke.yaml --layers 4 --decode-steps 8 \
        --output_dir traces/llm_decode

KEY PARAMETERS
    --decode-steps        how many tokens to generate. MORE IS BETTER: one
                          buffer refill takes ~20 us, so a short run measures
                          pipeline warm-up rather than steady state.
    --compute-per-access  ALU instructions per memory instruction (default 16)
    --target_trace_mb     hard cap on output size
===============================================================================
"""

import argparse
import collections
import json
import os
import sys

WARP = 32
LANE_BYTES = 16                       # LDG.E.128 / STG.E.128
WARP_BYTES = WARP * LANE_BYTES        # 512 B per warp-wide access
TRACE_VERSION = 4                     # no threadblock/warp prefix on inst lines

DEFAULTS = {
    "num_layers": 4, "hidden_dim": 2048, "num_attention_heads": 16,
    "num_kv_heads": 2, "intermediate_dim": 5632, "vocab_size": 8192,
    "head_dim": 128, "weight_dtype_bytes": 1,
    "num_cache_heads": 16, "kv_dtype_bytes": 2,
    "input_seq_len": 512, "output_seq_len": 512, "sequence_length": 4096,
    "num_gpus": 8, "tensor_parallel_size": 8,
    "hbm_base_addr": 0x0, "hbf_base_addr": 0x3000000000,
    "tensor_alignment_bytes": 4096,
}


def load_config(path):
    cfg = dict(DEFAULTS)
    if not path or not os.path.exists(path):
        return cfg
    try:
        import yaml
    except ImportError:
        print("note: PyYAML missing; using built-in defaults", file=sys.stderr)
        return cfg
    doc = yaml.safe_load(open(path)) or {}
    for sec in ("model", "kv_cache", "serving", "address_map"):
        for k, v in (doc.get(sec) or {}).items():
            if k in cfg:
                cfg[k] = int(v, 0) if isinstance(v, str) and v.startswith("0x") else v
    return cfg


# --------------------------------------------------------------------------
# Tensor layout -- MUST match LLMPrefetchScheduler::build_schedule()
# --------------------------------------------------------------------------
def build_layout(c, layers):
    """Return (tensors, hbf_bytes, hbm_bytes).

    tensors: list of dicts {name, layer, region, addr, size, phase}
    Same order, same formulas and same 4 KiB alignment as the C++ scheduler.
    """
    align = c["tensor_alignment_bytes"]
    tp = c["tensor_parallel_size"]
    H, A, hd = c["hidden_dim"], c["num_attention_heads"], c["head_dim"]
    KV, I, V = c["num_kv_heads"], c["intermediate_dim"], c["vocab_size"]
    dt = c["weight_dtype_bytes"]

    cur = {"hbf": 0, "hbm": 0}
    base = {"hbf": c["hbf_base_addr"], "hbm": c["hbm_base_addr"]}
    out = []

    def add(name, layer, region, size, phase):
        if size <= 0:
            return
        off = cur[region]
        out.append({"name": name, "layer": layer, "region": region,
                    "addr": base[region] + off, "size": size, "phase": phase})
        cur[region] = ((off + size + align - 1) // align) * align

    # ---- HBF: weights, in execution order --------------------------------
    add("embed_tokens", -1, "hbf", V * H * dt // tp, "embed")
    for l in range(layers):
        add(f"layer{l}.input_norm", l, "hbf", H * dt, "norm")
        add(f"layer{l}.q_proj", l, "hbf", H * A * hd * dt // tp, "qkv")
        add(f"layer{l}.k_proj", l, "hbf", H * KV * hd * dt // tp, "qkv")
        add(f"layer{l}.v_proj", l, "hbf", H * KV * hd * dt // tp, "qkv")
        add(f"layer{l}.o_proj", l, "hbf", A * hd * H * dt // tp, "o_proj")
        add(f"layer{l}.post_norm", l, "hbf", H * dt, "norm")
        add(f"layer{l}.gate_proj", l, "hbf", H * I * dt // tp, "mlp")
        add(f"layer{l}.up_proj", l, "hbf", H * I * dt // tp, "mlp")
        add(f"layer{l}.down_proj", l, "hbf", I * H * dt // tp, "mlp")
    add("lm_head", -1, "hbf", V * H * dt // tp, "embed")

    # ---- HBF: shared precomputed KV cache, one slice per layer -----------
    kv_tok = 2 * layers * c["num_cache_heads"] * hd * c["kv_dtype_bytes"]
    gen_tokens = c["input_seq_len"] + c["output_seq_len"]
    shared_tokens = max(0, c["sequence_length"] - gen_tokens)
    shared_per_layer = kv_tok * shared_tokens // c["num_gpus"] // max(1, layers)
    for l in range(layers):
        add(f"layer{l}.kv_shared", l, "hbf", shared_per_layer, "attention")

    # ---- HBM: generated KV cache, then activations -----------------------
    gen_per_layer = kv_tok * gen_tokens // c["num_gpus"] // max(1, layers)
    for l in range(layers):
        add(f"layer{l}.kv_generated", l, "hbm", gen_per_layer, "attention")
    act = (H + I) * 2 * c["kv_dtype_bytes"] * 2
    add("activations", -1, "hbm", act, "activation")

    return out, cur["hbf"], cur["hbm"]


# --------------------------------------------------------------------------
# Trace emission
# --------------------------------------------------------------------------
HDR_FMT = ("#traces format = PC mask dest_num reg_dests opcode src_num reg_srcs "
           "mem_width mem_addresses")


def kernel_header(name, kid, blocks, threads):
    return (f"-kernel name = {name}\n-kernel id = {kid}\n"
            f"-grid dim = ({blocks},1,1)\n-block dim = ({threads},1,1)\n"
            f"-shmem = 0\n-nregs = 40\n-binary version = 90\n"
            f"-cuda stream id = 0\n-shmem base_addr = 0x0\n"
            f"-local mem base_addr = 0x0\n-nvbit version = 1.5.5\n"
            f"-accelsim tracer version = {TRACE_VERSION}\n\n{HDR_FMT}\n\n")


def ld(pc, addr, dst="R4", src="R2"):
    return (f"{pc:04x} ffffffff 1 {dst} LDG.E.128 1 {src} "
            f"{LANE_BYTES} 1 0x{addr:x} {LANE_BYTES} 0")


def st(pc, addr):
    return (f"{pc:04x} ffffffff 0 STG.E.128 2 R2 R6 "
            f"{LANE_BYTES} 1 0x{addr:x} {LANE_BYTES} 0")


def alu(pc):
    return f"{pc:04x} ffffffff 1 R8 FFMA 3 R4 R6 R8 0"


def main():
    ap = argparse.ArgumentParser(
        description="Generate a realistic LLM decode workload for H3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", default="configs/llama_smoke.yaml")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--decode-steps", type=int, default=8,
                    help="tokens to generate; more reaches steady state")
    ap.add_argument("--output_dir", default="./traces/llm_decode")
    ap.add_argument("--target_trace_mb", type=float, default=48.0)
    ap.add_argument("--compute-per-access", type=int, default=16,
                    help="ALU instructions per memory access (FP8 decode ~16)")
    ap.add_argument("--warps-per-block", type=int, default=8)
    ap.add_argument("--blocks-per-kernel", type=int, default=16)
    ap.add_argument("--no-memcpy", dest="memcpy", action="store_false")
    args = ap.parse_args()

    c = load_config(args.config)
    layers = args.layers
    tensors, hbf_bytes, hbm_bytes = build_layout(c, layers)
    by_layer = {}
    for t in tensors:
        by_layer.setdefault(t["layer"], []).append(t)

    os.makedirs(args.output_dir, exist_ok=True)

    # Budget: ~72 bytes per instruction line, split across all kernels.
    kernels_total = args.decode_steps * layers
    budget = int(args.target_trace_mb * 1024 * 1024)
    inst_total = budget // 72
    per_kernel = max(64, inst_total // max(1, kernels_total))
    warps = args.warps_per_block * args.blocks_per_kernel
    insts_per_warp = max(16, per_kernel // warps)

    # Each warp slot walks its tensor from a different offset, so a kernel
    # streams the tensor rather than hammering one address.
    counts = {"hbf": 0, "hbm_r": 0, "hbm_w": 0, "alu": 0}
    files = []
    kid = 1

    for step in range(args.decode_steps):
        for l in range(layers):
            ts = by_layer.get(l, [])
            if not ts:
                continue
            hbf_t = [t for t in ts if t["region"] == "hbf"]
            hbm_t = [t for t in ts if t["region"] == "hbm"]
            fname = f"kernel-{kid}.traceg"
            path = os.path.join(args.output_dir, fname)
            with open(path, "w") as f:
                f.write(kernel_header(f"_Z12h3_decode_s{step}_l{l}", kid,
                                      args.blocks_per_kernel,
                                      args.warps_per_block * WARP))
                n_warps = args.blocks_per_kernel * args.warps_per_block
                for b in range(args.blocks_per_kernel):
                    f.write("#BEGIN_TB\n\n")
                    f.write(f"thread block = {b},0,0\n\n")
                    for w in range(args.warps_per_block):
                        # Each warp owns a contiguous slice of every tensor and
                        # walks it in order, which is how a tiled GEMM reads.
                        slot = b * args.warps_per_block + w
                        cursor = collections.defaultdict(int)
                        kv_cursor = [0]
                        lines = []
                        pc = 0
                        i = 0
                        ti = 0
                        while i < insts_per_warp:
                            t = hbf_t[ti % len(hbf_t)] if hbf_t else None
                            ti += 1
                            if t is not None:
                                # SEQUENTIAL STREAMING, not a strided walk.
                                # A real GEMM partitions the weight matrix into
                                # tiles and each warp reads its tile
                                # contiguously. That contiguity is what makes
                                # prefetching possible at all, so getting it
                                # wrong would both misrepresent the hardware and
                                # understate the latency hiding buffer.
                                span = max(1, t["size"] // WARP_BYTES)
                                chunk = max(1, span // n_warps)
                                start = (slot % n_warps) * chunk
                                idx = (start + cursor[t["name"]]) % span
                                cursor[t["name"]] += 1
                                lines.append(ld(pc, t["addr"] + idx * WARP_BYTES))
                                counts["hbf"] += 1
                                pc += 16; i += 1
                            for _ in range(args.compute_per_access):
                                if i >= insts_per_warp:
                                    break
                                lines.append(alu(pc)); counts["alu"] += 1
                                pc += 16; i += 1
                            # attention touches the generated KV cache in HBM:
                            # read it, and append this step's new entry
                            if hbm_t and i < insts_per_warp:
                                # Attention walks the KV cache sequentially too:
                                # it scans the whole context in order.
                                h = hbm_t[0]
                                span = max(1, h["size"] // WARP_BYTES)
                                chunk = max(1, span // n_warps)
                                start = (slot % n_warps) * chunk
                                idx = (start + kv_cursor[0]) % span
                                kv_cursor[0] += 1
                                a = h["addr"] + idx * WARP_BYTES
                                if (i + slot) % 8 == 0:
                                    lines.append(st(pc, a)); counts["hbm_w"] += 1
                                else:
                                    lines.append(ld(pc, a, "R10", "R12"))
                                    counts["hbm_r"] += 1
                                pc += 16; i += 1
                        f.write(f"warp = {w}\ninsts = {len(lines)}\n")
                        f.write("\n".join(lines) + "\n\n")
                    f.write("#END_TB\n\n")
            files.append(fname)
            kid += 1

    # kernelslist: declare only the footprint actually touched, because
    # perf_memcpy_to_gpu walks each declared region in 32-byte steps.
    with open(os.path.join(args.output_dir, "kernelslist.g"), "w") as f:
        if args.memcpy:
            f.write(f"MemcpyHtoD,0x{c['hbm_base_addr']:016x},{hbm_bytes}\n")
            f.write(f"MemcpyHtoD,0x{c['hbf_base_addr']:016x},{hbf_bytes}\n")
        for fn in files:
            f.write(fn + "\n")

    total_mem = counts["hbf"] + counts["hbm_r"] + counts["hbm_w"]
    manifest = {
        "generator": "generate_llm_trace.py",
        "config": args.config, "layers": layers,
        "decode_steps": args.decode_steps,
        "kernels": len(files), "insts_per_warp": insts_per_warp,
        "compute_per_access": args.compute_per_access,
        "hbf_bytes_footprint": hbf_bytes, "hbm_bytes_footprint": hbm_bytes,
        "hbm_base_addr": hex(c["hbm_base_addr"]),
        "hbf_base_addr": hex(c["hbf_base_addr"]),
        "expected_accesses": counts,
        "expected_hbf_access_fraction": counts["hbf"] / max(1, total_mem),
        "bytes_per_warp_access": WARP_BYTES,
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    size = sum(os.path.getsize(os.path.join(args.output_dir, f))
               for f in os.listdir(args.output_dir))
    print(f"LLM decode workload written to {args.output_dir}")
    print(f"  layers x decode steps  : {layers} x {args.decode_steps} "
          f"= {len(files)} kernels")
    print(f"  tensors laid out       : {len(tensors)}")
    print(f"  HBF footprint          : {hbf_bytes/2**20:.2f} MiB "
          f"(weights + shared KV cache)")
    print(f"  HBM footprint          : {hbm_bytes/2**20:.2f} MiB "
          f"(generated KV + activations)")
    print(f"  HBF loads              : {counts['hbf']:,}")
    print(f"  HBM loads / stores     : {counts['hbm_r']:,} / {counts['hbm_w']:,}")
    print(f"  compute instructions   : {counts['alu']:,}")
    print(f"  HBF access fraction    : {manifest['expected_hbf_access_fraction']:.3f}")
    print(f"  trace size             : {size/2**20:.2f} MiB "
          f"(cap {args.target_trace_mb:.0f})")


if __name__ == "__main__":
    sys.exit(main())
