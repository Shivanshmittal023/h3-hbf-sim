# H3 Simulator — Hybrid HBM + HBF Memory for Cost-Efficient LLM Inference

A cycle-accurate simulation environment for the **H3** hybrid memory
architecture, built on [Accel-Sim](https://github.com/accel-sim/accel-sim-framework)
(GPGPU-Sim 4.x) with a Ramulator 2.1 device model.

Reference: *H3: Hybrid Architecture Using High Bandwidth Memory and High
Bandwidth Flash for Cost-Efficient LLM Inference*, IEEE Computer Architecture
Letters, 2026.

---

## Overview

Serving very long context LLMs is a **capacity** problem. A Llama 3.1 405B
deployment at 1M tokens needs ~1 TB per GPU of KV cache; at 10M tokens, ~2.5 TB.
HBM cannot reach those capacities at acceptable cost.

H3 pairs each GPU's HBM3e with **High Bandwidth Flash (HBF)** — an SLC NAND
stack behind an HBM-compatible interface. HBF matches HBM's bandwidth (~8 TB/s
per GPU) at 16× the capacity, but its array read latency is **20 µs, roughly
57,000× slower** than HBM. The architecture makes that usable through three
mechanisms, all modeled here:

| Mechanism | What it does | Where |
|---|---|---|
| **Address router** | Splits requests between HBM (mutable, generated KV) and HBF (read-only weights + shared KV) | `h3-components/h3_address_router.*` |
| **Latency Hiding Buffer** | Double-buffered SRAM that prefetches from HBF while the GPU consumes the previous chunk | `h3-components/latency_hiding_buffer.*` |
| **LLM prefetch scheduler** | Exploits the fully deterministic access pattern of transformer inference to issue hints early enough to hide the 20 µs | `h3-components/llm_prefetch_scheduler.*` |

**Why prefetching can work at all:** LLM inference has no data-dependent
addresses. Every tensor read is known before the first token is generated, so
there is nothing to predict — only to schedule.

### What has been measured so far

On the synthetic smoke workload (4 layers, ~36 MiB HBF footprint, 200k cycles):

| Metric | Result |
|---|---|
| LHB hit rate | **0.9992** |
| Work done with LHB vs without | **6.5×** more instructions in the same cycle budget |
| Writes to read-only HBF / unmapped addresses | **0 / 0** |

These validate that the pipeline is correct and that latency hiding works.
They are **not** a reproduction of the paper's 2.69× throughput-per-watt claim —
see [Honest limitations](#honest-limitations).

---

## Repository structure

```
h3-sim/
├── h3-components/              THE H3 MEMORY SYSTEM (all our simulator code)
│   ├── h3_address_router.{h,cpp}        HBM/HBF decode, D2D latency, read-only enforcement
│   ├── latency_hiding_buffer.{h,cpp}    double-buffered SRAM prefetch buffer
│   ├── llm_prefetch_scheduler.{h,cpp}   tensor layout + deterministic hint schedule
│   ├── h3_memory_backend.{h,cpp}        dram_t-compatible backend tying it together
│   ├── ramulator_hbf/hbf.py             HBF device model (Ramulator 2.1 DSL)
│   └── CMakeLists.txt                   builds libh3components.a
├── configs/                    ALL TUNABLE PARAMETERS (every key commented)
│   ├── h3_router_config.yaml            address map, D2D latency, write policy
│   ├── lhb_config.yaml                  buffer size, SRAM latency, bypass switch
│   ├── llama_405b_config.yaml           Llama 3.1 405B architecture
│   ├── llama_smoke.yaml                 small model matched to the smoke trace
│   ├── h3_backend_config.yaml           top-level backend wiring
│   ├── accel_sim_h3.cfg                 Accel-Sim overlay enabling H3
│   └── hbf_h3.py                        Ramulator config (source of record)
├── tests/                      STANDALONE UNIT TESTS (no simulator needed)
├── tools/                      trace generation, validation, result parsing
├── scripts/                    build, patch, and experiment runners
├── patches/                    the GPGPU-Sim hook, as a reviewable diff
├── docs/                       architecture and large-machine setup
├── Dockerfile.h3sim            CUDA 12.8 / Ubuntu 24.04 build environment
└── docker-compose.yml          container `h3sim-env`
```

Everything H3-specific lives in **our** directories. Upstream trees
(`gpu-simulator/`, `ramulator2/`) are never edited in place: changes are applied
by `scripts/apply_gpgpu_sim_patch.sh` and `scripts/install_hbf.sh`, so upstream
stays clean and every modification is reviewable as a diff.

---

## Quick start (Mac / local machine)

### Prerequisites

- **Docker** with ≥ 8 GB RAM allocated (Settings → Resources). 4 GB will OOM
  during the GPGPU-Sim link step.
- **~15 GB disk**: simulator build ~2 GB, Docker image ~8 GB.
- **No GPU required.** Everything here is trace-driven.
- Apple Silicon works — the container runs `linux/amd64` under emulation.

### 1. Build the container

```bash
git clone --recurse-submodules https://github.com/Shivanshmittal023/h3-hbf-sim.git h3-hbf-sim
cd h3-sim
docker compose build            # ~10 min
docker compose run --rm --name h3sim-env h3sim bash
```

### 2. Build the simulator (inside the container)

```bash
./scripts/install_hbf.sh                 # install the HBF model into Ramulator
./scripts/apply_gpgpu_sim_patch.sh       # add the H3 backend hook to GPGPU-Sim

source ./gpu-simulator/setup_environment.sh release
cmake -S ./gpu-simulator/ -B ./gpu-simulator/build/release
cmake --build ./gpu-simulator/build/release -j4      # ~30 min under emulation
cmake --install ./gpu-simulator/build/release
```

Use `-j4`, not `-j8` — higher parallelism under emulation is the usual OOM cause.

### 3. Verify the H3 components (seconds, no simulator needed)

```bash
./scripts/run_unit_tests.sh
```

Expect **166 assertions across 4 tests, 0 failures**. These run without
Ramulator or Accel-Sim, so component logic is verifiable in seconds rather than
after a 30-minute build.

### 4. Run the smoke test

```bash
./scripts/run_smoke_test.sh              # H3 enabled
./scripts/run_smoke_test_hbm_only.sh     # idealised HBM-only baseline
./scripts/run_smoke_test.sh --no-lhb     # ablation: H3 without the buffer
python3 tools/parse_results.py
```

The trace is generated locally (~6 MiB). **Nothing is downloaded.**

### Expected output

```
  RESULT: VALID -- grammar, limits and address map all check out
  ...
  LHB hit rate          : 0.9992
  writes to HBF (must be 0): 0
  unmapped reqs (must be 0): 0

  LHB ablation (H3 with the buffer disabled)
  Instructions              24,384       158,240   6.490x  (+549.0%)
    -> the LHB delivers 6.5x more instructions

  VERDICT: pipeline healthy (no routing or placement errors)
```

The three **must-be-zero** counters are the real pass/fail signal:
`write_attempts_to_hbf`, `unmapped_requests`, `backend_full_rejects`. Any
non-zero value means a data-placement or backpressure bug, and every timing
number should be discarded until it is fixed.

---

## Full experiment setup (large machine)

The laptop flow above proves correctness. Reproducing the paper's results needs
real hardware — see **[docs/large_machine_requirements.md](docs/large_machine_requirements.md)**
for GPU, disk, RAM, non-Docker HPC setup, and NVBit trace generation.

Summary:

| Experiment | GPUs | Sequence | Trace size | Notes |
|---|---|---|---|---|
| Smoke test | 0 | 1K | ~6 MiB | this repo, any laptop |
| Llama 405B, 1M | 8 | 1M | ~2–5 TB | NVBit tracing on A100/H100 |
| Llama 405B, 10M | 32 | 10M | ~20–50 TB | multi-node |

---

## H3 component documentation

| Component | Purpose | Key parameters |
|---|---|---|
| **HBF memory model** (`ramulator_hbf/hbf.py`) | SLC NAND behind an HBM interface. NAND page read maps naturally onto DRAM's two-phase structure: `ACT` = array→page register (tR), `RD` = page register→bus. `Bank` = NAND plane. | `tR` (20 µs), planes/pseudo-channel (64) |
| **H3 address router** | Decodes HBM vs HBF, charges the D2D hop, enforces HBF read-only, counts violations. | region bases/sizes, `d2d.hop_latency_ns`, `hbf_write_policy` |
| **Latency Hiding Buffer** | Streaming double buffer. A half drains in 20 MB / 1 TB/s = 20 µs = tR — that identity *is* Eq. (1). | `buffer_size_mb` (40/cube), `sram_access_latency_ns`, `enabled` |
| **LLM prefetch scheduler** | Computes every tensor's size and address analytically, then issues hints `lead_time` before use. | `lead_time_ns`, `max_outstanding_hints` |

Deep dives: **[docs/architecture.md](docs/architecture.md)** (call flow, data
structures) and **[docs/hbf_model.md](docs/hbf_model.md)** (NAND→DRAM mapping,
approximations, sensitivity ranking).

---

## Configuration parameters

| Parameter | File | Default | Effect |
|---|---|---|---|
| `TR_NS` | `configs/hbf_h3.py` | 20000 | NAND array read latency. **The** primary knob; sets LHB capacity via Eq. (1). |
| `bank` (planes/PC) | `h3-components/ramulator_hbf/hbf.py` | 4 (→64/PC) | Below ~40 planes HBF cannot cover tR and bandwidth collapses. A cliff, not a gradient. |
| `lhb.buffer_size_mb` | `configs/lhb_config.yaml` | 40 | Per cube. Divided across memory partitions at construction. |
| `lhb.enabled` | `configs/lhb_config.yaml` | true | `false` = bypass, the ablation baseline. |
| `prefetch_hint_lead_time_ns` | `configs/lhb_config.yaml` | 45000 | Must exceed tR + transfer. Too short → late misses; too long → buffer pressure. |
| `d2d.hop_latency_ns` | `configs/h3_router_config.yaml` | 25 | Physical routing only — **not** tR. |
| `hbf_write_policy` | `configs/h3_router_config.yaml` | reject | `reject` / `redirect_hbm` / `allow`. |
| `hbf_max_outstanding` | `configs/h3_backend_config.yaml` | 256 | Models the MSHR limit that makes the LHB necessary. |
| `backend.dram_clock_mhz` | `configs/h3_backend_config.yaml` | 3106 | **Must match** `-gpgpu_clock_domains` field 4, or every latency is rescaled. |
| `num_cache_heads` | `configs/llama_405b_config.yaml` | 128 | 128 = MHA (reproduces the paper); 8 = true GQA (16× less KV). |

Unknown keys in any H3 YAML are a **hard error** — a typo can never silently
keep a default.

---

## Simulation parameters vs the paper

| Paper | This repo | Notes |
|---|---|---|
| B200-style GPU | `SM90_H100` config | Closest Accel-Sim config; Hopper-class |
| HBM3e: 192 GB, 8 TB/s | `hbm_bandwidth_gbps: 8000` | Per GPU, divided across 40 partitions |
| HBF: 3 TB, 8 TB/s, tR 20 µs | `HBF_384Gb_16hi` + `tR20us` | 4 TB addressable (power-of-2 decode); router exposes 3 TB |
| LHB 40 MB double-buffered | `buffer_size_mb: 40` | **Per cube.** 8 cubes ⇒ 320 MB/GPU |
| Llama 3.1 405B, FP8 | `llama_405b_config.yaml` | Analytic count = **405.9 B params** ✓ |
| KV ~35% @1M, ~84% @10M | `num_cache_heads: 128`, FP16 | Only MHA+FP16 reproduces these; GQA gives 16× less |
| ISL/OSL 1K, rest shared KV | `input_seq_len`/`output_seq_len` | Shared KV → HBF (read-only); generated → HBM |
| 2.69× throughput/power | **not yet reproduced** | Needs the power path; see below |

Two figures in the paper are internally inconsistent and are documented where
they are used: the Fig. 3(c) address literals (`0x030000000` = 768 MB, not
192 GB) and the per-cube vs per-GPU scope of capacity and bandwidth.

---

## Honest limitations

Read this before quoting any number from this repo.

1. **The 2.69× throughput/power claim is not reproduced.** AccelWattch DRAM
   power under H3 is first-order: reads and writes are real, but row-command
   counts (`n_act`/`n_pre`) are derived, not measured. Power numbers are not
   publication-grade.
2. **The Ramulator device path is not yet exercised.** `device_backend:
   analytic` (closed-form latency + bandwidth) is the default and what the
   results above use. The HBF Ramulator model is written and validated, but
   `h3_ramulator_device.cpp` remains to be written; build with
   `-DH3_WITH_RAMULATOR=ON` once it is.
3. **The HBM-only baseline is idealised.** With the stock backend every address
   is served at HBM speed, i.e. a machine with 3 TB of HBM — which cannot be
   built. It is an upper bound H3 approaches, never a target H3 beats. A real
   capacity-constrained baseline needs multi-GPU scale-out.
4. **The synthetic trace has no compute between memory accesses.** It is a pure
   memory-latency stress test — the worst case for a tiered memory system. Use
   real NVBit traces for performance conclusions.
5. **NAND cache-read mode is not modeled.** Real NAND overlaps the next array
   read with the current page's data-out, roughly halving exposed tR for
   sequential streams. Omitting it makes HBF look *worse* than hardware — the
   conservative direction.

---

## Extending the simulator

**Add a memory model.** Write a `DRAMStandard` subclass in
`h3-components/ramulator_hbf/`, install it with `scripts/install_hbf.sh`, and
let Ramulator's codegen emit the C++. Never hand-edit
`ramulator2/src/ramulator/dram/impl/*.cpp` — it is regenerated on every build.

**Sweep a parameter.** Use `HBF.register_timing_preset()`, **not** keyword
overrides: Ramulator applies overrides *after* secondary-timing resolution, so
an overridden `nRAS` leaves the derived `nRC` stale. This affects every standard
in Ramulator, not only HBF.

**Add a component.** Keep it free of GPGPU-Sim and Ramulator headers, behind an
interface like `IH3MemoryDevice` / `ILhbFillEngine`. That property is what makes
`scripts/run_unit_tests.sh` run in seconds. Enumerators must be **CamelCase, not
SCREAMING_CASE** — GPGPU-Sim defines 435 object-like macros including `READ` and
`WRITE`; `tests/test_macro_hazard.cpp` guards this.

---

## License

This project is MIT licensed (see [LICENSE](LICENSE)).

It builds on, but does not vendor, two BSD-licensed projects:
[Accel-Sim / GPGPU-Sim](https://github.com/accel-sim/accel-sim-framework)
(BSD-3-Clause) and [Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2)
(MIT). Both are git submodules and are never modified in place — changes are
applied at setup time by `scripts/apply_gpgpu_sim_patch.sh` and
`scripts/install_hbf.sh`, so their licences and provenance stay intact.

## Citation

```bibtex
@article{h3_2026,
  title  = {H3: Hybrid Architecture Using High Bandwidth Memory and High
            Bandwidth Flash for Cost-Efficient LLM Inference},
  journal = {IEEE Computer Architecture Letters},
  year   = {2026}
}
```

Please also cite [Accel-Sim](https://accel-sim.github.io/) (Khairy et al., ISCA
2020) and [Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2) (Luo et al.,
CAL 2023). Upstream Accel-Sim documentation lives in the submodule at
`accel-sim-framework/README.md`.
