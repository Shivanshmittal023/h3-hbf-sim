# H3 Simulator — Software Architecture

How the components connect at the code level, what happens to a single memory
request, and what to change for a given experiment.

Companion to [../README.md](../README.md) and [hbf_model.md](hbf_model.md).

---

## 1. Layering

```
┌───────────────────────────────────────────────────────────────────────┐
│  Accel-Sim / GPGPU-Sim 4.x                            (upstream)      │
│    trace-parser  →  SM cores  →  L1  →  interconnect  →  L2           │
│                                                          │            │
│                                    memory_partition_unit │ x40        │
│                                        h3_dram_cycle()   │            │
└──────────────────────────────────────────┬────────────────────────────┘
                                           │  mem_fetch*  (as void*)
        ═══════════ patch boundary ════════╪═══════════════════════
                                           ▼
┌───────────────────────────────────────────────────────────────────────┐
│  h3-components/                    (this repository, no GPGPU-Sim deps)│
│                                                                        │
│   H3MemoryBackend            one per memory partition                  │
│     ├── H3AddressRouter      decode HBM|HBF, D2D hop, read-only check   │
│     ├── LatencyHidingBuffer  double-buffered SRAM prefetch              │
│     │     └── ILhbFillEngine     AnalyticFillEngine | Ramulator         │
│     ├── LLMPrefetchScheduler deterministic tensor schedule → hints      │
│     └── IH3MemoryDevice x2   AnalyticMemoryDevice | RamulatorDevice     │
└───────────────────────────────────────────────────────────────────────┘
```

**The one rule that keeps this maintainable:** nothing under `h3-components/`
includes a GPGPU-Sim or Ramulator header. The originating `mem_fetch*` crosses
the boundary as an opaque `void*`. Consequences:

- the whole H3 stack compiles with a bare `g++ -std=c++17`
- `scripts/run_unit_tests.sh` verifies 166 assertions in seconds, not after a
  30-minute simulator build
- an upstream API change cannot silently alter H3 behaviour

The only file that sees both worlds is `patches/gpgpu-sim-h3-backend.patch`.

---

## 1b. Two ways the router decodes an address

| Mode | When | How |
|---|---|---|
| **Fixed regions** | synthetic traces | `addr < 192 GB` → HBM, else HBF |
| **Allocation map** | real traces | look the address up in a table of buffers built by `tools/classify_trace.py` |

The second exists because a real trace's addresses (around `0x7f35ab700000`)
fall in neither fixed region, so every request would be reported unmapped and
the HBF path would never run. The map is a sorted vector searched by binary
search; addresses in no known buffer (stack, local, constant) go to HBM and are
counted separately as `unallocated_requests` so they stay visible rather than
being silently misreported.

## 2. Life of a memory request

```
 (1) L2 miss in memory_partition_unit
       │
       ▼
 (2) h3_dram_cycle()                          [patched into l2cache.cc]
       │  full()? ──yes──► RETRY next cycle (never dropped)
       ▼
 (3) H3MemoryBackend::push(H3MemRequest{addr, is_write, size, mem_fetch*})
       │
       ▼
 (4) H3AddressRouter::route()
       ├── decode(addr)
       │     ├── HBM region  ──────────────────────────────┐
       │     ├── HBF region  ──┐                           │
       │     └── unmapped ──► REJECTED_UNMAPPED (counted)  │
       │                       │                           │
       │        write to HBF? ─┴─► REJECTED_WRITE_TO_HBF   │
       │                            (data-placement bug)   │
       │        charge D2D hop: issue_time += 25 ns        │
       ▼                                                   ▼
 (5) LatencyHidingBuffer::access(local_addr, size, t)   HBM device
       ├── Hit        → t + 2 ns  (SRAM)                    │
       ├── MissLate   → stall until the in-flight fill lands │
       └── MissAbsent → full demand read from HBF            │
       │                                                     │
       ▼                                                     ▼
 (6) Pending queue, ordered by ready time
       │
       ▼
 (7) cycle(): scheduler.tick() → lhb.tick() → devices.tick() → retire_ready()
       │
       ▼
 (8) return_queue → h3_dram_cycle() pops → dram_L2_queue_push(mem_fetch*)
```

Two invariants worth preserving:

- **`full()` is a retry signal, not a drop.** It must include *device* queue
  occupancy, not just the backend's own queues. An earlier version did not, and
  77% of requests were retired with zero latency — the timing model was silently
  bypassed while every counter still looked plausible.
- **Correctness counters must read zero:** `write_attempts_to_hbf`,
  `unmapped_requests`, `backend_full_rejects`. Any non-zero value invalidates
  the timing results; `tools/parse_results.py` flags them.

---

## 3. The prefetch path

Demand traffic alone cannot sustain HBF. Per partition the latency-bandwidth
product is `20 µs × 200 GB/s / 32 B ≈ 125,000` outstanding sector requests — no
controller has that many MSHRs. The LHB converts that into a few large
sequential fills:

```
LLMPrefetchScheduler                      LatencyHidingBuffer
  build_schedule()                          half[0]:  Idle
    tensors laid out in execution order        │  issue_hint()
    needed_at = cumulative_bytes / BW          ▼
    hint_at   = needed_at - lead_time       PrefetchIssued
                                               ▼   start_fill()
  tick(now, lhb) ──issue_hint()──►          Filling   (tR + transfer)
                                               ▼
                                             Ready
                                               ▼   GPU access() → Hit
                                            Serving
                                               ▼   consumed >= stored_bytes
                                             Idle  (swap, refill)
```

### Address interleaving — the subtlety that matters

GPGPU-Sim spreads consecutive cache lines across all 40 memory partitions, so a
partition sees only every 40th line of any contiguous range. A per-partition LHB
slice must therefore cover the **same global address span** as the full buffer
while storing only its 1/N share. Each `Half` tracks two distinct quantities:

| field | meaning |
|---|---|
| `length` | global address span the half covers |
| `stored_bytes` | SRAM actually occupied = `length / interleave_factor` |

Residency uses `length`; fill timing, occupancy, waste and the drain/swap test
use `stored_bytes`. Getting this wrong is not a subtle error: without it the
window is N× too narrow, prefetch efficiency collapses to 1/N, and absent misses
dominate. `tests/test_lhb.cpp` §11 covers it, including a control run with the
factor reset to 1.

---

## 4. Units and scopes — the two easiest mistakes

**Time.** Every H3 component works in **picoseconds** (`uint64_t`). GPGPU-Sim
counts DRAM cycles. Conversion happens once, in `H3BackendConfig::cycles_to_ps`,
from `backend.dram_clock_mhz`. That value **must** match field 4 of
`-gpgpu_clock_domains` in the active `gpgpusim.config`, or every latency in the
system is rescaled — with no error message.

**Scope.** Config values are **per GPU**; GPGPU-Sim instantiates one backend
**per memory partition**. The backend divides by the partition count it is
given (`-gpgpu_n_mem`, passed in from `l2cache.cc` rather than duplicated in
YAML). Writing per-partition values in the YAML would give the simulated machine
40× the bandwidth and 40× the buffer, and no single statistic would look wrong.

Per the paper's own arithmetic, the LHB is a **per-cube** structure: Eq. (1)
only yields 40 MB with `BW = 1 TB/s`, which is per-cube. A GPU carries
8 × 40 MB = 320 MB.

---

## 5. What to change for a given experiment

| Goal | Change |
|---|---|
| Sweep NAND latency | `TR_NS` in `configs/hbf_h3.py`; use `HBF.register_timing_preset()`, never keyword overrides |
| Sweep LHB capacity | `lhb.buffer_size_mb`; compare against `lhb_required_capacity_eq1_bytes` in the output |
| Measure the LHB's contribution | `./scripts/run_smoke_test.sh --no-lhb` |
| Measure the *scheduler's* contribution | `backend.enable_prefetch_scheduler: false` — keeps the buffer, removes the hints |
| Starve HBF of parallelism | `ORG_PRESET: HBF_384Gb_16hi_lowpar` (32 planes < the ~40 needed) |
| Use real Ramulator timing | build with `-DH3_WITH_RAMULATOR=ON`, set `device_backend: ramulator` |
| Different GPU | point `BASE_CONFIG` at another `tested-cfgs/` entry; update `dram_clock_mhz` to match |
| Run a real trace | `scripts/run_real_trace.sh <dir>` — classifies buffers, screens, then simulates |
| Test the router on a trace with no read-only data | `classify_trace.py --hbf-traffic-target 0.5` forces a split so both paths carry traffic |
| True GQA instead of the paper's KV shape | `num_cache_heads: 8` in `configs/llama_405b_config.yaml` |

---

## 6. Build integration

```
h3-components/CMakeLists.txt          →  libh3components.a  (PIC, C++17)
        ▲
        │ add_subdirectory(), applied by the patch
src/gpgpu-sim/CMakeLists.txt          →  libgpgpusim.a
        │
        └── target_link_libraries(gpgpusim PUBLIC h3components)
```

`l2cache.h` only **forward-declares** `h3::H3MemoryBackend`. It must not include
the header: `l2cache.h` is pulled into `libcuda`, `libopencl` and `cuda-sim`,
none of which link `h3components`. The full header is included by `l2cache.cc`,
which builds inside the `gpgpusim` target.

---

## 7. Testing strategy

| Test | Assertions | Guards against |
|---|---:|---|
| `test_h3_router.cpp` | 61 | boundary decode, read-only enforcement, D2D accounting, backpressure |
| `test_lhb.cpp` | 56 | Eq. (1), double buffering, streaming, bypass, **address interleaving** |
| `test_llm_scheduler.cpp` | 39 | 405B parameter count, KV occupancy vs the paper, layout, hint timing |
| `test_macro_hazard.cpp` | 11 | identifier collisions with GPGPU-Sim's 435 macros |
| `test_h3_backend.cpp` | 15 | assembled-system invariants: an LHB hit must generate no device traffic; per-GPU values divided across partitions |

Total: **207 assertions**, all runnable without Ramulator or Accel-Sim.

Plus `tools/validate_trace.py`, which re-implements the Accel-Sim trace grammar
and its limits (`MAX_SRC`, `MAX_DST`, the version-dependent line layout) so a
malformed trace fails in one second with a named cause, rather than as an
assertion thousands of lines into a run. It runs automatically before every
smoke test.
