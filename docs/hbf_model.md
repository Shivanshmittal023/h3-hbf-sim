# HBF Memory Model — Design Notes, Approximations, and Sensitivity

Companion to `h3-components/ramulator_hbf/hbf.py` and `configs/hbf_h3.py`.
Covers Phase 2 Step 2c: what was approximated, what would need to change for a
NAND-native model, and which parameters matter for sensitivity analysis.

---

## 1. Where the model lives, and why

Ramulator 2.1 generates its C++ DRAM models from a Python DSL:

```
h3-components/ramulator_hbf/hbf.py        ← we write this (source of record)
        │  scripts/install_hbf.sh
        ▼
ramulator2/python/ramulator/dram/hbf.py   ← installed copy (submodule, untracked)
        │  python -m ramulator codegen     (run automatically by CMake)
        ▼
ramulator2/src/ramulator/dram/impl/HBF.cpp ← generated; DO NOT EDIT
```

`src/ramulator/dram/impl/HBM3.cpp` carries an explicit `AUTO-GENERATED FILE — DO
NOT EDIT` banner and is regenerated on every build. Hand-writing `HBF.cpp` would
be silently overwritten, so the DSL is the only correct place to define HBF.

---

## 2. The NAND → DRAM mapping

A NAND page read already has DRAM's two-phase structure, so the mapping is
natural rather than forced:

| NAND operation | DRAM command | Timing parameter | Value |
|---|---|---|---|
| READ cmd + array → page register | `ACT` | `nRCDRD`, `nRAS` | **tR = 20 µs** |
| page register → I/O bus | `RD` | `nCL`, `nBL` | 20 ns, bus-rate |
| release page register | `PREpb` | `nRP` | 100 ns |
| page register → array (program) | `WR` | `nWR` | tPROG = 200 µs |

Level names are reinterpreted; the structure is unchanged:

| Ramulator level | HBF meaning |
|---|---|
| Channel | HBF channel (16 per cube, matching HBM3e) |
| PseudoChannel | 32-bit pseudo-channel (2 per channel) |
| Sid | NAND die group in the 16-high stack |
| BankGroup | die within a group |
| **Bank** | **NAND plane** — the unit of array-level parallelism |
| Row | NAND page (wordline), 16 KB |
| Column | byte offset within the page register |

### Why bandwidth survives a 20 µs latency

This is the crux of the device, and it is pure plane-level parallelism:

```
per-PC bus bandwidth      32 dq × 8.0 GT/s              = 32.0 GB/s
bytes that must be in flight to cover tR
                          32.0 GB/s × 20 µs            = 640 KB
pages in flight           640 KB / 16 KB page          = 40 planes minimum

provided                  sid(4) × bankgroup(4) × bank(4) = 64 planes
achievable                64 × 16 KB / 20 µs           = 52.4 GB/s  ✓ 1.64× headroom
```

Aggregates: 32 PC × 32 GB/s = **1.024 TB/s per cube**; × 8 cubes = **8.19 TB/s
per GPU** (paper: 8 TB/s). Physically, 64 planes = a 16-high stack × 4
planes/die.

---

## 3. Approximations made

These are deliberate, and each is a place a reviewer will look.

1. **NAND is modeled with a DRAM command protocol.** HBF issues
   `ACT`/`RD`/`PRE`, not NAND's native `00h-30h` read sequence with a
   ready/busy signal. This is accurate for *timing* — the observable behaviour
   is "command, then 20 µs, then a page is available at bus rate" — but it does
   not reproduce NAND command-bus encoding or R/B# polling overhead.

2. **The refresh command set is vestigial.** `REFab`/`REFpb`/`RFMab`/`RFMpb` are
   declared purely because `HBM34Controller` resolves them by name via
   `spec.get_command_id()` at init and would fail without them. NAND never
   refreshes. They are paired with `refresh_manager=NoRefresh()` and given inert
   timings (`nRFC=1`, `nREFI=2^40`). **If a real refresh manager is ever
   substituted, these values become live and are wrong.**

3. **Addressable capacity is 4 TB/GPU, not 3 TB.** The paper's 3 TB (16 × 192 GB)
   is not a power of two, and Ramulator's address decoder slices power-of-two
   fields. The device is therefore given 512 GB/cube of decode space. The H3
   address router exposes exactly 3 TB; higher addresses are never generated.
   Over-provisioning decode space has no timing effect.

4. **No program/erase modeling beyond `nWR`.** HBF is read-only in H3 (model
   weights + shared precomputed KV cache). `tPROG` is present for completeness;
   the address router rejects and counts writes to the HBF region. There is no
   erase block, no garbage collection, no program-suspend.

5. **Write endurance is not modeled.** Justified by the read-only use case, and
   explicitly non-critical per the paper.

6. **No NAND-specific error behaviour.** No ECC latency, no read-retry, no
   read-disturb, no temperature or wear-dependent tR variation. Real SLC tR has
   a distribution; here it is a constant.

7. **Uniform tR across all planes.** No per-die or per-block variation.

8. **Plane conflicts are modeled as bank conflicts.** Real multi-plane NAND
   operations often require the same page offset across planes; that constraint
   is not modeled, so plane parallelism here is slightly optimistic.

---

## 4. What a NAND-native model would require

If the approximations above prove load-bearing, the escalation path is:

- **A NAND command set** in the DSL: replace `ACT`/`RD` with
  `READ_CMD`/`ARRAY_READ`/`CACHE_READ`/`PROGRAM`/`ERASE`, plus a `Busy` device
  state and R/B# semantics. The DSL supports arbitrary command and state names,
  so this is additive — but it needs a matching controller, since
  `HBM34Controller` hardcodes `ACT`/`PREpb`/`PREab`/`REFab`.
- **A custom `HBFController`** deriving from `HBMControllerBase`
  (`src/ramulator/controller/impl/hbm_controller_base.h`), overriding
  `slot_matches()` for NAND's asymmetric command/data timing.
- **Cache-read (pipelined) mode**: real NAND overlaps the next array read with
  the current page's data-out, effectively halving the exposed tR for sequential
  streams. This would *improve* HBF's numbers and is currently omitted — the
  model is conservative here.
- **Erase-block structure and a distributional tR** if write traffic is ever
  introduced.
- **Multi-plane addressing constraints** (same-offset requirement) to remove the
  optimism noted in approximation 8.

---

## 5. Sensitivity analysis — ranked by expected impact

Sweep tR with `HBF.register_timing_preset()`, **not** keyword overrides.
Ramulator's `DRAMStandard.resolve()` applies overrides *after*
`resolve_secondary_timings()`, so an overridden `nRAS` leaves the derived `nRC`
stale. The helper writes a complete preset instead. (This affects every standard
in Ramulator, not only HBF.)

| Rank | Parameter | Where | Default | Why it matters |
|---|---|---|---|---|
| 1 | **tR** | `TR_NS` in `configs/hbf_h3.py` | 20 µs | Sets the LHB capacity requirement (Eq. 1) and the entire latency-hiding premise. Directly trades against LHB SRAM area. |
| 2 | **planes per PC** (`bank`) | `ORG_PRESET` in `hbf.py` | 64 | Below ~40 planes the device cannot cover tR and bandwidth collapses. This is a cliff, not a gradient. |
| 3 | **LHB capacity** | Step 4 config | 40 MB/cube | Coupled to tR × BW. Under-provisioning reintroduces the full 20 µs stall. |
| 4 | **prefetch lead time** | Step 5 config | — | Determines whether the 20 µs is actually hidden. Too short → misses; too long → buffer pressure. |
| 5 | **page size** (`column`) | `org_presets` | 16 KB | Larger pages amortise tR better but waste bandwidth on partial reads. |
| 6 | **rate / tCK** | `RATE_MTPS` | 8000 MT/s | Scales bandwidth, and so rescales the planes needed to cover tR. |
| 7 | **read_buffer_size** | `configs/hbf_h3.py` | 64 | Only binding if the LHB is disabled; see the note in that file. |
| 8 | tPROG, endurance | — | — | Inert under read-only use. Sweep only if a write path is added. |

**Provided sweep points.** `hbf.py` ships `HBF_8000Mbps_tR10us` / `tR20us` /
`tR40us` and a reduced-parallelism org preset `HBF_384Gb_16hi_lowpar` (32
planes). Verified behaviour of the combinations:

| org | tR | planes | needed | verdict |
|---|---|---|---|---|
| `HBF_384Gb_16hi` | 10 µs | 64 | 19.5 | sustains |
| `HBF_384Gb_16hi` | 20 µs | 64 | 39.1 | sustains (baseline, 1.64×) |
| `HBF_384Gb_16hi` | 40 µs | 64 | 78.1 | **starved** |
| `..._lowpar` | 10 µs | 32 | 19.5 | sustains |
| `..._lowpar` | 20 µs | 32 | 39.1 | **starved** |

The starved rows are intentional: they are the experiment showing that HBF
bandwidth is a parallelism property, not a device-rate property.
