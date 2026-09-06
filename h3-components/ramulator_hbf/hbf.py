"""
===============================================================================
 hbf.py -- High Bandwidth Flash (HBF) device model for Ramulator 2.1
===============================================================================

WHAT THIS IS
    A Ramulator `DRAMStandard` describing the HBF (High Bandwidth Flash) memory
    device from the H3 paper: an SLC-NAND stack that presents an HBM-compatible
    electrical interface but has ~20 us array read latency instead of ~20 ns.

WHAT IT CONNECTS TO
    * Ramulator's DRAM DSL: subclasses `ramulator.dram.spec.DRAMStandard`,
      exactly like `python/ramulator/dram/hbm3.py`.
    * Ramulator codegen: when this file is installed into
      `ramulator2/python/ramulator/dram/hbf.py`, `python -m ramulator codegen`
      discovers it via `pkgutil.iter_modules()` and emits
      `src/ramulator/dram/impl/HBF.cpp` plus an updated `dram/__init__.py`.
      DO NOT hand-edit the generated .cpp -- it is overwritten on every build.
    * Ramulator's `HBM34` memory controller, which is reused unmodified. That
      controller calls `spec.get_command_id()` for ACT / PREpb / PREab / REFab,
      which is why the refresh commands below are declared even though NAND
      flash never refreshes. Pair it with `refresh_manager=NoRefresh()` so they
      are never actually issued.
    * `h3-sim/configs/hbf_h3.py`, which instantiates this standard.

    Install with: h3-sim/scripts/install_hbf.sh
    (kept outside the ramulator2 submodule so the submodule stays pristine)

HOW NAND IS MAPPED ONTO A DRAM-CENTRIC MODEL
    The key insight is that a NAND page read already has DRAM's two-phase
    structure, so the mapping is natural rather than forced:

        NAND operation                     DRAM command    Timing parameter
        ---------------------------------  --------------  ----------------
        READ cmd + array -> page register  ACT             nRCDRD = tR = 20 us
        page register -> I/O bus           RD              nCL, nBL (bus rate)
        release page register              PREpb           nRP (short)
        page register -> array (program)   WR + nWR        nWR = tPROG = 200 us

    Level names are reinterpreted (structure is identical, meaning differs):

        Ramulator level   HBF meaning
        ---------------   -----------------------------------------------
        Channel           HBF channel (16 per cube, = HBM3e channel count)
        PseudoChannel     32-bit pseudo-channel (2 per channel)
        Sid               NAND die group within the 16-high stack
        BankGroup         die within a group
        Bank              NAND PLANE  <-- the unit of array-level parallelism
        Row               NAND page (wordline), 16 KB
        Column            byte offset within the page register

    Bandwidth is sustained *despite* 20 us tR purely by plane-level
    parallelism, which is the mechanism the paper describes. See the
    "BANDWIDTH SUSTAINABILITY" arithmetic below.

CONFIGURABLE PARAMETERS
    Everything lives in `HBF.org_presets` and `HBF.timing_presets` at the
    bottom of this file. The two intended for sensitivity sweeps are:
        tR      -- nRCDRD / nRAS  (NAND array read latency; the whole point)
        bank    -- planes per pseudo-channel (parallelism available to hide tR)
    Any preset field can also be overridden per-instantiation, e.g.
        HBF(org_preset=..., timing_preset=..., bank=8, nRCDRD=80000)

    CAVEAT -- do not use overrides to sweep tR. Ramulator's
    `DRAMStandard.resolve()` applies overrides AFTER calling
    `resolve_secondary_timings()`, so an overridden nRAS/nRCDRD leaves the
    derived nRC stale and the device becomes internally inconsistent. (This
    affects every standard in Ramulator, not just HBF.) For latency sweeps use
    `HBF.register_timing_preset()` below, which builds a complete, consistent
    preset that resolve() then processes normally.

UNITS
    All timing_presets values are in CK cycles, per Ramulator convention.
    `tick_multiplier = 2` means the engine internally uses half-CK ticks; the
    DSL performs that conversion once, in `spec.to_config()`. Do not pre-scale.
===============================================================================
"""

import math

from ramulator.dram.spec import DRAMStandard, TimingConstraint


class HBF(DRAMStandard):
    name = "HBF"

    internal_prefetch_size = 8   # BL8, matching the HBM-compatible PHY
    data_payload_bytes = 32      # one 32-bit pseudo-channel x BL8 = 32 B
    tick_multiplier = 2          # half-CK ticks, same convention as HBM3
    read_latency = "nCL + nBL"

    # ---- Hierarchy (level name -> init state) ----
    # Structurally identical to HBM3 so the HBM34 controller can drive it.
    levels = {
        "Channel":       "N_A",
        "PseudoChannel": "N_A",
        "Sid":           "N_A",   # NAND die group
        "BankGroup":     "N_A",   # die within group
        "Bank":          "Closed",  # NAND plane
        "Row":           "Closed",  # NAND page
        "Column":        "N_A",
    }

    # ---- Commands ----
    # REFab/REFpb/RFMab/RFMpb are declared ONLY because HBM34Controller looks
    # them up by name at init. NAND has no refresh; use NoRefresh so they are
    # never issued, and see the deliberately inert refresh timings below.
    commands = [
        "ACT", "PREpb", "PREab",
        "RD", "WR", "RDA", "WRA",
        "REFab", "REFpb",
        "RFMab", "RFMpb",
    ]

    command_cycles = {
        "ACT": 1.5,
        "PREpb": 0.5, "PREab": 0.5,
        "REFab": 0.5, "REFpb": 0.5,
        "RFMab": 0.5, "RFMpb": 0.5,
    }

    row_commands = ["ACT", "PREpb", "PREab", "REFab", "REFpb", "RFMab", "RFMpb"]
    column_commands = ["RD", "WR", "RDA", "WRA"]

    states = ["Opened", "Closed", "N_A"]

    timing_params = [
        "rate", "nBL", "nCL", "nRCDRD", "nRCDWR",
        "nRP", "nRAS", "nRC", "nWR", "nRTP", "nCWL",
        "nCCDS", "nCCDL", "nCCDR",
        "nRRDS", "nRRDL",
        "nWTRS", "nWTRL", "nRTW",
        "nFAW", "nPPD",
        "nRFC", "nRFCpb", "nRFMab", "nRFMpb",
        "nRREFD",
        "nREFI", "nREFIpb",
        "tCK_ps",
    ]

    supported_requests = {"Read": "RD", "Write": "WR"}

    # ---- Timing constraints ----
    # Mirrors HBM3's constraint graph: the *shape* of the timing relationships
    # is unchanged (ACT->RD, ACT->PRE, PRE->ACT, bus turnaround, ...). Only the
    # magnitudes differ, and they come from timing_presets. Reusing the proven
    # HBM3 graph avoids inventing a NAND command protocol Ramulator cannot
    # schedule.
    timing_constraints = [
        # ---------- PseudoChannel ----------
        TimingConstraint(level="PseudoChannel", preceding=["RD", "RDA"], following=["RD", "RDA"], latency="nBL"),
        TimingConstraint(level="PseudoChannel", preceding=["WR", "WRA"], following=["WR", "WRA"], latency="nBL"),
        TimingConstraint(level="PseudoChannel", preceding=["RD", "RDA"], following=["WR", "WRA"], latency="nRTW"),
        TimingConstraint(level="PseudoChannel", preceding=["WR", "WRA"], following=["RD", "RDA"], latency="nCWL + nBL + nWTRS"),
        TimingConstraint(level="PseudoChannel", preceding=["RD", "RDA"], following=["PREab"], latency="nRTP"),
        TimingConstraint(level="PseudoChannel", preceding=["WR", "WRA"], following=["PREab"], latency="nCWL + nBL + nWR"),
        TimingConstraint(level="PseudoChannel", preceding=["ACT"], following=["ACT"], latency="nRRDS"),
        TimingConstraint(level="PseudoChannel", preceding=["ACT", "REFpb", "RFMpb"], following=["ACT", "REFpb", "RFMpb"], latency="nFAW", window=4, shared_window=True),
        TimingConstraint(level="PseudoChannel", preceding=["ACT"], following=["PREab"], latency="nRAS"),
        TimingConstraint(level="PseudoChannel", preceding=["PREab"], following=["ACT"], latency="nRP"),
        TimingConstraint(level="PseudoChannel", preceding=["PREpb", "PREab"], following=["PREpb", "PREab"], latency="nPPD"),
        TimingConstraint(level="PseudoChannel", preceding=["ACT"], following=["REFab"], latency="nRC"),
        TimingConstraint(level="PseudoChannel", preceding=["PREpb", "PREab"], following=["REFab"], latency="nRP"),
        TimingConstraint(level="PseudoChannel", preceding=["RDA"], following=["REFab"], latency="nRP + nRTP"),
        TimingConstraint(level="PseudoChannel", preceding=["WRA"], following=["REFab"], latency="nCWL + nBL + nWR + nRP"),
        TimingConstraint(level="PseudoChannel", preceding=["REFab"], following=["ACT", "PREab", "REFab", "REFpb", "RFMab", "RFMpb"], latency="nRFC"),
        TimingConstraint(level="PseudoChannel", preceding=["REFpb"], following=["REFpb", "RFMpb", "ACT"], latency="nRREFD"),
        TimingConstraint(level="PseudoChannel", preceding=["REFpb"], following=["REFab", "RFMab"], latency="nRFCpb"),
        TimingConstraint(level="PseudoChannel", preceding=["ACT"], following=["REFpb"], latency="nRRDS"),
        TimingConstraint(level="PseudoChannel", preceding=["ACT"], following=["RFMab"], latency="nRC"),
        TimingConstraint(level="PseudoChannel", preceding=["PREpb", "PREab"], following=["RFMab"], latency="nRP"),
        TimingConstraint(level="PseudoChannel", preceding=["RDA"], following=["RFMab"], latency="nRP + nRTP"),
        TimingConstraint(level="PseudoChannel", preceding=["WRA"], following=["RFMab"], latency="nCWL + nBL + nWR + nRP"),
        TimingConstraint(level="PseudoChannel", preceding=["RFMab"], following=["ACT", "PREab", "REFab", "REFpb", "RFMab", "RFMpb"], latency="nRFMab"),
        TimingConstraint(level="PseudoChannel", preceding=["RFMpb"], following=["REFpb", "RFMpb", "ACT"], latency="nRREFD"),
        TimingConstraint(level="PseudoChannel", preceding=["RFMpb"], following=["REFab", "RFMab"], latency="nRFMpb"),
        TimingConstraint(level="PseudoChannel", preceding=["ACT"], following=["RFMpb"], latency="nRRDS"),

        # ---------- Sid (die group) ----------
        TimingConstraint(level="Sid", preceding=["RD", "RDA"], following=["RD", "RDA"], latency="nCCDS"),
        TimingConstraint(level="Sid", preceding=["WR", "WRA"], following=["WR", "WRA"], latency="nCCDS"),
        TimingConstraint(level="Sid", preceding=["RD", "RDA"], following=["RD", "RDA"], latency="nCCDR", sibling=True),

        # ---------- BankGroup (die) ----------
        TimingConstraint(level="BankGroup", preceding=["RD", "RDA"], following=["RD", "RDA"], latency="nCCDL"),
        TimingConstraint(level="BankGroup", preceding=["WR", "WRA"], following=["WR", "WRA"], latency="nCCDL"),
        TimingConstraint(level="BankGroup", preceding=["WR", "WRA"], following=["RD", "RDA"], latency="nCWL + nBL + nWTRL"),
        TimingConstraint(level="BankGroup", preceding=["ACT"], following=["ACT"], latency="nRRDL"),
        TimingConstraint(level="BankGroup", preceding=["ACT"], following=["REFpb", "RFMpb"], latency="nRRDL"),
        TimingConstraint(level="BankGroup", preceding=["REFpb", "RFMpb"], following=["ACT"], latency="nRRDL"),

        # ---------- Bank (NAND plane) ----------
        # This is where tR lives: ACT -> RD is the array read.
        TimingConstraint(level="Bank", preceding=["ACT"], following=["ACT"], latency="nRC"),
        TimingConstraint(level="Bank", preceding=["ACT"], following=["RD", "RDA"], latency="nRCDRD"),
        TimingConstraint(level="Bank", preceding=["ACT"], following=["WR", "WRA"], latency="nRCDWR"),
        TimingConstraint(level="Bank", preceding=["ACT"], following=["PREpb"], latency="nRAS"),
        TimingConstraint(level="Bank", preceding=["PREpb"], following=["ACT"], latency="nRP"),
        TimingConstraint(level="Bank", preceding=["RD"], following=["PREpb"], latency="nRTP"),
        TimingConstraint(level="Bank", preceding=["WR"], following=["PREpb"], latency="nCWL + nBL + nWR"),
        TimingConstraint(level="Bank", preceding=["RDA"], following=["ACT"], latency="nRTP + nRP"),
        TimingConstraint(level="Bank", preceding=["WRA"], following=["ACT"], latency="nCWL + nBL + nWR + nRP"),
        TimingConstraint(level="Bank", preceding=["REFpb"], following=["REFpb", "RFMpb", "ACT"], latency="nRFCpb"),
        TimingConstraint(level="Bank", preceding=["ACT"], following=["REFpb"], latency="nRC"),
        TimingConstraint(level="Bank", preceding=["PREpb"], following=["REFpb"], latency="nRP"),
        TimingConstraint(level="Bank", preceding=["RFMpb"], following=["REFpb", "RFMpb", "ACT"], latency="nRFMpb"),
        TimingConstraint(level="Bank", preceding=["ACT"], following=["RFMpb"], latency="nRC"),
        TimingConstraint(level="Bank", preceding=["PREpb"], following=["RFMpb"], latency="nRP"),
    ]

    # ---- Secondary timing resolution ----
    @classmethod
    def resolve_secondary_timings(cls, timing_dict, org_dict):
        tCK_ps = timing_dict["tCK_ps"]

        # ACT-to-ACT on the same plane: array read + page-register release.
        timing_dict["nRC"] = timing_dict["nRAS"] + timing_dict["nRP"]

        # Column-to-column, same die: bus-limited, identical to HBM3.
        timing_dict["nCCDL"] = max(4, math.ceil(2_500 / tCK_ps))
        timing_dict["nCCDR"] = (
            timing_dict["nCCDS"] if org_dict["sid"] == 1 else 3
        )
        timing_dict["nRTW"] = cls._resolve_nRTW(timing_dict, tCK_ps)

        # ---- Refresh: NAND flash does not refresh. ----
        # These commands are declared only to satisfy HBM34Controller's
        # get_command_id() lookups and are never issued under NoRefresh.
        # Values are set inert: minimal blocking latency, effectively infinite
        # interval. If you ever swap in a real refresh manager, these become
        # meaningful and MUST be revisited.
        timing_dict["nRFC"] = 1
        timing_dict["nRFCpb"] = 1
        timing_dict["nRFMab"] = 1
        timing_dict["nRFMpb"] = 1
        timing_dict["nRREFD"] = 1
        # "Never fires", but it must still fit in a 32-bit int: Ramulator parses
        # timings with std::stoi, so anything above 2^31-1 throws
        # std::out_of_range deep inside DRAMSpec::load_config, with only
        # "what(): stoi" to go on. tick_multiplier doubles these on export, so
        # the exported value is 1e9 ticks -- 0.25 s of simulated time at
        # tCK = 250 ps, i.e. never, with 2x headroom under the limit.
        timing_dict["nREFI"] = 500_000_000
        timing_dict["nREFIpb"] = 500_000_000

    @staticmethod
    def _resolve_nRTW(timing_dict, tCK_ps):
        # Same derivation as HBM3 (JESD238 Tables 92/93, Note 18) because the
        # HBF I/O layer is electrically HBM-compatible.
        base_cycles = timing_dict["nCL"] + timing_dict["nBL"] - timing_dict["nCWL"]
        analog_numerator = max(-2_000, -2 * tCK_ps) + 5 * tCK_ps + 10 * (2_500 + 20)
        return base_cycles + math.ceil(analog_numerator / (10 * tCK_ps))

    @classmethod
    def register_timing_preset(cls, name, tR_ns, tPROG_ns=200_000, rate=8000,
                               base="HBF_8000Mbps_tR20us"):
        """Build and register a complete, self-consistent HBF timing preset.

        This is the supported way to sweep NAND array latency. It writes a
        whole preset (so `resolve_secondary_timings` recomputes nRC, nRTW and
        friends from it) rather than patching individual values afterwards.

        Args:
            name:     preset name to register, e.g. "HBF_8000Mbps_tR30us"
            tR_ns:    NAND array read latency in ns (20_000 = 20 us)
            tPROG_ns: NAND program latency in ns (read-only path; rarely matters)
            rate:     data rate in MT/s; tCK_ps is derived as 4e6 / rate
            base:     preset to inherit all unrelated timings from

        Returns the preset name, so it can be used inline:
            HBF(org_preset=..., timing_preset=HBF.register_timing_preset("t30", 30_000))
        """
        tCK_ps = 4_000_000 // rate
        preset = dict(cls.timing_presets[base])
        preset["rate"] = rate
        preset["tCK_ps"] = tCK_ps
        n_tR = int(round(tR_ns * 1000 / tCK_ps))
        preset["nRCDRD"] = n_tR   # ACT -> RD : the array read
        preset["nRAS"] = n_tR     # page register held for the whole read
        preset["nWR"] = int(round(tPROG_ns * 1000 / tCK_ps))
        cls.timing_presets[name] = preset
        return name

    @classmethod
    def check_int32(cls, timing_dict):
        """Reject timings Ramulator's std::stoi cannot parse.

        Called from to_config() via validate_organization's sibling path is not
        possible, so hbf_h3.py calls it explicitly. Failing here names the
        offending parameter; failing in C++ gives only "what(): stoi".
        """
        INT32_MAX = 2 ** 31 - 1
        limit = INT32_MAX // max(1, cls.tick_multiplier)   # value is scaled on export
        bad = {k: v for k, v in timing_dict.items()
               if isinstance(v, int) and v > limit}
        if bad:
            raise ValueError(
                f"HBF: timing values exceed what Ramulator can parse "
                f"(std::stoi, int32, and tick_multiplier={cls.tick_multiplier} "
                f"doubles them on export): {bad}. Keep every timing below {limit}."
            )

    @classmethod
    def validate_organization(cls, org_dict):
        super().validate_organization(org_dict)

        # NAND page size must equal the row size implied by the column count.
        page_bytes = org_dict["column"] * org_dict["dq"] // 8
        if page_bytes != org_dict.get("page_bytes", page_bytes):
            raise ValueError(
                f"HBF: column count implies a {page_bytes} B page, but "
                f"page_bytes={org_dict['page_bytes']} was declared."
            )

        # Warn loudly if plane parallelism cannot cover tR. This is the single
        # most important structural property of the device: see the bandwidth
        # arithmetic in the module docstring.
        planes = org_dict["sid"] * org_dict["bankgroup"] * org_dict["bank"]
        if planes < 8:
            raise ValueError(
                f"HBF: {planes} planes per pseudo-channel is far too few to "
                f"hide a ~20 us tR; bandwidth will collapse. Expected >= 40."
            )


# =============================================================================
# ORGANIZATION PRESETS
# =============================================================================
# BANDWIDTH SUSTAINABILITY (why these numbers, per the H3 paper's mechanism)
#
#   Per pseudo-channel bus rate : 32 dq x 8.0 GT/s        = 32.0 GB/s
#   Per cube                    : 32 PC x 32.0 GB/s       = 1.024 TB/s
#   Per GPU (8 cubes)           : 8 x 1.024 TB/s          = 8.19 TB/s   [paper: 8 TB/s]
#
#   To sustain 32 GB/s per PC through a 20 us array latency, enough page reads
#   must be in flight to cover the latency-bandwidth product:
#       bytes in flight  = 32.0 GB/s x 20 us              = 640 KB
#       pages in flight  = 640 KB / 16 KB page            = 40 planes minimum
#   This organization provides sid(4) x bankgroup(4) x bank(4) = 64 planes per
#   pseudo-channel, i.e. 1.6x headroom:
#       achievable       = 64 x 16 KB / 20 us             = 52.4 GB/s > 32 GB/s
#   Physically this is a 16-high stack x 4 planes/die = 64 planes. Dropping
#   `bank` below 3 (=48 planes) will make HBF bandwidth-starved -- that is a
#   real effect worth sweeping, not a bug.
#
# CAPACITY
#   row size   = 4096 columns x 4 B      = 16 KB   (one NAND page)
#   plane      = 16384 rows x 16 KB      = 256 MB
#   per PC     = 64 planes x 256 MB      = 16 GB
#   per cube   = 32 PC x 16 GB           = 512 GB
#   per GPU    = 8 cubes x 512 GB        = 4 TB addressable
#
#   NOTE: the paper specifies 3 TB per GPU (16x a 192 GB HBM3e system). 3 TB is
#   not a power of two, and Ramulator's address decoder slices power-of-two
#   fields, so the device is given 4 TB of *addressable* space. The H3 address
#   router (h3_address_router) exposes exactly 3 TB; addresses above that are
#   never generated. Over-provisioning the decode space has no timing effect.
# =============================================================================
HBF.org_presets = {
    # Baseline: one HBF cube channel = 2 pseudo-channels, 16-high SLC stack.
    "HBF_384Gb_16hi": {
        "die_density": 393216,     # Mbit per die (48 GB), metadata only
        "channel_density": 32768,  # Mbit per channel, metadata only
        "stack_height": 16,        # 16-high NAND stack
        "page_bytes": 16384,       # NAND page = 16 KB (must match column x dq/8)
        "dq": 32,
        "channel_width": 64,       # 2 x 32-bit pseudo-channels
        "pseudochannel": 2,
        "sid": 4,                  # die groups
        "bankgroup": 4,            # dies per group
        "bank": 4,                 # PLANES per die  <-- parallelism knob
        "row": 1 << 14,            # 16384 pages per plane
        "column": (1 << 9) << 3,   # 4096 -> 16 KB page
    },

    # Reduced-parallelism variant for sensitivity analysis: 32 planes per PC,
    # deliberately BELOW the 40 needed to cover tR. Expect bandwidth collapse.
    "HBF_384Gb_16hi_lowpar": {
        "die_density": 393216,
        "channel_density": 32768,
        "stack_height": 16,
        "page_bytes": 16384,
        "dq": 32,
        "channel_width": 64,
        "pseudochannel": 2,
        "sid": 2,
        "bankgroup": 4,
        "bank": 4,
        "row": 1 << 15,
        "column": (1 << 9) << 3,
    },
}

# =============================================================================
# TIMING PRESETS  (all values in CK cycles unless the name ends in _ps)
# =============================================================================
# tCK_ps follows the HBM3 convention in this codebase: tCK_ps = 4e6 / rate,
# so BL8 spans nBL = 2 CK.  For rate = 8000 MT/s -> tCK_ps = 500 ps.
#
#   tR    = 20 us  = 40000 CK   -> nRCDRD, nRAS      <-- PRIMARY SWEEP KNOB
#   tPROG = 200 us = 400000 CK  -> nWR
#
# HBF is read-only in the H3 use case (model weights + shared precomputed KV
# cache). tPROG is modeled for completeness but should never be exercised; the
# H3 address router rejects writes to the HBF region and counts them as errors.
# Write endurance is likewise not modeled: at read-only usage it is irrelevant.
# =============================================================================
HBF.timing_presets = {
    "HBF_8000Mbps_tR20us": {
        "rate": 8000,
        "tCK_ps": 500,          # 4e6 / 8000

        "nBL": 2,               # BL8 over 2 CK
        "nCL": 40,              # page register -> DQ, 20 ns
        "nCWL": 20,             # 10 ns

        "nRCDRD": 40_000,       # tR  = 20 us   <-- NAND ARRAY READ
        "nRAS": 40_000,         # page register must be held for the whole read
        "nRCDWR": 20,           # data load into page register is fast
        "nWR": 400_000,         # tPROG = 200 us (read-only path; unused)

        "nRP": 200,             # page register release, 100 ns
        "nRTP": 4,
        "nCCDS": 2,             # bus-limited, not array-limited
        "nRRDS": 4,             # ACT-to-ACT across planes: command-bus limited
        "nRRDL": 8,
        "nFAW": 16,             # non-binding here (ACTs are ~312 ns apart)
        "nWTRS": 8,
        "nWTRL": 12,
        "nPPD": 2,
    },

    # Sensitivity variants: same device, different array latency.
    "HBF_8000Mbps_tR10us": {
        "rate": 8000, "tCK_ps": 500,
        "nBL": 2, "nCL": 40, "nCWL": 20,
        "nRCDRD": 20_000, "nRAS": 20_000, "nRCDWR": 20, "nWR": 400_000,
        "nRP": 200, "nRTP": 4, "nCCDS": 2, "nRRDS": 4, "nRRDL": 8,
        "nFAW": 16, "nWTRS": 8, "nWTRL": 12, "nPPD": 2,
    },
    "HBF_8000Mbps_tR40us": {
        "rate": 8000, "tCK_ps": 500,
        "nBL": 2, "nCL": 40, "nCWL": 20,
        "nRCDRD": 80_000, "nRAS": 80_000, "nRCDWR": 20, "nWR": 400_000,
        "nRP": 200, "nRTP": 4, "nCCDS": 2, "nRRDS": 4, "nRRDL": 8,
        "nFAW": 16, "nWTRS": 8, "nWTRL": 12, "nPPD": 2,
    },
}
