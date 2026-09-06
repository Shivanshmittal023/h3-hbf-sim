"""
===============================================================================
 configs/hbm3e_h3.py -- Ramulator 2.1 configuration for one H3 HBM3e stack
===============================================================================

WHAT THIS IS
    The fast tier of H3: an HBM3e stack holding the generated KV cache and
    activations. Companion to configs/hbf_h3.py, which describes the slow,
    high-capacity tier. Both must be exported before running the simulator with
    `device_backend: ramulator`.

WHAT IT CONNECTS TO
    * Ramulator's built-in HBM3 device model (python/ramulator/dram/hbm3.py).
      HBM3e is HBM3 at a higher data rate, so no new device model is needed --
      only a timing preset. Contrast with HBF, which needed a whole new model.
    * h3-components/h3_memory_backend.cpp, via `ramulator.hbm_config` in
      configs/h3_backend_config.yaml.
    * The H3 address router, which exposes the HBM region this stack backs.

WHY THE PRESET IS REGISTERED HERE RATHER THAN EDITED INTO RAMULATOR
    ramulator2/ is a pinned git submodule. Adding presets from our own config
    file keeps it untouched, exactly as scripts/install_hbf.sh keeps the HBF
    model in this repository.

TIMING DERIVATION
    Ramulator ships HBM3_6400Mbps (tCK = 625 ps). HBM3e here runs at 8000 MT/s
    (tCK = 500 ps), matching HBF's data rate -- matched bandwidth between the
    two tiers is the paper's central premise; only LATENCY differs, by ~57x.

    Every timing is converted to absolute time at 6400 and re-quantised to the
    new tCK, rounding UP so no constraint is accidentally relaxed:

        param     6400 CK      ns    8000 CK
        nCL            20   12.50         25
        nRCDRD         31   19.38         39
        nRCDWR         15    9.38         19
        nRP            26   16.25         33
        nRAS           45   28.12         57
        nWR            33   20.62         42
        nRTP            9    5.62         12
        nCWL           10    6.25         13
        nRRDS           4    2.50          5
        nRRDL           5    3.12          7
        nFAW           24   15.00         30
        nWTRS           7    4.38          9
        nWTRL          10    6.25         13

    nBL, nCCDS and nPPD stay in CK: they are bus-structural, not analog.
    Ramulator's own resolve_secondary_timings() derives nRC, nRTW, nCCDL and
    every refresh timing from these.

CAPACITY AND BANDWIDTH
    org preset HBM3_32Gb_8hi (32 Gb dies, 8-high):
        per pseudo-channel   1.00 GiB
        per channel (2 PC)   2 GiB
        per stack (16 ch)    32 GiB
        per GPU (8 stacks)   256 GiB addressable

    NOTE: the paper specifies 192 GB per GPU. 192 is not a power of two and
    Ramulator's decoder slices power-of-two fields, so the device is given
    256 GiB of decode space and the H3 router exposes exactly 192 GB. Addresses
    above that are never generated. This is the same accommodation made for HBF
    (3 TB exposed out of 4 TB addressable); see docs/hbf_model.md.

    bandwidth: 32 dq x 8.0 GT/s = 32.0 GB/s per PC
               x32 PC = 1.024 TB/s per stack
               x8 stacks = 8.19 TB/s per GPU     [paper: 8 TB/s]

USAGE
    ./scripts/export_hbf_yaml.sh          # exports this and the HBF config

CONFIGURABLE PARAMETERS
    RATE_MTPS          data rate; drives tCK and therefore every timing
    CHANNELS_PER_STACK 16, matching HBM3e
    ORG_PRESET         organization (capacity and bank structure)
===============================================================================
"""

import ramulator
from ramulator.dram.hbm3 import HBM3

# =============================================================================
# H3 PARAMETERS
# =============================================================================
RATE_MTPS = 8000                  # matched to HBF; see the note above
CHANNELS_PER_STACK = 16
ORG_PRESET = "HBM3_32Gb_8hi"

READ_BUFFER_SIZE = 64
WRITE_BUFFER_SIZE = 64
PRIORITY_BUFFER_SIZE = 1568
CLOCK_RATIO = 3

# =============================================================================
# TIMING PRESET
# =============================================================================
# tCK_ps follows this codebase's HBM convention: tCK_ps = 4e6 / rate, so BL8
# spans nBL = 2 CK.
_TCK_PS = 4_000_000 // RATE_MTPS
_BASE = HBM3.timing_presets["HBM3_6400Mbps"]
_BASE_TCK_PS = _BASE["tCK_ps"]

_STRUCTURAL = {"rate", "tCK_ps", "nBL", "nCCDS", "nPPD"}


def _rescale(cycles):
    """Convert a CK count at the base rate to the equivalent at RATE_MTPS.

    Rounds UP: quantising down would silently relax a timing constraint and
    make the device look faster than the standard allows.
    """
    ps = cycles * _BASE_TCK_PS
    return -(-ps // _TCK_PS)          # ceiling division


TIMING_PRESET = f"HBM3E_{RATE_MTPS}Mbps"
HBM3.timing_presets[TIMING_PRESET] = {
    **{k: v for k, v in _BASE.items() if k in _STRUCTURAL},
    **{k: _rescale(v) for k, v in _BASE.items() if k not in _STRUCTURAL},
    "rate": RATE_MTPS,
    "tCK_ps": _TCK_PS,
}

# =============================================================================
# CONTROLLERS -- one per HBM3e channel
# =============================================================================
# refresh_manager=AllBank and row_policy=Open are the standard HBM choices.
# Unlike HBF, HBM3e genuinely does refresh, so a real refresh manager is used.
controllers = []
for _ch in range(CHANNELS_PER_STACK):
    controllers.append(
        ramulator.controller.HBM34(
            dram=HBM3(
                org_preset=ORG_PRESET,
                timing_preset=TIMING_PRESET,
                verbose=(_ch == 0),
            ),
            scheduler=ramulator.scheduler.FRFCFS(),
            refresh_manager=ramulator.refresh_manager.AllBank(),
            row_policy=ramulator.row_policy.Open(),
            addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
            read_buffer_size=READ_BUFFER_SIZE,
            write_buffer_size=WRITE_BUFFER_SIZE,
            priority_buffer_size=PRIORITY_BUFFER_SIZE,
        )
    )

memory_system = ramulator.memory_system.GenericDRAM(
    clock_ratio=CLOCK_RATIO,
    controllers=controllers,
    channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
)

# External frontend: driven by Accel-Sim through the H3 memory backend.
frontend = ramulator.frontend.External(clock_ratio=CLOCK_RATIO)

sim = ramulator.Simulation(frontend, memory_system)

if __name__ == "__main__":
    sim.run()
