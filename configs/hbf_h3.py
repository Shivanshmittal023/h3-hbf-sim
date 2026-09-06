"""
===============================================================================
 configs/hbf_h3.py -- Ramulator 2.1 configuration for one H3 HBF cube
===============================================================================

WHAT THIS IS
    The source-of-record configuration for the HBF (High Bandwidth Flash)
    memory system described in the H3 paper. It instantiates the HBF device
    model, wraps it in a memory controller per channel, and assembles a
    16-channel memory system representing ONE HBF cube.

WHAT IT CONNECTS TO
    * `h3-components/ramulator_hbf/hbf.py` -- the HBF device model, which must
      first be installed into the ramulator2 submodule:
          ./scripts/install_hbf.sh
    * Ramulator's `External` frontend -- the entry point Accel-Sim drives via
      `receive_external_requests()`. Step 6's `h3_memory_backend` attaches here.
    * `h3-components/h3_address_router` (Step 3) sits UPSTREAM of this: it
      decides HBM-vs-HBF, and only HBF-bound requests reach this memory system.

WHY THIS IS A .py AND NOT A HAND-WRITTEN .yaml
    Ramulator 2.1 no longer ships hand-written YAML configs. `base/config.h`
    states configs must be "fully-specified, machine-generated YAML (produced
    by `python -m ramulator export`)" with no preset resolution or overrides
    performed in C++. Every timing constraint is pre-expanded into an integer
    array by the Python DSL. Hand-authoring that YAML would mean hand-computing
    ~60 constraint tuples, which is exactly the error-prone step the DSL exists
    to eliminate. So this file is the human-editable artifact; the YAML is
    generated from it:

        ./scripts/export_hbf_yaml.sh
        -> configs/generated/HBF_H3.yaml

USAGE
    python -m ramulator export configs/hbf_h3.py -o configs/generated/HBF_H3.yaml

CONFIGURABLE PARAMETERS  (all at the top of the file, under H3 PARAMETERS)
    TR_NS             NAND array read latency. THE primary sensitivity knob.
    CHANNELS_PER_CUBE Channels per HBF cube (16, matching an HBM3e stack).
    ORG_PRESET        Device organization; controls plane parallelism.
    READ/WRITE_BUFFER Controller queue depths. Must be deep enough to hold
                      enough outstanding reads to cover TR_NS -- see note below.
    CLOCK_RATIO       Ramulator ticks per frontend clock.
===============================================================================
"""

import ramulator
from ramulator.dram.hbf import HBF   # direct import: works before codegen regenerates dram/__init__.py

# =============================================================================
# H3 PARAMETERS
# =============================================================================
TR_NS = 20_000          # NAND array read latency (ns). Paper: 20 us.
TPROG_NS = 200_000      # NAND program latency (ns). Read-only path; unused.
RATE_MTPS = 8000        # HBF I/O rate (MT/s) -> 32 GB/s per pseudo-channel

CHANNELS_PER_CUBE = 16  # matches HBM3e channel count (16 ch x 2 pseudo-ch)
ORG_PRESET = "HBF_384Gb_16hi"   # 64 planes per PC; see hbf.py for the arithmetic

# Controller queue depths.
# IMPORTANT: to sustain full bandwidth the controller must keep enough reads
# outstanding to cover tR. Per pseudo-channel that is
#     32 GB/s x 20 us / 64 B request = ~10,000 requests
# which is unrealistic for a per-channel queue -- and that is precisely WHY the
# Latency Hiding Buffer (Step 4) exists. The LHB issues large sequential tensor
# prefetches far ahead of use, so the controller only ever sees a modest number
# of in-flight requests. These depths are sized for that regime, not for
# demand-miss traffic.
READ_BUFFER_SIZE = 64
WRITE_BUFFER_SIZE = 64
PRIORITY_BUFFER_SIZE = 1568

CLOCK_RATIO = 3

# =============================================================================
# DEVICE
# =============================================================================
# Sweep tR via register_timing_preset(), NOT via keyword overrides: Ramulator
# applies overrides after secondary-timing resolution, which would leave nRC
# stale. See the CAVEAT in hbf.py.
TIMING_PRESET = HBF.register_timing_preset(
    name=f"HBF_{RATE_MTPS}Mbps_tR{TR_NS // 1000}us",
    tR_ns=TR_NS,
    tPROG_ns=TPROG_NS,
    rate=RATE_MTPS,
)

# Fail at export time, with a named parameter, rather than at simulator startup
# with an opaque "what(): stoi" from deep inside Ramulator.
_probe = HBF(org_preset=ORG_PRESET, timing_preset=TIMING_PRESET)
HBF.check_int32(_probe.resolve()[1])

# =============================================================================
# CONTROLLERS -- one per HBF channel
# =============================================================================
# refresh_manager=NoRefresh: NAND flash never refreshes. The REF* commands
# exist in the HBF spec only because HBM34Controller resolves them by name at
# init; NoRefresh guarantees they are never issued.
#
# row_policy=Open: an open "row" is an open NAND page register. Keeping it open
# lets sequential reads within a 16 KB page avoid re-paying the 20 us tR, which
# is the dominant locality effect for the LHB's large sequential prefetches.
controllers = []
for _ch in range(CHANNELS_PER_CUBE):
    controllers.append(
        ramulator.controller.HBM34(
            dram=HBF(
                org_preset=ORG_PRESET,
                timing_preset=TIMING_PRESET,
                verbose=(_ch == 0),   # print resolved timings once
            ),
            scheduler=ramulator.scheduler.FRFCFS(),
            refresh_manager=ramulator.refresh_manager.NoRefresh(),
            row_policy=ramulator.row_policy.Open(),
            addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
            read_buffer_size=READ_BUFFER_SIZE,
            write_buffer_size=WRITE_BUFFER_SIZE,
            priority_buffer_size=PRIORITY_BUFFER_SIZE,
        )
    )

# =============================================================================
# MEMORY SYSTEM -- one HBF cube
# =============================================================================
memory_system = ramulator.memory_system.GenericDRAM(
    clock_ratio=CLOCK_RATIO,
    controllers=controllers,
    channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
)

# =============================================================================
# FRONTEND -- External: driven by Accel-Sim through the H3 memory backend
# =============================================================================
frontend = ramulator.frontend.External(clock_ratio=CLOCK_RATIO)

sim = ramulator.Simulation(frontend, memory_system)

if __name__ == "__main__":
    # `python -m ramulator export` monkey-patches Simulation and stops here.
    # Running this script directly would require the built native module.
    sim.run()
