#!/usr/bin/env bash
# =============================================================================
#  run_smoke_test.sh -- end-to-end smoke test of the H3 memory backend
# =============================================================================
#  WHAT THIS DOES
#      Runs Accel-Sim with the H3 hybrid memory backend enabled
#      (-gpgpu_memory_backend 2) on the small synthetic trace, for a bounded
#      number of cycles, and prints whether the whole pipeline works:
#
#        * did the simulation complete without crashing?
#        * how many requests went to HBM vs HBF?
#        * what was the LHB hit rate?
#        * were there any errors (writes to read-only HBF, unmapped addresses)?
#
#      It is a CORRECTNESS check, not a performance experiment. The trace is a
#      few MiB and models 4 transformer layers, not 126.
#
#  NO LARGE DOWNLOADS. The trace is generated locally by
#  tools/generate_synthetic_trace.py and is capped at a few MiB.
#
#  WHAT IT CONNECTS TO
#      scripts/lib/smoke_common.sh   shared runner (identical setup for both runs)
#      configs/accel_sim_h3.cfg      the H3 backend overlay
#      scripts/run_smoke_test_hbm_only.sh   the baseline to compare against
#      tools/parse_results.py        side-by-side comparison of the two
#
#  USAGE
#      ./scripts/run_smoke_test.sh                 # default: 4 layers, 200k cycles
#      ./scripts/run_smoke_test.sh --cycles 50000
#      ./scripts/run_smoke_test.sh --no-lhb        # ablation: H3 without the LHB
#      ./scripts/run_smoke_test.sh --layers 2 --regen
#
#  CONFIGURABLE PARAMETERS
#      --cycles N     cycle limit (-gpgpu_max_cycle)
#      --layers N     transformer layers in the synthetic trace
#      --seq N        sequence length in the synthetic trace
#      --trace-mb N   trace size cap in MiB
#      --no-lhb       run H3 with the latency hiding buffer disabled
#      --regen        regenerate the trace even if one exists
# =============================================================================
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/smoke_common.sh"

NO_LHB=0
LABEL="H3 (HBM + HBF + LHB)"
RESULTS="$H3_ROOT/results/smoke_test"

while [ $# -gt 0 ]; do
  case "$1" in
    --cycles)   MAX_CYCLE="$2"; shift 2 ;;
    --layers)   NUM_LAYERS="$2"; shift 2 ;;
    --seq)      SEQ_LENGTH="$2"; shift 2 ;;
    --trace-mb) TRACE_MB="$2"; shift 2 ;;
    --regen)    FORCE_TRACE=1; shift ;;
    --no-lhb)   NO_LHB=1; LABEL="H3 (LHB disabled -- ablation)"
                RESULTS="$H3_ROOT/results/smoke_test_no_lhb"; shift ;;
    -h|--help)  sed -n '2,42p' "$0"; exit 0 ;;
    *) h3_die "unknown option: $1" ;;
  esac
done

h3_preflight
h3_ensure_trace

# Both modes use a backend config composed for this run: the same scheduler
# model (matched to the trace) so that ONLY the LHB differs between them.
BACKEND_CFG="$RESULTS/.backend.yaml"
mkdir -p "$RESULTS"
if [ "$NO_LHB" = "1" ]; then
  h3_make_backend_config "$BACKEND_CFG" lhb_off
  echo "Ablation mode: LHB disabled (bypass). Every HBF access pays full tR."
else
  h3_make_backend_config "$BACKEND_CFG" lhb_on
fi
echo "Scheduler model: $MODEL_CONFIG (num_layers=$NUM_LAYERS, matched to the trace)"

EXTRA=("-config" "$H3_ROOT/configs/accel_sim_h3.cfg"
       "-gpgpu_h3_config_file" "$BACKEND_CFG")

mkdir -p "$RESULTS"
h3_run_sim "$LABEL" "$RESULTS" "${EXTRA[@]}"
h3_summarize "$LABEL" "$RESULTS"

echo
echo "Results in: $RESULTS"
echo "Next: ./scripts/run_smoke_test_hbm_only.sh   then   python3 tools/parse_results.py"
