#!/usr/bin/env bash
# =============================================================================
#  run_smoke_test_hbm_only.sh -- HBM-only baseline for comparison against H3
# =============================================================================
#  WHAT THIS DOES
#      Runs the SAME trace, the SAME GPU config and the SAME cycle limit as
#      run_smoke_test.sh, changing exactly one thing: the memory backend is
#      GPGPU-Sim's stock dram_t (-gpgpu_memory_backend 0) instead of H3.
#
#  WHAT THE BASELINE ACTUALLY REPRESENTS
#      With the stock backend there is no HBM/HBF split: every address --
#      including those in the HBF region -- is served by the HBM timing model.
#      So this baseline is an IDEALISED HBM-ONLY MACHINE: one with enough HBM to
#      hold the entire 3 TB working set at full HBM speed.
#
#      Such a machine cannot be built, which is the whole premise of the paper.
#      It is still the right smoke-test baseline because it is an upper bound:
#      H3 cannot beat it on latency, only approach it. If H3 lands close, the
#      latency hiding is working. A REAL HBM-only comparison at 3 TB requires
#      scaling out to many more GPUs and belongs in the full experiment on a
#      large machine, not in a laptop smoke test.
#
#      For the complementary ablation -- H3 with the buffer switched off -- use
#          ./scripts/run_smoke_test.sh --no-lhb
#
#  USAGE
#      ./scripts/run_smoke_test_hbm_only.sh [--cycles N] [--layers N] [--regen]
# =============================================================================
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/smoke_common.sh"

RESULTS="$H3_ROOT/results/smoke_test_hbm_only"

while [ $# -gt 0 ]; do
  case "$1" in
    --cycles)   MAX_CYCLE="$2"; shift 2 ;;
    --layers)   NUM_LAYERS="$2"; shift 2 ;;
    --seq)      SEQ_LENGTH="$2"; shift 2 ;;
    --trace-mb) TRACE_MB="$2"; shift 2 ;;
    --regen)    FORCE_TRACE=1; shift ;;
    -h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
    *) h3_die "unknown option: $1" ;;
  esac
done

h3_preflight
h3_ensure_trace   # same trace as the H3 run: identical input is the point

mkdir -p "$RESULTS"
# Backend 0 = stock dram_t. No -config overlay, so nothing else differs.
h3_run_sim "HBM-only baseline (stock dram_t)" "$RESULTS" \
    "-gpgpu_memory_backend" "0"
h3_summarize "HBM-only baseline (stock dram_t)" "$RESULTS"

echo
echo "Results in: $RESULTS"
echo "Next: python3 tools/parse_results.py"
