#!/usr/bin/env bash
# =============================================================================
#  run_real_trace.sh -- run the H3 memory system on a REAL Accel-Sim trace
# =============================================================================
#  WHY THIS IS SEPARATE FROM run_smoke_test.sh
#      The smoke test uses a trace we generate ourselves, deliberately placed in
#      the H3 address regions. A real trace uses whatever addresses CUDA
#      allocated (around 0x7f35ab700000), which fall in NEITHER region, so the
#      router would report every request unmapped and the HBF path would never
#      run.
#
#      This script closes that gap in three steps:
#        1. classify  - work out which buffers are read-only bulk data
#        2. screen    - refuse to simulate if nothing lands in HBF
#        3. simulate  - run with the router in allocation-map mode
#
#  THE SCREEN IS THE POINT
#      Simulating a trace with no read-only bulk data tells you nothing about
#      H3 and can cost hours. Classification takes seconds. Step 2 stops early
#      and says so, rather than producing a meaningless run.
#
#      Measured on the rodinia traces: HBF traffic share 0.0-0.5%, i.e. rodinia
#      cannot exercise H3 at all. That result took 4 seconds to obtain.
#
#  USAGE
#      ./scripts/run_real_trace.sh <trace_dir> [--cycles N] [--min-hbf-share 5]
#      ./scripts/run_real_trace.sh <trace_dir> --classify-only
#
#  EXAMPLE
#      ./scripts/run_real_trace.sh \
#          hw_run/cudasdk/9.1/matrixMul/.../traces --cycles 200000
# =============================================================================
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/smoke_common.sh"

TRACE=""
CLASSIFY_ONLY=0
MIN_SHARE=1.0
HBF_MIN_BYTES=65536

while [ $# -gt 0 ]; do
  case "$1" in
    --cycles)         MAX_CYCLE="$2"; shift 2 ;;
    --classify-only)  CLASSIFY_ONLY=1; shift ;;
    --min-hbf-share)  MIN_SHARE="$2"; shift 2 ;;
    --hbf-min-bytes)  HBF_MIN_BYTES="$2"; shift 2 ;;
    -h|--help)        sed -n '2,32p' "$0"; exit 0 ;;
    -*)               h3_die "unknown option: $1" ;;
    *)                TRACE="$1"; shift ;;
  esac
done

[ -n "$TRACE" ] || h3_die "usage: $0 <trace_dir> [--cycles N]"
[ -d "$TRACE" ] || h3_die "no such trace directory: $TRACE"
TRACE="$(cd "$TRACE" && pwd)"
NAME="$(basename "$(dirname "$(dirname "$TRACE")")")"
[ "$NAME" = "." ] && NAME="trace"

RESULTS="$H3_ROOT/results/real_${NAME}"
MAP="$H3_ROOT/configs/generated/${NAME}_map.yaml"
mkdir -p "$RESULTS" "$(dirname "$MAP")"

# ---- 1. Validate the trace before anything else -----------------------------
h3_hr
echo "  Real-trace run: $NAME"
echo "  trace: $TRACE"
h3_hr
python3 "$H3_ROOT/tools/validate_trace.py" "$TRACE" \
    --hbm-base 0 --hbm-size-gb 192 --hbf-base 0x3000000000 --hbf-size-tb 3 \
    2>&1 | sed 's/^/  /' || true
echo "  (unmapped addresses are EXPECTED here -- a real trace does not use the"
echo "   H3 address map. That is exactly what the classifier fixes.)"

# ---- 2. Classify buffers into HBM and HBF -----------------------------------
echo
h3_hr
echo "  Classifying buffers"
h3_hr
CLS=$(python3 "$H3_ROOT/tools/classify_trace.py" "$TRACE" -o "$MAP" \
        --hbf-min-bytes "$HBF_MIN_BYTES" 2>/dev/null)
echo "$CLS" | sed 's/^/  /'

SHARE=$(echo "$CLS" | awk '/traffic share to HBF/ {gsub("%","",$6); print $6}')
SHARE="${SHARE:-0}"

# ---- 3. Screen: is this workload worth simulating at all? -------------------
if awk "BEGIN{exit !($SHARE < $MIN_SHARE)}"; then
  echo
  echo "  STOPPING: only ${SHARE}% of traffic would reach HBF (threshold ${MIN_SHARE}%)."
  echo "  This workload has almost no large read-only data, so an H3 run would"
  echo "  measure nothing. Pick a trace with big read-only buffers -- matrix"
  echo "  multiply, inference, or anything weight-driven."
  echo
  echo "  Override with --min-hbf-share 0 if you want to run it anyway."
  exit 2
fi

[ "$CLASSIFY_ONLY" = "1" ] && { echo; echo "  --classify-only: stopping here."; exit 0; }

# ---- 4. Simulate with the router in allocation-map mode ---------------------
h3_preflight
BACKEND_CFG="$RESULTS/.backend.yaml"
h3_make_backend_config "$BACKEND_CFG" lhb_on

ROUTER_CFG="$RESULTS/.router.yaml"
sed "s|^address_map:|address_map:\n  allocation_map: $MAP|" \
    "$H3_ROOT/configs/h3_router_config.yaml" > "$ROUTER_CFG"
sed -i.bak "s|router_config: configs/h3_router_config.yaml|router_config: $ROUTER_CFG|" \
    "$BACKEND_CFG" && rm -f "$BACKEND_CFG.bak"

TRACE_DIR="$TRACE"
h3_run_sim "H3 on real trace: $NAME" "$RESULTS" \
    "-config" "$H3_ROOT/configs/accel_sim_h3.cfg" \
    "-gpgpu_h3_config_file" "$BACKEND_CFG"
h3_summarize "H3 on real trace: $NAME" "$RESULTS"

echo
echo "  allocation map : $MAP"
echo "  results        : $RESULTS"
