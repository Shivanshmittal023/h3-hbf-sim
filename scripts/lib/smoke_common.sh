# =============================================================================
#  scripts/lib/smoke_common.sh -- shared logic for the H3 smoke tests
# =============================================================================
#  Sourced by run_smoke_test.sh and run_smoke_test_hbm_only.sh. Both runs MUST
#  differ in exactly one thing -- the memory backend -- or the comparison is
#  meaningless. Keeping the shared machinery in one file is what guarantees
#  that: trace, base GPU config, cycle limit and environment are identical by
#  construction rather than by careful copy-paste.
# =============================================================================

H3_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ---- Locate Accel-Sim -------------------------------------------------------
# Two supported layouts:
#   development : h3-sim/ IS the accel-sim checkout (gpu-simulator/ at the root)
#   published   : accel-sim-framework/ is a git submodule beside h3-components/
# Auto-detected so the same scripts work in both without editing.
if [ -z "${ACCELSIM_DIR:-}" ]; then
  if [ -d "$H3_ROOT/gpu-simulator" ]; then
    ACCELSIM_DIR="$H3_ROOT"
  elif [ -d "$H3_ROOT/accel-sim-framework/gpu-simulator" ]; then
    ACCELSIM_DIR="$H3_ROOT/accel-sim-framework"
  else
    echo "ERROR: cannot find Accel-Sim. Expected gpu-simulator/ at $H3_ROOT" >&2
    echo "       or $H3_ROOT/accel-sim-framework/. Set ACCELSIM_DIR to override." >&2
    exit 1
  fi
fi
export ACCELSIM_DIR

# ---- Defaults (override via flags or environment) ---------------------------
: "${TRACE_DIR:=$H3_ROOT/traces/synthetic_h3_smoke_test}"
: "${BASE_CONFIG:=$ACCELSIM_DIR/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config}"
: "${MAX_CYCLE:=200000}"
: "${SIM_BIN:=$ACCELSIM_DIR/gpu-simulator/bin/release/accel-sim.out}"
: "${NUM_LAYERS:=4}"
: "${SEQ_LENGTH:=1024}"
: "${TRACE_MB:=8}"
# The computed HBF share at 1M context is ~0.999, which is faithful but leaves
# the HBM path almost untested. For the smoke test we deliberately rebalance so
# BOTH paths get real coverage. Experiments should use the computed default.
: "${HBF_FRACTION:=0.75}"
# Model config for the prefetch scheduler.
#
# The scheduler lays out EVERY tensor of the configured model and prefetches
# them. Pointing it at the full 405B config while the trace touches ~36 MiB
# would prefetch ~1.6 GB that is never read, making lhb_prefetch_efficiency
# meaningless (~0.04). configs/llama_smoke.yaml describes a small transformer
# whose HBF footprint matches the synthetic trace. Use the 405B config for
# real experiments on a large machine.
: "${MODEL_CONFIG:=$H3_ROOT/configs/llama_smoke.yaml}"

h3_die() { echo "ERROR: $*" >&2; exit 1; }

# Ramulator builds libramulator.so into its own source directory, which is on
# no standard loader path. Newly built binaries carry an RPATH for it (see
# h3-components/CMakeLists.txt), but adding it here as well means binaries
# built before that fix still run, and manual invocations work too.
h3_setup_library_path() {
  local ram
  if [ -d "$H3_ROOT/ramulator2" ]; then ram="$H3_ROOT/ramulator2"; else ram="$H3_ROOT/../ramulator2"; fi
  if [ -f "$ram/libramulator.so" ]; then
    export LD_LIBRARY_PATH="$ram${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
}
h3_setup_library_path

# Compose a backend config for this run.
#   $1 = output path
#   $2 = "lhb_on" | "lhb_off"
# Keeps the scheduler's num_layers equal to the trace's, so the two cannot
# drift apart and quietly ruin the prefetch statistics.
h3_make_backend_config() {
  local out="$1" lhb_mode="$2"
  local dir; dir="$(dirname "$out")"
  mkdir -p "$dir"

  local model="$dir/.model_smoke.yaml"
  sed "s/^  num_layers: .*/  num_layers: ${NUM_LAYERS}/" "$MODEL_CONFIG" > "$model"

  local lhb="$dir/.lhb.yaml"
  if [ "$lhb_mode" = "lhb_off" ]; then
    sed 's/^  enabled: true/  enabled: false/' "$H3_ROOT/configs/lhb_config.yaml" > "$lhb"
  else
    cp "$H3_ROOT/configs/lhb_config.yaml" "$lhb"
  fi

  sed -e "s|model_config: configs/llama_405b_config.yaml|model_config: $model|" \
      -e "s|lhb_config: configs/lhb_config.yaml|lhb_config: $lhb|" \
      "$H3_ROOT/configs/h3_backend_config.yaml" > "$out"
}

h3_hr() { printf '%s\n' "------------------------------------------------------------"; }

# ---- Preflight --------------------------------------------------------------
# Resolve the interconnect config referenced by the base GPU config.
#
# gpgpusim.config contains `-inter_config_file config_hopper_islip.icnt`, a
# path relative to the CONFIG's directory. The canonical Accel-Sim flow
# (run_simulations.py) copies both files into a per-run directory and runs
# there. We instead stay in the repo root -- the H3 YAML configs reference each
# other relatively, so the working directory has to be the root -- and override
# the option with an absolute path. Later options win, so this takes effect.
h3_icnt_override() {
  local cfg_dir name
  cfg_dir="$(cd "$(dirname "$BASE_CONFIG")" && pwd)"
  name="$(grep -E '^[[:space:]]*-inter_config_file' "$BASE_CONFIG" | tail -1 | awk '{print $2}')"
  [ -n "$name" ] || return 0
  if [ -f "$cfg_dir/$name" ]; then
    printf '%s\n' "-inter_config_file" "$cfg_dir/$name"
  else
    echo "WARNING: interconnect config '$name' not found in $cfg_dir" >&2
  fi
}

h3_preflight() {
  [ -x "$SIM_BIN" ] || h3_die "simulator not found at $SIM_BIN
       Build it first:
         source ./gpu-simulator/setup_environment.sh release
         cmake -S ./gpu-simulator/ -B ./gpu-simulator/build/release
         cmake --build ./gpu-simulator/build/release -j4
         cmake --install ./gpu-simulator/build/release"

  [ -f "$BASE_CONFIG" ] || h3_die "base GPU config not found: $BASE_CONFIG"
}

# ---- Trace generation (small, local, no downloads) --------------------------
h3_ensure_trace() {
  if [ -f "$TRACE_DIR/kernelslist.g" ] && [ "${FORCE_TRACE:-0}" != "1" ]; then
    echo "Using existing trace: $TRACE_DIR"
    h3_validate_trace
    return
  fi
  echo "Generating synthetic trace (${NUM_LAYERS} layers, seq ${SEQ_LENGTH}, <= ${TRACE_MB} MiB)..."
  python3 "$H3_ROOT/tools/generate_synthetic_trace.py" \
      --num_layers "$NUM_LAYERS" \
      --seq_length "$SEQ_LENGTH" \
      --target_trace_mb "$TRACE_MB" \
      --hbf_read_fraction "$HBF_FRACTION" \
      --output_dir "$TRACE_DIR"
  h3_validate_trace
}

# Validate before the simulator sees it. A malformed trace surfaces as an
# assertion deep inside trace_parser.cc; validate_trace.py reproduces the
# parser's grammar and limits and reports the offending line in one second.
h3_validate_trace() {
  python3 "$H3_ROOT/tools/validate_trace.py" "$TRACE_DIR" \
    || h3_die "trace validation failed -- fix the trace before simulating"
}

# ---- Run one simulation -----------------------------------------------------
#   $1 label            e.g. "h3" or "hbm_only"
#   $2 results dir
#   $3.. extra -config files / options appended AFTER the base config, so they
#        override it (option_parser applies files in order).
h3_run_sim() {
  local label="$1"; shift
  local outdir="$1"; shift

  mkdir -p "$outdir"
  local log="$outdir/sim.log"

  echo
  h3_hr
  echo "  Running: $label"
  echo "  trace      : $TRACE_DIR/kernelslist.g"
  echo "  base config: $(basename "$BASE_CONFIG")"
  echo "  extra      : $*"
  echo "  max cycles : $MAX_CYCLE"
  echo "  log        : $log"
  h3_hr

  # Run from the repo root: the H3 YAML configs reference each other by
  # relative path (configs/...), so the working directory matters.
  cd "$H3_ROOT"

  # Absolute-path override for the interconnect config (see h3_icnt_override).
  local icnt=()
  while IFS= read -r line; do icnt+=("$line"); done < <(h3_icnt_override)

  local start; start=$(date +%s)
  set +e
  "$SIM_BIN" \
      -config "$BASE_CONFIG" \
      ${icnt[@]+"${icnt[@]}"} \
      "$@" \
      -gpgpu_max_cycle "$MAX_CYCLE" \
      -trace "$TRACE_DIR/kernelslist.g" \
      > "$log" 2>&1
  local rc=$?
  set -e
  local end; end=$(date +%s)

  echo "$rc" > "$outdir/exit_code"
  echo "$((end - start))" > "$outdir/wall_seconds"

  if [ "$rc" -ne 0 ]; then
    echo "  SIMULATION FAILED (exit $rc)"
    # Surface the actual cause. GPGPU-Sim prints a long config dump on every
    # run, so a bare `tail` usually buries the one line that matters.
    local hits
    hits=$(grep -aniE 'could not open|cannot open|no such file|error|assert|abort|terminate called|segmentation fault|unknown option|fatal' "$log" \
           | grep -aviE 'error_rate|0 errors' | tail -8)
    if [ -n "$hits" ]; then
      echo "  --- likely cause ---"
      printf '%s\n' "$hits" | sed 's/^/    /'
    fi
    echo "  --- last 10 lines ---"
    tail -10 "$log" | sed 's/^/    /'
    echo "  (full log: $log)"
  else
    echo "  completed in $((end - start))s"
  fi
  return 0   # never abort the caller: we still want the summary
}

# ---- Summary ----------------------------------------------------------------
h3_summarize() {
  local label="$1" outdir="$2"
  local log="$outdir/sim.log"
  local rc; rc=$(cat "$outdir/exit_code" 2>/dev/null || echo "?")

  echo
  h3_hr
  echo "  SUMMARY: $label"
  h3_hr

  if [ ! -f "$log" ]; then
    echo "  no log produced"
    return
  fi

  # 1. Did it complete without crashing?
  if [ "$rc" = "0" ] && grep -qa "exit detected" "$log"; then
    echo "  completed cleanly     : YES"
  elif [ "$rc" = "0" ]; then
    echo "  completed cleanly     : reached the cycle limit (no 'exit detected')"
  else
    echo "  completed cleanly     : NO (exit code $rc)"
  fi

  local g
  g() { grep -a "^ *$1 *=" "$log" | tail -1 | sed 's/.*= *//' ; }

  local cyc insn
  cyc=$(g gpu_tot_sim_cycle); insn=$(g gpu_tot_sim_insn)
  echo "  gpu_tot_sim_cycle     : ${cyc}"
  echo "  gpu_tot_sim_insn      : ${insn}"

  # H3 runs disable GPGPU-Sim's deadlock detector (HBF's 20 us read exceeds its
  # 10 us threshold, so it fires on correct behaviour). Check forward progress
  # here instead: an IPC far below 0.01 over a full budget means the GPU is
  # stalled, not merely slow.
  if [ -n "$cyc" ] && [ -n "$insn" ] && [ "$cyc" -gt 0 ] 2>/dev/null; then
    if [ "$((insn * 1000 / cyc))" -lt 5 ]; then
      echo "  *** WARNING: IPC < 0.005 -- the GPU made almost no forward progress."
      echo "      Likely an unhidden HBF stall or a fill that never completed."
      echo "      Check lhb_fills_started vs lhb_fills_completed in the log."
    fi
  fi

  # 2/3/4. H3-specific counters (absent in the HBM-only run, which is expected)
  local hbm hbf
  hbm=$(g h3_router_requests_hbm); hbf=$(g h3_router_requests_hbf)
  if [ -n "$hbm$hbf" ]; then
    echo "  requests HBM / HBF    : ${hbm:-0} / ${hbf:-0}"
    echo "  LHB hit rate          : $(g lhb_hit_rate)"
    echo "  LHB avg stall (ns)    : $(g lhb_avg_stall_ns_per_miss)"
    echo "  HBF bandwidth (GB/s)  : $(g h3_router_hbf_bandwidth_gbps)"
  else
    echo "  (no H3 counters -- this is the HBM-only baseline)"
  fi

  # 5. Errors that indicate a broken configuration rather than a slow one.
  local werr uerr
  werr=$(g h3_router_write_attempts_to_hbf)
  uerr=$(g h3_router_unmapped_requests)
  echo "  writes to HBF (must be 0): ${werr:-n/a}"
  echo "  unmapped reqs (must be 0): ${uerr:-n/a}"
  if [ -n "$werr" ] && [ "$werr" != "0" ]; then
    echo "  *** ERROR: writes reached the read-only HBF region (data placement bug)"
  fi
  if [ -n "$uerr" ] && [ "$uerr" != "0" ]; then
    echo "  *** ERROR: addresses fell outside the H3 address map"
  fi

  if grep -qaiE "assert|segmentation fault|terminate called" "$log"; then
    echo "  *** ERROR: assertion or crash detected in the log"
  fi
}
