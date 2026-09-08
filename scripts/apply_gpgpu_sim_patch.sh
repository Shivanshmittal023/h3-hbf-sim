#!/usr/bin/env bash
# =============================================================================
# apply_gpgpu_sim_patch.sh -- add (or remove) the H3 backend hook in GPGPU-Sim
# =============================================================================
# The GPGPU-Sim tree is cloned from upstream by setup_environment.sh and is not
# tracked by this repository, so the H3 hook lives here as a patch and is
# applied on demand. See patches/gpgpu-sim-h3-backend.patch for what it changes
# and why.
#
#   ./scripts/apply_gpgpu_sim_patch.sh            apply
#   ./scripts/apply_gpgpu_sim_patch.sh --revert   undo
#   ./scripts/apply_gpgpu_sim_patch.sh --check    dry run
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Support both layouts (see scripts/lib/smoke_common.sh for the rationale).
if [ -n "${ACCELSIM_DIR:-}" ]; then
  GPGPUSIM="$ACCELSIM_DIR/gpu-simulator/gpgpu-sim"
elif [ -d "$ROOT/gpu-simulator/gpgpu-sim" ]; then
  GPGPUSIM="$ROOT/gpu-simulator/gpgpu-sim"
else
  GPGPUSIM="$ROOT/accel-sim-framework/gpu-simulator/gpgpu-sim"
fi
PATCH="$ROOT/patches/gpgpu-sim-h3-backend.patch"

[ -d "$GPGPUSIM/src/gpgpu-sim" ] || {
  echo "ERROR: GPGPU-Sim not found at $GPGPUSIM" >&2
  echo >&2
  echo "  GPGPU-Sim is not a submodule -- Accel-Sim CLONES it during setup, so" >&2
  echo "  setup_environment.sh has to run before this patch can be applied:" >&2
  echo >&2
  echo "    source ${ACCELSIM_DIR:-<accel-sim>}/gpu-simulator/setup_environment.sh release" >&2
  echo "    ./scripts/apply_gpgpu_sim_patch.sh" >&2
  exit 1
}

MODE="apply"
case "${1:-}" in
  --revert) MODE="revert" ;;
  --check)  MODE="check" ;;
  "")       ;;
  *) echo "usage: $0 [--revert|--check]" >&2; exit 2 ;;
esac

cd "$GPGPUSIM"
case "$MODE" in
  check)  patch -p1 --dry-run < "$PATCH" ;;
  revert) patch -p1 -R < "$PATCH" && echo "H3 hook removed from GPGPU-Sim." ;;
  apply)
    if patch -p1 --dry-run --reverse --force < "$PATCH" >/dev/null 2>&1; then
      echo "H3 hook is already applied; nothing to do."
    else
      patch -p1 < "$PATCH"
      echo "H3 hook applied. Rebuild the simulator to pick it up."
    fi
    ;;
esac
