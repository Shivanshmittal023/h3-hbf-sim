#!/usr/bin/env bash
# =============================================================================
# install_hbf.sh -- install the H3 HBF device model into the ramulator2 tree
# =============================================================================
# WHAT THIS DOES
#   Copies h3-components/ramulator_hbf/hbf.py into
#   ramulator2/python/ramulator/dram/hbf.py, where Ramulator's codegen can
#   discover it (codegen enumerates that directory with pkgutil.iter_modules).
#
# WHY A SCRIPT INSTEAD OF EDITING RAMULATOR DIRECTLY
#   ramulator2/ is a git submodule pinned to the upstream CMU-SAFARI repo. Our
#   HBF model is H3 project code and must live in OUR repository so it is
#   version-controlled, reviewable, and survives a submodule update. This
#   script is the one-line bridge between the two.
#
# WHEN TO RUN IT
#   * After the first clone (before building Ramulator)
#   * After editing h3-components/ramulator_hbf/hbf.py
#   * After updating the ramulator2 submodule
#
# WHAT HAPPENS NEXT
#   Ramulator's CMake runs `python -m ramulator codegen` as a build dependency,
#   which turns hbf.py into src/ramulator/dram/impl/HBF.cpp and regenerates
#   python/ramulator/dram/__init__.py to export HBF.
#
# USAGE
#   ./scripts/install_hbf.sh            # copy (default)
#   ./scripts/install_hbf.sh --link     # symlink instead, for live editing
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/h3-components/ramulator_hbf/hbf.py"
DST_DIR="$ROOT/ramulator2/python/ramulator/dram"
DST="$DST_DIR/hbf.py"

[ -f "$SRC" ] || { echo "ERROR: missing $SRC" >&2; exit 1; }
[ -d "$DST_DIR" ] || {
  echo "ERROR: $DST_DIR not found." >&2
  echo "       Is the ramulator2 submodule initialised? Try:" >&2
  echo "         git submodule update --init --recursive" >&2
  exit 1
}

if [ "${1:-}" = "--link" ]; then
  ln -sf "$SRC" "$DST"
  echo "Symlinked $DST -> $SRC"
else
  cp "$SRC" "$DST"
  echo "Copied hbf.py -> $DST"
fi

# ---------------------------------------------------------------------------
# 1. Generate the C++ device model from the Python DSL.
#
# Ramulator normally runs codegen as a CMake target, but ONLY when
# RAMULATOR_PYTHON_BINDINGS is ON. We build with it OFF (it needs nanobind and
# Python headers we do not otherwise require), so codegen must run here.
# Without it, src/ramulator/dram/impl/HBF.cpp never exists and the simulator
# dies at startup with "Unknown DRAM standard: HBF".
#
# Codegen is pure Python -- no build, no native module.
# ---------------------------------------------------------------------------
cd "$ROOT/ramulator2" 2>/dev/null || cd "$ROOT/../ramulator2"
echo "Generating C++ device models from the Python DSL..."
PYTHONPATH=python python3 -m ramulator codegen --src-dir src/ramulator \
    | tail -2 | sed 's/^/  /'

[ -f src/ramulator/dram/impl/HBF.cpp ] || {
  echo "ERROR: codegen did not produce src/ramulator/dram/impl/HBF.cpp" >&2
  exit 1
}

# ---------------------------------------------------------------------------
# 2. Add the generated file to Ramulator's build.
#
# src/ramulator/dram/CMakeLists.txt lists its sources EXPLICITLY, so a newly
# generated model is not picked up automatically. Insert it next to the other
# HBM models. Idempotent: re-running changes nothing.
# ---------------------------------------------------------------------------
DRAM_CMAKE="src/ramulator/dram/CMakeLists.txt"
if grep -q "impl/HBF.cpp" "$DRAM_CMAKE"; then
  echo "  HBF.cpp already in $DRAM_CMAKE"
else
  # Portable in-place edit (macOS sed and GNU sed disagree about -i).
  awk '{ print; if ($0 ~ /impl\/HBM1\.cpp/) print "  impl/HBF.cpp" }' \
      "$DRAM_CMAKE" > "$DRAM_CMAKE.tmp" && mv "$DRAM_CMAKE.tmp" "$DRAM_CMAKE"
  grep -q "impl/HBF.cpp" "$DRAM_CMAKE" \
    && echo "  added impl/HBF.cpp to $DRAM_CMAKE" \
    || { echo "ERROR: could not add HBF.cpp to $DRAM_CMAKE" >&2; exit 1; }
fi

# ---------------------------------------------------------------------------
# 3. Validate the DSL. Pure Python (no native module), so it catches structural
#    errors long before the C++ build would.
# ---------------------------------------------------------------------------
PYTHONPATH=python python3 - <<'PY'
from ramulator.dram.hbf import HBF
HBF.validate()
d = HBF(org_preset="HBF_384Gb_16hi", timing_preset="HBF_8000Mbps_tR20us")
org, tim = d.resolve()
d.to_config()
planes = org["sid"] * org["bankgroup"] * org["bank"]
page = org["column"] * org["dq"] // 8
tR_us = tim["nRCDRD"] * tim["tCK_ps"] / 1e6
print(f"  HBF model OK: tR={tR_us:.0f}us, {planes} planes/PC, {page//1024}KB page")
PY
echo
echo "Done. HBF is generated, registered and in the build."
echo "Next: configure with -DH3_WITH_RAMULATOR=ON and build."
