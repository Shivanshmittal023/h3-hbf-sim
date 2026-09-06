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

# Validate the DSL immediately. This is pure Python (no native module needed),
# so it catches structural errors long before the C++ build would.
cd "$ROOT/ramulator2"
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
echo "Done. Next: build Ramulator (codegen runs automatically)."
