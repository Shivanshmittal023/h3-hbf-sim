#!/usr/bin/env bash
# =============================================================================
# export_hbf_yaml.sh -- generate the machine-readable Ramulator configs
# =============================================================================
# Ramulator 2.1 consumes only fully-expanded, machine-generated YAML (see
# ramulator2/src/ramulator/base/config.h): every timing constraint is
# pre-computed into integer arrays by the Python DSL. This script turns the
# human-editable configs into that form.
#
#   configs/hbf_h3.py    ->  configs/generated/HBF_H3.yaml     (slow tier)
#   configs/hbm3e_h3.py  ->  configs/generated/HBM3E_H3.yaml   (fast tier)
#
# BOTH are required by device_backend: ramulator -- the backend builds one
# memory system per tier.
#
# Requires only Python, no build: `ramulator export` monkey-patches Simulation,
# so the native extension module is never imported.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/configs/generated"
mkdir -p "$OUT_DIR"

if [ -d "$ROOT/ramulator2" ]; then
  RAM_DIR="$ROOT/ramulator2"
else
  RAM_DIR="$ROOT/../ramulator2"
fi
[ -d "$RAM_DIR/python" ] || {
  echo "ERROR: Ramulator not found at $RAM_DIR" >&2
  echo "       Run: git submodule update --init --recursive" >&2
  exit 1
}

# The HBF device model must be installed before its config can be exported.
if [ ! -f "$RAM_DIR/python/ramulator/dram/hbf.py" ]; then
  echo "HBF model not installed; running scripts/install_hbf.sh first."
  "$ROOT/scripts/install_hbf.sh"
fi

cd "$RAM_DIR"
for pair in "hbf_h3.py:HBF_H3.yaml" "hbm3e_h3.py:HBM3E_H3.yaml"; do
  src="${pair%%:*}"
  dst="${pair#*:}"
  echo "=== exporting $src -> configs/generated/$dst ==="
  PYTHONPATH=python python3 -m ramulator export "$ROOT/configs/$src" -o "$OUT_DIR/$dst"
  echo "    $(wc -l < "$OUT_DIR/$dst") lines"
done

echo
echo "Done. Both tiers exported to $OUT_DIR"
