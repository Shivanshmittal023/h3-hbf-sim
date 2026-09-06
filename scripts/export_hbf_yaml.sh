#!/usr/bin/env bash
# =============================================================================
# export_hbf_yaml.sh -- generate the machine-readable Ramulator HBF config
# =============================================================================
# Ramulator 2.1 consumes only fully-expanded, machine-generated YAML (see
# ramulator2/src/ramulator/base/config.h). This script turns the human-editable
# configs/hbf_h3.py into that YAML.
#
# Output: configs/generated/HBF_H3.yaml
#
# Requires only Python (no build): `ramulator export` monkey-patches
# Simulation, so the native extension module is never imported.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/configs/generated"
OUT="$OUT_DIR/HBF_H3.yaml"

mkdir -p "$OUT_DIR"
cd "$ROOT/ramulator2"
PYTHONPATH=python python3 -m ramulator export "$ROOT/configs/hbf_h3.py" -o "$OUT"
echo "Wrote $OUT ($(wc -l < "$OUT") lines)"
