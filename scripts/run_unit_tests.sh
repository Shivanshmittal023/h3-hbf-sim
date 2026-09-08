#!/usr/bin/env bash
# =============================================================================
# run_unit_tests.sh -- build and run the standalone H3 component unit tests
# =============================================================================
# These tests need only a C++17 compiler: no Ramulator, no Accel-Sim, no CMake.
# That is deliberate -- H3 component logic must be verifiable in seconds,
# independently of the ~30 min simulator build.
#
# Usage:  ./scripts/run_unit_tests.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p build/tests

CXX="${CXX:-g++}"
FLAGS="-std=c++17 -O2 -Wall -Wextra -I h3-components"

fail=0
for test_src in tests/test_*.cpp; do
  name="$(basename "$test_src" .cpp)"
  # Each test links the components it needs; build them all, it is cheap.
  srcs="$test_src"
  for comp in h3-components/*.cpp; do
    srcs="$srcs $comp"
  done

  echo "=== building $name ==="
  # shellcheck disable=SC2086
  $CXX $FLAGS $srcs -o "build/tests/$name"

  echo "=== running $name ==="
  if ! "./build/tests/$name"; then
    fail=1
  fi
done

# The LLM trace generator and the C++ prefetch scheduler independently compute
# the same tensor layout. If they drift apart every prefetch hint points at
# bytes the trace never reads, the hit rate silently goes to zero, and nothing
# reports an error. Check that they still agree.
echo "=== checking generator/scheduler layout agreement ==="
if ! python3 "$ROOT/tools/check_layout_agreement.py"; then
  fail=1
fi

exit $fail
