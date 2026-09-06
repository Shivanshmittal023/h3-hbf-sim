#!/usr/bin/env bash
# =============================================================================
#  prepare_release_repo.sh -- build a clean, publishable H3 repository
# =============================================================================
#  WHY THIS EXISTS
#      The development tree is itself a clone of accel-sim-framework: `origin`
#      points at accel-sim, its 48 MB of history is in .git, and its content
#      sits at the repository ROOT. Content at a repo root cannot be a submodule
#      of that same repo, so the working tree can never satisfy "accel-sim and
#      ramulator2 as submodules" in place.
#
#      This script therefore ASSEMBLES a separate, clean repository containing
#      only H3 project files, with both upstreams as submodules. The working
#      tree is left untouched, so an in-progress build is never disturbed.
#
#  RESULT LAYOUT
#      <target>/
#        accel-sim-framework/   submodule -> accel-sim/accel-sim-framework
#        ramulator2/            submodule -> CMU-SAFARI/ramulator2
#        h3-components/ configs/ scripts/ tools/ tests/ docs/ patches/
#        Dockerfile.h3sim  docker-compose.yml  README.md  .gitignore
#
#      The scripts detect either layout (see scripts/lib/smoke_common.sh), so
#      they work unchanged in both the development tree and the release repo.
#
#  USAGE
#      ./scripts/prepare_release_repo.sh [target_dir]      # default ../h3-sim-release
#      ./scripts/prepare_release_repo.sh --no-submodules   # skip network fetch
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-$ROOT/../h3-sim-release}"
ADD_SUBMODULES=1
[ "${1:-}" = "--no-submodules" ] && { ADD_SUBMODULES=0; TARGET="$ROOT/../h3-sim-release"; }
[ "${2:-}" = "--no-submodules" ] && ADD_SUBMODULES=0

ACCELSIM_URL="https://github.com/accel-sim/accel-sim-framework.git"
RAMULATOR_URL="https://github.com/CMU-SAFARI/ramulator2.git"

# Pin the exact upstream commits this project was developed and tested against.
ACCELSIM_REF="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo dev)"
RAMULATOR_REF="$(git -C "$ROOT/ramulator2" rev-parse HEAD 2>/dev/null || echo main)"

echo "Assembling a clean H3 release repository"
echo "  source : $ROOT"
echo "  target : $TARGET"
echo "  accel-sim pinned at   : $ACCELSIM_REF"
echo "  ramulator2 pinned at  : $RAMULATOR_REF"
echo

[ -e "$TARGET" ] && { echo "ERROR: $TARGET already exists; remove it or pick another path" >&2; exit 1; }
mkdir -p "$TARGET"

# ---- H3 project files (everything that is genuinely ours) -------------------
for d in h3-components configs scripts tools tests docs patches; do
  cp -R "$ROOT/$d" "$TARGET/$d"
done
for f in Dockerfile.h3sim docker-compose.yml .dockerignore .gitignore README.md requirements.txt; do
  [ -f "$ROOT/$f" ] && cp "$ROOT/$f" "$TARGET/$f"
done

# Never ship generated or local artefacts.
rm -rf "$TARGET"/configs/generated
# Upstream Accel-Sim documentation and images: redundant once accel-sim is a
# submodule, and docs/img/ is ~700 KB of upstream assets.
rm -rf "$TARGET"/docs/img "$TARGET"/docs/accel-sim-README.md
rm -rf "$TARGET"/**/__pycache__ 2>/dev/null || true
find "$TARGET" -name '.DS_Store' -delete 2>/dev/null || true

cd "$TARGET"
git init -q
git symbolic-ref HEAD refs/heads/main

if [ "$ADD_SUBMODULES" = "1" ]; then
  echo "Adding submodules (requires network)..."
  git submodule add -q "$ACCELSIM_URL" accel-sim-framework
  git submodule add -q "$RAMULATOR_URL" ramulator2
  git -C accel-sim-framework checkout -q "$ACCELSIM_REF" 2>/dev/null || \
    echo "  note: could not pin accel-sim to $ACCELSIM_REF; leaving on its default branch"
  git -C ramulator2 checkout -q "$RAMULATOR_REF" 2>/dev/null || \
    echo "  note: could not pin ramulator2 to $RAMULATOR_REF; leaving on its default branch"
  git add .gitmodules accel-sim-framework ramulator2
else
  # Record the intended submodules even when the network is unavailable, so the
  # file is reviewable and `git submodule update --init` works after pushing.
  cat > .gitmodules <<GITMOD
[submodule "accel-sim-framework"]
	path = accel-sim-framework
	url = $ACCELSIM_URL
[submodule "ramulator2"]
	path = ramulator2
	url = $RAMULATOR_URL
GITMOD
  git add .gitmodules
  echo "  (--no-submodules: wrote .gitmodules without fetching)"
fi

git add -A
git -c user.name="H3 Simulator" -c user.email="noreply@example.com" \
    commit -q -m "H3 hybrid HBM+HBF memory simulator

Simulation environment for the H3 architecture (IEEE CAL 2026), built on
Accel-Sim/GPGPU-Sim 4.x with a Ramulator 2.1 HBF device model.

Components:
  - HBF device model (SLC NAND, tR = 20 us, plane-level parallelism)
  - H3 address router (HBM/HBF split, D2D hop, read-only enforcement)
  - Latency Hiding Buffer (double-buffered SRAM prefetch, Eq. 1)
  - LLM prefetch scheduler (deterministic transformer access schedule)
  - Accel-Sim backend integration, delivered as a reviewable patch

Upstream trees are submodules and are never edited in place.
166 standalone unit assertions; no large traces or weights are tracked."

echo
echo "Repository assembled."
echo "  files tracked : $(git ls-files | wc -l | tr -d ' ')"
echo "  size          : $(du -sh --exclude=.git --exclude=accel-sim-framework --exclude=ramulator2 . 2>/dev/null | cut -f1 || du -sh . | cut -f1)"
echo
echo "Next:"
echo "  cd $TARGET"
echo "  git remote add origin YOUR_GITHUB_REPO_URL"
echo "  git push -u origin main"
