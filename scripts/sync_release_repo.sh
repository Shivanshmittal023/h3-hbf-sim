#!/usr/bin/env bash
# =============================================================================
#  sync_release_repo.sh -- push development-tree changes into the release repo
# =============================================================================
#  prepare_release_repo.sh creates the publishable repository once. This script
#  updates an EXISTING one in place, preserving its git history, remote and
#  submodule pointers -- so ongoing work can be published without recreating
#  the repo (and losing its origin) each time.
#
#  Copies only H3 project directories. Submodules, .git and generated artefacts
#  are never touched.
#
#  USAGE
#      ./scripts/sync_release_repo.sh [release_dir]     # default ../h3-sim-release
#      ./scripts/sync_release_repo.sh ../h3-sim-release --commit "message"
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-$ROOT/../h3-sim-release}"
shift || true

COMMIT_MSG=""
if [ "${1:-}" = "--commit" ]; then COMMIT_MSG="${2:-Update H3 components}"; fi

[ -d "$TARGET/.git" ] || {
  echo "ERROR: $TARGET is not a git repository." >&2
  echo "       Create it first with ./scripts/prepare_release_repo.sh" >&2
  exit 1
}

echo "Syncing $ROOT  ->  $TARGET"

for d in h3-components configs scripts tools tests docs patches; do
  rsync -a --delete \
        --exclude '__pycache__' --exclude '.DS_Store' \
        "$ROOT/$d/" "$TARGET/$d/"
done
for f in Dockerfile.h3sim docker-compose.yml .dockerignore .gitignore README.md requirements.txt LICENSE; do
  [ -f "$ROOT/$f" ] && cp "$ROOT/$f" "$TARGET/$f"
done

# Generated Ramulator configs and upstream docs never ship.
rm -rf "$TARGET/configs/generated" "$TARGET/docs/img" "$TARGET/docs/accel-sim-README.md"

cd "$TARGET"
echo
git status --short | sed 's/^/  /'
echo

if [ -n "$COMMIT_MSG" ]; then
  git add -A
  git commit -q -m "$COMMIT_MSG" && echo "Committed: $COMMIT_MSG"
  echo "Now run:  cd $TARGET && git push"
else
  echo "Review the changes above, then:"
  echo "  cd $TARGET && git add -A && git commit -m '...' && git push"
fi
