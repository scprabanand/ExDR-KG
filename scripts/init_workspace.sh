#!/usr/bin/env bash
# Create the input/output folder tree of the rerun in the repository root (safe to run repeatedly; never deletes anything).
#   cd <repository root> && bash rerun/scripts/init_workspace.sh
set -euo pipefail
ROOT="${ROOT:-$(pwd)}"; LIST="$ROOT/rerun/configs/workspace_dirs.txt"
[ -f "$LIST" ] || { echo "run from the repository root (rerun/configs/workspace_dirs.txt not found)"; exit 1; }
grep -v '^\s*#' "$LIST" | grep -v '^\s*$' | while read -r d; do mkdir -p "$ROOT/$d"; [ -e "$ROOT/$d/.gitkeep" ] || [ -n "$(ls -A "$ROOT/$d" 2>/dev/null)" ] || touch "$ROOT/$d/.gitkeep"; done
echo "workspace folders ready under $ROOT"
