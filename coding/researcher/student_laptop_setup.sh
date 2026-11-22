#!/usr/bin/env bash
# Run ONCE on a student's laptop, inside their cloned Hakan folder
# (Terminal on Mac, Git Bash on Windows):
#     cd Hakan && bash student_laptop_setup.sh
#
# - checks that Git LFS works (otherwise audio arrives as tiny text files)
# - sets the student's name/e-mail for the history
# - makes the laptop forget old recordings: after every branch switch,
#   `git lfs prune` deletes downloaded audio that the current interview doesn't use.
set -euo pipefail

if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "❌ Run this inside the Hakan folder (cd Hakan)"; exit 1
fi
if ! git lfs version >/dev/null 2>&1; then
    echo "❌ Git LFS is not installed – install it first (Mac: brew install git-lfs)"; exit 1
fi
git lfs install --local >/dev/null

read -r -p "Student's full name: " NAME
read -r -p "Student's RUC e-mail: " EMAIL
git config user.name "$NAME"
git config user.email "$EMAIL"

# Only keep LFS audio for the interview that is checked out right now.
git config lfs.fetchrecentrefsdays 0
git config lfs.fetchrecentcommitsdays 0
git config lfs.fetchrecentremoterefs false
git config lfs.pruneoffsetdays 0

HOOK="$(git rev-parse --git-path hooks)/post-checkout"
if ! grep -q "git lfs prune" "$HOOK" 2>/dev/null; then
    [ -f "$HOOK" ] || printf '#!/bin/sh\ngit lfs post-checkout "$@"\n' > "$HOOK"
    printf '\n# Remove old interview recordings from this laptop\ngit lfs prune --quiet || true\n' >> "$HOOK"
    chmod +x "$HOOK"
fi

git lfs pull
echo "✅ Done. Recordings on this laptop:"
ls compact_audio 2>/dev/null || echo "  (none)"
