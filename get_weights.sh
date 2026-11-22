#!/usr/bin/env bash
# Put the two large checkpoints in weights/ and verify them.
# Skips any file that is already present with the right checksum, so it is safe to
# run on the submitted Thapar_INCOGNITO/ folder (which already contains them) and in Docker.
# Tolerates Windows line endings in SHA256SUMS. (refine.json and policy.json are small
# and always in weights/.)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="https://github.com/prabindersinghh/ustad-samsung-xr/releases/download/weights-v1"
cd "$HERE/weights"
mkdir -p clip
SUMS="$(tr -d '\r' < SHA256SUMS)"
while read -r sum file; do
  [ -z "${file:-}" ] && continue
  if [ -f "$file" ] && echo "$sum  $file" | sha256sum -c --status; then
    echo "ok       $file"
  else
    echo "fetching $file"
    curl -fL --retry 3 -o "$file" "$BASE/$(basename "$file")"
  fi
done <<< "$SUMS"
echo "$SUMS" | sha256sum -c
