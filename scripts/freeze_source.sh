#!/usr/bin/env bash
# Materialize an immutable source snapshot of HEAD for one experiment.
#
# A multi-node job reads the shared working tree at every launch, so editing the
# checkout while a sweep runs silently changes later configs -- two runs in the
# same sweep then differ by more than the variable under test. Freeze once, then
# point every config in the sweep at the snapshot.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:?Usage: freeze_source.sh <destination>}"

[[ -e "${DEST}" ]] && { echo "Refusing to overwrite an existing snapshot: ${DEST}" >&2; exit 2; }
if ! git -C "${ROOT}" diff --quiet HEAD -- ops distributed models data training configs utils scripts; then
  echo 'Commit the training sources before freezing a snapshot.' >&2
  git -C "${ROOT}" diff --name-only HEAD -- ops distributed models data training configs utils scripts >&2
  exit 3
fi

mkdir -p "${DEST}"
git -C "${ROOT}" archive HEAD | tar -x -C "${DEST}"
# `git archive` ships tracked files only, and .gitignore excludes the suite's
# images while keeping suite.json, so a snapshot used to carry a validation
# config whose inputs were absent. In-training validation then generated its
# T2I cases, died on the first I2T case and took the whole job down with it.
# They are small and the suite is useless without them, so copy them in.
if [[ -d "${ROOT}/validation_assets" ]]; then
  cp -a "${ROOT}/validation_assets/." "${DEST}/validation_assets/"
fi
# Weights, wheels and the venv are large, immutable and shared; link them.
for shared in .venv-cuda .venv-core17 .assets .deps .python; do
  [[ -e "${ROOT}/${shared}" ]] && ln -s "${ROOT}/${shared}" "${DEST}/${shared}"
done
git -C "${ROOT}" rev-parse HEAD >"${DEST}/SOURCE_COMMIT"
chmod -R a-w "${DEST}/ops" "${DEST}/models" "${DEST}/distributed" "${DEST}/training" 2>/dev/null || true
echo "${DEST}"
