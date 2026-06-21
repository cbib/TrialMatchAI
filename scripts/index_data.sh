#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

GREEN='\033[0;32m'
NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $*"; }
error() { echo -e "[ERROR] $*" >&2; exit 1; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"
info "Building LanceDB search tables ..."
uv run trialmatchai-index \
  --processed-trials-folder data/processed_trials \
  --processed-criteria-folder data/processed_criteria \
  --recreate
info "Indexing complete."
