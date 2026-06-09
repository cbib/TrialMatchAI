#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

GREEN='\033[0;32m'
NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $*"; }
error() { echo -e "[ERROR] $*" >&2; exit 1; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR/utils/Indexer"
info "Starting index_criteria.py (trials_eligibility) ..."
nohup python index_criteria.py \
  --config           config.json \
  --processed-folder ../../data/processed_criteria \
  --index-name       trials_eligibility \
  --batch-size       100 \
  --max-workers      100 \
  > criteria.log 2>&1 &

info "Starting index_trials.py (clinical_trials) ..."
nohup python index_trials.py \
  --config           config.json \
  --processed-folder ../../data/processed_docs \
  --index-name       clinical_trials \
  --batch-size       100 \
  > trials.log 2>&1 &

info "Waiting for indexing jobs to complete..."
wait
info "✅ Indexing complete."
