#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

#=== COLORS ===#
GREEN='\033[0;32m'
NC='\033[0m' # No Color

#=== HELPERS ===#
info() { echo -e "${GREEN}[INFO]${NC} $*"; }
error() { echo -e "[ERROR] $*" >&2; exit 1; }

#=== MAIN SCRIPT ===#
info "Starting TrialMatchAI setup..."

# 0) Check for available GPUs
info "Checking for available GPUs..."

if command -v nvidia-smi &> /dev/null; then
  if nvidia-smi &> /dev/null; then
    info "NVIDIA GPUs detected:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
  else
    info "nvidia-smi found, but no NVIDIA GPU detected or driver not loaded."
  fi
else
  info "No NVIDIA GPUs detected."
fi

# 1) Install Python dependencies
if command -v uv &> /dev/null; then
  info "Installing Python dependencies with uv..."
  uv sync
  RUNNER=(uv run)
else
  if ! command -v pip &> /dev/null; then
    error "pip not found. Please install Python and pip first."
  fi
  info "Installing Python requirements with pip..."
  pip install --upgrade pip
  pip install -r requirements.txt
  pip install -e .
  RUNNER=()
fi

# 2) Prepare data and models
info "Bootstrapping data and models..."
"${RUNNER[@]}" trialmatchai-bootstrap-data

# 3) Build local LanceDB search tables
info "Indexing trial data..."
"${RUNNER[@]}" trialmatchai-index

info "TrialMatchAI setup is complete."
