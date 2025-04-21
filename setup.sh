#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

#=== CONFIGURATION ===#
# Zenodo links (replace with your actual Zenodo file URLs)
DATA_URL_1="https://zenodo.org/records/15254844/files/processed_trials.tar.gz?download=1"
DATA_URL_2="https://zenodo.org/records/15254844/files/processed_criteria.tar.gz?download=1"
RESOURCES_URL="https://zenodo.org/records/15254844/files/resources.tar.gz?download=1"
MODELS_URL="https://zenodo.org/records/15254844/files/models.tar.gz?download=1"

# Names to save the downloaded archives as
ARCHIVE_1="processed_trials.tar.gz"
ARCHIVE_2="processed_criteria.tar.gz"
RESOURCES_ARCHIVE="resources.tar.gz"
MODELS_ARCHIVE="models.tar.gz"

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
  info "NVIDIA GPUs detected:"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv
else
  info "No NVIDIA GPUs detected."
fi

# Note: Repository should already be cloned and current directory is its root

# 1) Install Python dependencies
if ! command -v pip &> /dev/null; then
  error "pip not found. Please install Python and pip first."
fi
info "Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# 2) Download and unpack data archives
info "Preparing data directory..."
cd data

# Download dataset parts
info "Downloading ${ARCHIVE_1}..."
wget --quiet "$DATA_URL_1" -O "$ARCHIVE_1"
info "Downloading ${ARCHIVE_2}..."
wget --quiet "$DATA_URL_2" -O "$ARCHIVE_2"

# Download resources
info "Downloading resources archive..."
wget --quiet "$RESOURCES_URL" -O "$RESOURCES_ARCHIVE"

# Download models
info "Downloading models archive..."
wget --quiet "$MODELS_URL" -O "$MODELS_ARCHIVE"

# Unpack datasets
info "Unpacking datasets..."
tar -xzvf "$ARCHIVE_1"
tar -xzvf "$ARCHIVE_2"

# Move resources into Parser
info "Moving resources to src/Parser..."
mkdir -p ../src/Parser
tar -xzvf "$RESOURCES_ARCHIVE" -C ../src/Parser

# Extract models into models/
info "Extracting models into models/..."
mkdir -p ../models
tar -xzvf "$MODELS_ARCHIVE" -C ../models

# Clean up archives
info "Cleaning up archives..."
rm -f "$ARCHIVE_1" "$ARCHIVE_2" "$RESOURCES_ARCHIVE" "$MODELS_ARCHIVE"

# 3) Build Elasticsearch mirror with Docker Compose
cd ..
info "Building Elasticsearch mirror via Docker Compose..."
cd docker
docker-compose up -d --build

# 4) Launch indexers in background and wait
cd ../src/Indexer
info "Starting index_criteria.py (trials_eligibility) ..."
nohup python index_criteria.py \
  --config          config.json \
  --processed-folder ../../data/processed_criteria \
  --index-name      trials_eligibility \
  --batch-size      100 \
  --max-workers     100 \
  > criteria.log 2>&1 &

info "Starting index_trials.py (clinical_trials) ..."
nohup python index_trials.py \
  --config           config.json \
  --processed-folder ../../data/processed_trials \
  --index-name       clinical_trials \
  --batch-size       100 \
  > trials.log 2>&1 &

# Wait for both background jobs to finish
info "Waiting for indexing jobs to complete..."
wait

# 5) Start the existing Elasticsearch container
info "Starting Elasticsearch container (no rebuild)..."
cd ../../docker
# Get the container ID for the elasticsearch service
container_id=$(docker-compose ps -q elasticsearch)
if [ -n "$container_id" ]; then
  info "Found existing container: $container_id. Starting..."
  docker start "$container_id"
else
  info "No existing container found. Launching via docker-compose up..."
  docker-compose up -d elasticsearch
fi

info "✅ TrialMatchAI setup is complete!"
