#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

#=== CONFIGURATION ===#
DATA_URL_1="https://zenodo.org/records/15424310/files/processed_trials.tar.gz?download=1"
DATA_URL_2="https://zenodo.org/records/15440795/files/processed_criteria_flattened.zip?download=1"
RESOURCES_URL="https://zenodo.org/records/15424310/files/resources.tar.gz?download=1"
MODELS_URL="https://zenodo.org/records/15424310/files/models.tar.gz?download=1"

ARCHIVE_1="processed_trials.tar.gz"
ARCHIVE_2="processed_criteria_flattened.zip"
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
if ! command -v pip &> /dev/null; then
  error "pip not found. Please install Python and pip first."
fi
info "Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# 2) Prepare data directory
info "Preparing data directory..."
mkdir -p data
cd data

# Download archives if not already downloaded
if [ ! -f "$ARCHIVE_1" ]; then
  info "Downloading ${ARCHIVE_1}..."
  wget --quiet "$DATA_URL_1" -O "$ARCHIVE_1"
else
  info "${ARCHIVE_1} already exists. Skipping download."
fi

if [ ! -f "$ARCHIVE_2" ]; then
  info "Downloading ${ARCHIVE_2}..."
  wget --quiet "$DATA_URL_2" -O "$ARCHIVE_2"
else
  info "${ARCHIVE_2} already exists. Skipping download."
fi

if [ ! -f "$RESOURCES_ARCHIVE" ]; then
  info "Downloading ${RESOURCES_ARCHIVE}..."
  wget --quiet "$RESOURCES_URL" -O "$RESOURCES_ARCHIVE"
else
  info "${RESOURCES_ARCHIVE} already exists. Skipping download."
fi

if [ ! -f "$MODELS_ARCHIVE" ]; then
  info "Downloading ${MODELS_ARCHIVE}..."
  wget --quiet "$MODELS_URL" -O "$MODELS_ARCHIVE"
else
  info "${MODELS_ARCHIVE} already exists. Skipping download."
fi

# 3) Extract files (tar or unzip)
extract_with_docker() {
  local archive_name="$1"
  local expected_dir="$2"

  if [ ! -d "$expected_dir" ]; then
    info "Extracting $archive_name into $expected_dir using Docker..."
    docker run --rm -v "$PWD:/data" ubuntu bash -c \
      "apt update && apt install -y tar > /dev/null && cd /data && tar -xzvf $archive_name"
  else
    info "$expected_dir already exists. Skipping extraction of $archive_name."
  fi
}

# Extract tar-based archive
extract_with_docker "$ARCHIVE_1" "processed_trials"

# Extract ZIP archive without hitting open file limits
if [ ! -d "processed_criteria" ]; then
  info "Extracting $ARCHIVE_2 into processed_criteria..."
  mkdir -p processed_criteria
  unzip -q "$ARCHIVE_2" -d processed_criteria
else
  info "processed_criteria already exists. Skipping extraction of $ARCHIVE_2."
fi

# Move resources into src/Parser
cd ..
info "Extracting resources into src/Parser..."
mkdir -p src/Parser
tar -xzvf data/"$RESOURCES_ARCHIVE" -C src/Parser

# Extract models into models/
info "Extracting models into models/..."
mkdir -p models
tar -xzvf data/"$MODELS_ARCHIVE" -C models

# Optional cleanup
info "Cleaning up archives..."
rm -f data/"$ARCHIVE_1" data/"$ARCHIVE_2" data/"$RESOURCES_ARCHIVE" data/"$MODELS_ARCHIVE"

# 4) Build Elasticsearch mirror with Docker Compose
cd docker
info "Building Elasticsearch mirror via Docker Compose..."
docker-compose up -d --build

# 5) Launch indexers in background
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

info "Waiting for indexing jobs to complete..."
wait

# 6) Start existing Elasticsearch container
cd ../../docker
container_id=$(docker-compose ps -q elasticsearch)
if [ -n "$container_id" ]; then
  info "Found existing container: $container_id. Starting..."
  docker start "$container_id"
else
  info "No existing container found. Launching via docker-compose up..."
  docker-compose up -d elasticsearch
fi

info "✅ TrialMatchAI setup is complete!"
