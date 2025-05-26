#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

#=== CONFIGURATION ===#
DATA_URL_1="https://zenodo.org/records/15424310/files/processed_trials.tar.gz?download=1"
RESOURCES_URL="https://zenodo.org/records/15424310/files/resources.tar.gz?download=1"
MODELS_URL="https://zenodo.org/records/15424310/files/models.tar.gz?download=1"
CRITERIA_ZIP_BASE_URL="https://zenodo.org/records/15424310/files"
CHUNK_PREFIX="criteria_part"
CHUNK_COUNT=6

ARCHIVE_1="processed_trials.tar.gz"
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

# Download core archives
if [ ! -f "$ARCHIVE_1" ]; then
  info "Downloading ${ARCHIVE_1}..."
  wget --quiet "$DATA_URL_1" -O "$ARCHIVE_1"
else
  info "${ARCHIVE_1} already exists. Skipping download."
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

# Download and extract processed_criteria ZIP chunks
if [ ! -d "processed_criteria" ]; then
  info "Downloading and extracting processed_criteria chunks..."
  mkdir -p processed_criteria

  for i in $(seq 0 $((CHUNK_COUNT - 1))); do
    chunk_zip="${CHUNK_PREFIX}_${i}.zip"
    chunk_url="${CRITERIA_ZIP_BASE_URL}/${chunk_zip}?download=1"

    if [ ! -f "$chunk_zip" ]; then
      info "Downloading $chunk_zip..."
      wget --quiet "$chunk_url" -O "$chunk_zip"
    else
      info "$chunk_zip already exists. Skipping download."
    fi

    info "Extracting $chunk_zip into processed_criteria..."
    unzip -q "$chunk_zip" -d processed_criteria
  done
else
  info "processed_criteria already exists. Skipping extraction."
fi

# Extract processed_trials
if [ ! -d "processed_trials" ]; then
  info "Extracting $ARCHIVE_1..."
  tar -xzvf "$ARCHIVE_1"
else
  info "processed_trials already exists. Skipping extraction of $ARCHIVE_1."
fi

cd ..

# Extract resources
info "Extracting resources into src/Parser..."
mkdir -p src/Parser
tar -xzvf data/"$RESOURCES_ARCHIVE" -C src/Parser

info "Extracting models into models/..."
mkdir -p models
tar -xzvf data/"$MODELS_ARCHIVE" -C models

info "Cleaning up archives..."
rm -f data/"$ARCHIVE_1" data/"$RESOURCES_ARCHIVE" data/"$MODELS_ARCHIVE"

for i in $(seq 0 $((CHUNK_COUNT - 1))); do
  rm -f data/"${CHUNK_PREFIX}_${i}.zip"
done

# 3) Launch Elasticsearch: Try Docker first, then Apptainer fallback
if command -v docker &> /dev/null && docker info &> /dev/null; then
  info "Docker is available. Setting up Elasticsearch with Docker Compose..."
  cd docker
  docker-compose up -d --build
  cd ..
elif command -v apptainer &> /dev/null; then
  info "Docker not found or not running. Falling back to Apptainer..."
  if [ ! -f "./docker/apptainer-run-es.sh" ]; then
    error "Apptainer script not found at ./docker/apptainer-run-es.sh"
  fi
  bash ./docker/apptainer-run-es.sh
else
  error "Neither Docker nor Apptainer is available. Cannot continue."
fi

# 4) Launch indexers in background
cd src/Indexer
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
  --processed-folder ../../data/processed_trials \
  --index-name       clinical_trials \
  --batch-size       100 \
  > trials.log 2>&1 &

info "Waiting for indexing jobs to complete..."
wait

info "✅ TrialMatchAI setup is complete!"
