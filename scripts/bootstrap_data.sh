#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

DATA_URL_1="https://zenodo.org/records/15516900/files/processed_trials.tar.gz?download=1"
RESOURCES_URL="https://zenodo.org/records/15516900/files/resources.tar.gz?download=1"
MODELS_URL="https://zenodo.org/records/15516900/files/models.tar.gz?download=1"
CRITERIA_ZIP_BASE_URL="https://zenodo.org/records/15516900/files"
CHUNK_PREFIX="criteria_part"
CHUNK_COUNT=6

ARCHIVE_1="processed_trials.tar.gz"
RESOURCES_ARCHIVE="resources.tar.gz"
MODELS_ARCHIVE="models.tar.gz"

GREEN='\033[0;32m'
NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $*"; }
error() { echo -e "[ERROR] $*" >&2; exit 1; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"

info "Preparing data directory..."
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ ! -f "$ARCHIVE_1" ]; then
  info "Downloading ARCHIVE_1: ${ARCHIVE_1}..."
  wget -O "$ARCHIVE_1" "$DATA_URL_1"
else
  info "${ARCHIVE_1} already exists. Skipping download."
fi

if [ ! -f "$RESOURCES_ARCHIVE" ]; then
  info "Downloading RESOURCES_ARCHIVE: ${RESOURCES_ARCHIVE}..."
  wget -O "$RESOURCES_ARCHIVE" "$RESOURCES_URL"
else
  info "${RESOURCES_ARCHIVE} already exists. Skipping download."
fi

if [ ! -f "$MODELS_ARCHIVE" ]; then
  info "Downloading MODELS_ARCHIVE: ${MODELS_ARCHIVE}..."
  wget -O "$MODELS_ARCHIVE" "$MODELS_URL"
else
  info "${MODELS_ARCHIVE} already exists. Skipping download."
fi

if [ ! -d "processed_criteria" ]; then
  info "Downloading and extracting processed_criteria chunks..."
  mkdir -p processed_criteria

  for i in $(seq 0 $((CHUNK_COUNT - 1))); do
    chunk_zip="${CHUNK_PREFIX}_${i}.zip"
    chunk_url="${CRITERIA_ZIP_BASE_URL}/${chunk_zip}?download=1"

    if [ ! -f "$chunk_zip" ]; then
      info "Downloading $chunk_zip..."
      wget -O "$chunk_zip" "$chunk_url"
    else
      info "$chunk_zip already exists. Skipping download."
    fi

    info "Extracting $chunk_zip into processed_criteria..."
    unzip -q "$chunk_zip" -d processed_criteria
  done
else
  info "processed_criteria already exists. Skipping extraction."
fi

if [ ! -d "processed_trials" ]; then
  info "Extracting $ARCHIVE_1..."
  tar -xzvf "$ARCHIVE_1"
else
  info "processed_trials already exists. Skipping extraction of $ARCHIVE_1."
fi

cd "$ROOT_DIR"

info "Extracting resources into source/Parser..."
mkdir -p source/Parser
tar -xzvf "$DATA_DIR/$RESOURCES_ARCHIVE" -C source/Parser

info "Extracting models into models/..."
mkdir -p models
tar -xzvf "$DATA_DIR/$MODELS_ARCHIVE" -C models

info "Cleaning up archives..."
rm -f "$DATA_DIR/$ARCHIVE_1" "$DATA_DIR/$RESOURCES_ARCHIVE" "$DATA_DIR/$MODELS_ARCHIVE"
for i in $(seq 0 $((CHUNK_COUNT - 1))); do
  rm -f "$DATA_DIR/${CHUNK_PREFIX}_${i}.zip"
done

info "✅ Data bootstrap complete."
