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
warn() { echo -e "[WARN] $*" >&2; }
error() { echo -e "[ERROR] $*" >&2; exit 1; }

verify_sha256() {
  local file="$1"
  local expected="$2"
  if [ -z "$expected" ]; then
    warn "No SHA-256 checksum configured for $file; skipping verification."
    return 0
  fi
  local actual
  actual="$(sha256sum "$file" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$file" | awk '{print $1}')"
  if [ "$actual" != "$expected" ]; then
    error "Checksum mismatch for $file: expected $expected, got $actual"
  fi
}

assert_safe_archive_paths() {
  local archive="$1"
  shift
  local list_cmd=("$@")
  local unsafe
  unsafe="$("${list_cmd[@]}" "$archive" | awk '($0 ~ /^\// || $0 ~ /(^|\/)\.\.($|\/)/) {print; exit}')"
  if [ -n "$unsafe" ]; then
    error "Archive $archive contains unsafe path: $unsafe"
  fi
}

extract_tar_gz() {
  local archive="$1"
  local target="$2"
  assert_safe_archive_paths "$archive" tar -tzf
  tar -xzf "$archive" -C "$target"
}

extract_zip() {
  local archive="$1"
  local target="$2"
  assert_safe_archive_paths "$archive" unzip -Z1
  unzip -q "$archive" -d "$target"
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"

info "Preparing data directory..."
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ ! -f "$ARCHIVE_1" ]; then
  info "Downloading ${ARCHIVE_1}..."
  curl -fsSL "$DATA_URL_1" -o "$ARCHIVE_1"
else
  info "${ARCHIVE_1} already exists. Skipping download."
fi
verify_sha256 "$ARCHIVE_1" "${TRIALMATCHAI_PROCESSED_TRIALS_SHA256:-}"

if [ ! -f "$RESOURCES_ARCHIVE" ]; then
  info "Downloading ${RESOURCES_ARCHIVE}..."
  curl -fsSL "$RESOURCES_URL" -o "$RESOURCES_ARCHIVE"
else
  info "${RESOURCES_ARCHIVE} already exists. Skipping download."
fi
verify_sha256 "$RESOURCES_ARCHIVE" "${TRIALMATCHAI_RESOURCES_SHA256:-}"

if [ ! -f "$MODELS_ARCHIVE" ]; then
  info "Downloading ${MODELS_ARCHIVE}..."
  curl -fsSL "$MODELS_URL" -o "$MODELS_ARCHIVE"
else
  info "${MODELS_ARCHIVE} already exists. Skipping download."
fi
verify_sha256 "$MODELS_ARCHIVE" "${TRIALMATCHAI_MODELS_SHA256:-}"

if [ ! -d "processed_criteria" ]; then
  info "Downloading and extracting processed_criteria chunks..."
  mkdir -p processed_criteria

  for i in $(seq 0 $((CHUNK_COUNT - 1))); do
    chunk_zip="${CHUNK_PREFIX}_${i}.zip"
    chunk_url="${CRITERIA_ZIP_BASE_URL}/${chunk_zip}?download=1"

    if [ ! -f "$chunk_zip" ]; then
      info "Downloading $chunk_zip..."
      curl -fsSL "$chunk_url" -o "$chunk_zip"
    else
      info "$chunk_zip already exists. Skipping download."
    fi

    checksum_var="TRIALMATCHAI_CRITERIA_PART_${i}_SHA256"
    verify_sha256 "$chunk_zip" "${!checksum_var:-}"
    info "Extracting $chunk_zip into processed_criteria..."
    extract_zip "$chunk_zip" processed_criteria
  done
else
  info "processed_criteria already exists. Skipping extraction."
fi

if [ ! -d "processed_trials" ]; then
  info "Extracting $ARCHIVE_1..."
  extract_tar_gz "$ARCHIVE_1" "$DATA_DIR"
else
  info "processed_trials already exists. Skipping extraction of $ARCHIVE_1."
fi

cd "$ROOT_DIR"

info "Extracting resources into source/Parser..."
mkdir -p source/Parser
extract_tar_gz "$DATA_DIR/$RESOURCES_ARCHIVE" source/Parser

info "Extracting models into models/..."
mkdir -p models
extract_tar_gz "$DATA_DIR/$MODELS_ARCHIVE" models

info "Cleaning up archives..."
rm -f "$DATA_DIR/$ARCHIVE_1" "$DATA_DIR/$RESOURCES_ARCHIVE" "$DATA_DIR/$MODELS_ARCHIVE"
for i in $(seq 0 $((CHUNK_COUNT - 1))); do
  rm -f "$DATA_DIR/${CHUNK_PREFIX}_${i}.zip"
done

info "Data bootstrap complete."
