#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

GREEN='\033[0;32m'
NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $*"; }
error() { echo -e "[ERROR] $*" >&2; exit 1; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/elasticsearch"

if command -v docker &> /dev/null && docker info &> /dev/null; then
  info "Docker is available. Starting Elasticsearch with Docker Compose..."
  if docker compose version &> /dev/null; then
    docker compose up -d
  elif command -v docker-compose &> /dev/null; then
    docker-compose up -d
  else
    error "Docker is available, but Docker Compose is not installed."
  fi
elif command -v apptainer &> /dev/null; then
  info "Docker not found or not running. Falling back to Apptainer..."
  if [ ! -f "./apptainer-run-es.sh" ]; then
    error "Apptainer script not found at ./elasticsearch/apptainer-run-es.sh"
  fi
  bash ./apptainer-run-es.sh
else
  error "Neither Docker nor Apptainer is available. Cannot continue."
fi
