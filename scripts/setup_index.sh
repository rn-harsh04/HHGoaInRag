#!/usr/bin/env bash
set -e

echo "=== Setting up MSMARCO-XI Index ==="
cd "$(dirname "$0")/../backend"

# Run indexing on MSMARCO-XI subset (English passages across Indic configs)
python -m indexing.build_index \
  --languages hi,ta,bn \
  --queries-per-lang 500 \
  --embedding-model BAAI/bge-small-en-v1.5

echo "=== Index Build Completed Successfully ==="
