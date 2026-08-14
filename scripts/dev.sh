#!/usr/bin/env bash
set -e

echo "=== Starting Voice RAG Local Development ==="

# Trap INT and TERM to kill background jobs on exit
trap 'kill $(jobs -p) 2>/dev/null' EXIT

# Start backend
cd "$(dirname "$0")/../backend"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload &

# Start frontend
cd ../frontend
npm run dev -- --host 127.0.0.1 --port 5173 &

wait
