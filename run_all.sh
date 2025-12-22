#!/usr/bin/env bash
set -e

if [ ! -d ".venv" ]; then
  echo "❌ .venv not found. Run ./setup.sh first."
  exit 1
fi

source .venv/bin/activate

echo "▶ Running full token identity drift suite..."
python run_suite.py

echo ""
echo "▶ Generating composite figures..."
python make_composites.py

echo ""
echo "✅ All experiments and composites completed."
echo "Results saved to ./results/"
