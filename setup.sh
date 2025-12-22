#!/usr/bin/env bash
set -e

# Load environment variables if .env exists
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# Create virtual environment (Python 3.13 if available, fallback otherwise)
if command -v python3.13 >/dev/null 2>&1; then
  python3.13 -m venv .venv
else
  python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo ""
echo "✅ Environment setup complete."
echo "To activate later, run:"
echo "  source .venv/bin/activate"
