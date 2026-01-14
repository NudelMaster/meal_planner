#!/bin/bash
# Run CLI interface

set -e

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the CLI
python3 -m src.inference.cli
