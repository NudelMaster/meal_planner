"""DEPRECATED: Use inference/cli.py instead.

This file is kept for backward compatibility.
Please run from project root: 
  python -m src.inference.cli
  OR
  ./scripts/run_cli.sh
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("⚠️  WARNING: This file is deprecated!")
print("Please use: python -m src.inference.cli")
print("Or from project root: ./scripts/run_cli.sh\n")
print("Redirecting to new CLI...\n")

from inference.cli import main

if __name__ == "__main__":
    main()
