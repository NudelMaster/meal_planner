"""Quick test to verify API imports resolve correctly."""

import sys
from pathlib import Path

# Simulate the same path setup as api.py
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from agent.core import create_agent, CulinaryAgent
    from agent.prompts import SYSTEM_PROMPT
    print("✅ All imports resolved successfully!")
    print(f"✅ CulinaryAgent class: {CulinaryAgent}")
    print(f"✅ create_agent function: {create_agent}")
    print(f"✅ SYSTEM_PROMPT loaded: {len(SYSTEM_PROMPT)} characters")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
