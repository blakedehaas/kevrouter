#!/usr/bin/env python3
"""Convenience runner to convert LLMRouterBench data into official Kev fine-tuning format."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.convert_to_kev import main

if __name__ == "__main__":
    main()
