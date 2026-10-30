#!/usr/bin/env python3
"""Convenience runner to evaluate all Kev model sizes (0.8B, 4B, 9B) across all splits."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.benchmark_matrix import main

if __name__ == "__main__":
    main()
