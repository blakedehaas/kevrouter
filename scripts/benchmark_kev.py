#!/usr/bin/env python3
"""Convenience runner to benchmark pre-trained Kev baseline router."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.benchmark_kev import main

if __name__ == "__main__":
    main()
