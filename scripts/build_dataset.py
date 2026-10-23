#!/usr/bin/env python3
"""Convenience script to build aligned, deduplicated, and split datasets."""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_dataset import main

if __name__ == "__main__":
    main()
