#!/usr/bin/env python3
"""Convenience script to run LLMRouterBench data download and extraction."""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download_and_extract import main

if __name__ == "__main__":
    main()
