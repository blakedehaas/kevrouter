#!/usr/bin/env python3
"""Runner script to optimize Kev loss functions and decision rules on the validation set."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.optimize_loss import main

if __name__ == "__main__":
    main()
