#!/usr/bin/env python3
"""Compat shim — implementation in evals/runners/run_retrieval.py."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evals.runners.run_retrieval import main  # noqa: E402

if __name__ == "__main__":
    main()
