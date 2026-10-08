#!/usr/bin/env python3
"""Gradio + SSE web entrypoint.

    python web_app.py
    # 浏览器打开 http://127.0.0.1:7860/ui/
    # SSE: POST /api/sessions/{id}/stream  (Authorization: Bearer <token>)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.web.app import main  # noqa: E402

if __name__ == "__main__":
    main()
