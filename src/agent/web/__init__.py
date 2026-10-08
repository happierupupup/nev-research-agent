"""Gradio + SSE web frontend for NEV Research Agent."""

from __future__ import annotations

__all__ = ["create_app"]


def create_app():
    from agent.web.app import create_app as _create

    return _create()
