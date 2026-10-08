"""Assemble FastAPI + Gradio (SSE API under /api, UI at /)."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from agent.paths import ROOT
from agent.web.api import mount_api
from agent.web.ui import build_ui


def create_app() -> FastAPI:
    load_dotenv(ROOT / ".env")
    api = FastAPI(title="NEV Research Agent", version="0.1.0")
    mount_api(api)

    @api.get("/")
    def _root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/ui/")

    demo = build_ui()
    # Gradio 3：queue()；Gradio 4+ 可带 concurrency 参数
    try:
        demo.queue(default_concurrency_limit=4)
    except TypeError:
        demo.queue()

    import gradio as gr

    # UI 挂在 /ui，避免盖住 /api/* SSE 路由
    app = gr.mount_gradio_app(api, demo, path="/ui")
    return app


def main() -> None:
    import uvicorn

    load_dotenv(ROOT / ".env")
    host = os.getenv("NEV_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("NEV_WEB_PORT", "7860"))
    uvicorn.run(
        "agent.web.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=os.getenv("NEV_WEB_RELOAD", "").lower() in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    main()
