"""LLM / embedding factories — OpenAI-compatible (incl. DashScope)."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from agent.paths import ROOT

load_dotenv(ROOT / ".env")


def _api_key() -> str:
    key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY (or DASHSCOPE_API_KEY). Copy .env.example → .env"
        )
    return key


def _base_url() -> str | None:
    raw = (os.getenv("OPENAI_BASE_URL") or "").strip().strip("'\"")
    return raw.rstrip("/") if raw else None


def get_llm(temperature: float = 0) -> ChatOpenAI:
    kwargs: dict = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": temperature,
        "api_key": _api_key(),
        "stream_usage": True,
    }
    base = _base_url()
    if base:
        kwargs["base_url"] = base
    if base and "dashscope" in base:
        enable_thinking = os.getenv("ENABLE_THINKING", "").lower() in {"1", "true", "yes"}
        kwargs["extra_body"] = {"enable_thinking": enable_thinking}
    return ChatOpenAI(**kwargs)


def get_embeddings() -> OpenAIEmbeddings:
    kwargs: dict = {
        "model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "api_key": _api_key(),
    }
    base = _base_url()
    if base:
        kwargs["base_url"] = base
    # DashScope: langchain 默认会用 tiktoken 把文本切成 token id 再请求；
    # 百炼兼容接口只接受 str / list[str]，否则 400 InvalidParameter (input.contents)。
    if base and "dashscope" in base.lower():
        kwargs["check_embedding_ctx_length"] = False
        openai_names = {
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        }
        if kwargs["model"] in openai_names:
            kwargs["model"] = "text-embedding-v3"
    return OpenAIEmbeddings(**kwargs)
