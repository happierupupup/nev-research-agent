"""Agent tools: knowledge base + web search."""

from __future__ import annotations

from agent.tools.kb_tools import kb_search
from agent.tools.web_tools import web_search

ALL_TOOLS = [kb_search, web_search]
