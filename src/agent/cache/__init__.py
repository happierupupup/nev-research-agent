"""Optional Redis tool-result cache (fail-open) + hot TTL / KB generation."""

from agent.cache.tool_cache import (
    bump_kb_generation,
    cache_get,
    cache_hits,
    cache_set,
    reset_cache_stats,
)

__all__ = [
    "bump_kb_generation",
    "cache_get",
    "cache_hits",
    "cache_set",
    "reset_cache_stats",
]
