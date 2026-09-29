"""Quality-gated tool retry: try the next source only if it can still help.

Not a fixed “retry 3 times”. One agent-facing tool call may internally
walk providers / one reformulation, then degrade to KB.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse
import re

GOOD_QUALITY = 0.55
MIN_GAIN = 0.08
# Safety ceiling only — the loop usually stops earlier via quality / unused providers.
MAX_ATTEMPTS = 5
CIRCUIT_OPEN_AFTER = 2
# Hard cap on network/embedding calls (connection retries share this).
MAX_NETWORK_TRIES = 3
MAX_PARAM_ROUNDS = 3

_JUNK_HINTS = ("job.", "jobs.", "careers", "zhaopin", "招聘", "campus")
_THIN_HOSTS = ("baike.baidu.com", "baike.com")
_OFFTOPIC_HOSTS = (
    "douban.com",
    "movie.douban.com",
    "iqiyi.com",
    "v.qq.com",
    "youku.com",
)
_THIN_PATHS = {"", "/", "/cn", "/cn/", "/cn/index.html", "/index.html", "/en", "/en/"}
_YEAR_TITLE = re.compile(r"^20\d{2}年(_|（|$)")

_fail_streak: dict[str, int] = {}
_open_circuits: set[str] = set()


@dataclass(frozen=True)
class Strategy:
    name: str
    provider: str
    query: str
    run: Callable[[], list[dict[str, Any]]]


@dataclass
class Attempt:
    name: str
    provider: str
    query: str
    hits: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    quality: float = 0.0


def reset_circuits() -> None:
    _fail_streak.clear()
    _open_circuits.clear()


def provider_blocked(provider: str) -> bool:
    return provider in _open_circuits


def record_provider(provider: str, *, ok: bool) -> None:
    if ok:
        _fail_streak[provider] = 0
        _open_circuits.discard(provider)
        return
    _fail_streak[provider] = _fail_streak.get(provider, 0) + 1
    if _fail_streak[provider] >= CIRCUIT_OPEN_AFTER:
        _open_circuits.add(provider)


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:  # noqa: BLE001
        return ""


def _path(url: str) -> str:
    try:
        return (urlparse(url).path or "").lower().rstrip("/") or "/"
    except Exception:  # noqa: BLE001
        return ""


def _hit_blob(hit: dict[str, Any]) -> str:
    return f"{hit.get('title') or ''} {hit.get('snippet') or ''} {hit.get('url') or ''}"


def filter_web_hits(
    hits: list[dict[str, Any]] | None, query: str
) -> list[dict[str, Any]]:
    """Drop calendar-year / movie / encyclopedia rows when the query is NEV."""
    from agent.core.intent import is_nev_query

    rows = list(hits or [])
    if not rows or not is_nev_query(query):
        return rows
    kept: list[dict[str, Any]] = []
    for h in rows:
        host = _host(str(h.get("url") or ""))
        title = str(h.get("title") or "")
        if any(off in host for off in _OFFTOPIC_HOSTS):
            continue
        if _YEAR_TITLE.match(title.strip()):
            continue
        if "baike.baidu.com" in host and re.search(r"/item/20\d{2}", str(h.get("url") or "")):
            continue
        if not is_nev_query(_hit_blob(h)):
            continue
        kept.append(h)
    return kept


def score_web_hits(hits: list[dict[str, Any]] | None) -> float:
    """0–1: more unique hosts + snippets, fewer job/baike/homepage-only rows."""
    rows = [h for h in (hits or []) if (h.get("url") or "").startswith("http")]
    if not rows:
        return 0.0
    n = len(rows)
    hosts = [_host(str(h.get("url") or "")) for h in rows]
    unique = len({h for h in hosts if h}) / n
    avg_snip = sum(len(str(h.get("snippet") or "")) for h in rows) / n
    junk = thin = 0
    for h, host in zip(rows, hosts, strict=True):
        blob = f"{host} {h.get('title') or ''} {h.get('url') or ''}".lower()
        if any(j in blob for j in _JUNK_HINTS):
            junk += 1
        path = _path(str(h.get("url") or ""))
        if any(t in host for t in _THIN_HOSTS) or path in _THIN_PATHS:
            thin += 1
    useful = (n - junk) / n
    score = 0.3 * min(n / 5.0, 1.0)
    score += 0.25 * unique
    score += 0.25 * min(avg_snip / 80.0, 1.0)
    score += 0.2 * useful
    score -= 0.35 * (thin / n)
    return round(max(0.0, min(1.0, score)), 4)


def decide_continue(history: list[Attempt], nxt: Strategy | None) -> bool:
    """Whether the next strategy is worth running given quality so far."""
    if nxt is None:
        return False
    if len(history) >= MAX_ATTEMPTS:
        return False
    if provider_blocked(nxt.provider):
        return False
    if not history:
        return True
    best = max(a.quality for a in history)
    if best >= GOOD_QUALITY:
        return False
    tried = {a.provider for a in history}
    if nxt.provider not in tried:
        return True
    same = [a for a in history if a.provider == nxt.provider]
    if len(same) >= 2:
        return False
    last = history[-1]
    if last.error:
        return False
    if last.quality <= 0 and nxt.query == last.query:
        return False
    if len(same) == 1 and last.provider == nxt.provider:
        return last.quality < GOOD_QUALITY
    return last.quality + MIN_GAIN < GOOD_QUALITY


def execute_strategies(strategies: list[Strategy]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run strategies until quality is good enough or the next step cannot help."""
    history: list[Attempt] = []
    for nxt in strategies:
        if history and max(a.quality for a in history) >= GOOD_QUALITY:
            break
        if not decide_continue(history, nxt):
            continue
        hits: list[dict[str, Any]] = []
        err: str | None = None
        try:
            hits = filter_web_hits(list(nxt.run() or []), nxt.query)
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            hits = []
        quality = score_web_hits(hits)
        record_provider(nxt.provider, ok=bool(hits) and err is None)
        history.append(
            Attempt(
                name=nxt.name,
                provider=nxt.provider,
                query=nxt.query,
                hits=hits,
                error=err,
                quality=quality,
            )
        )
        if quality >= GOOD_QUALITY:
            break

    best = max(history, key=lambda a: a.quality) if history else None
    hits = list(best.hits) if best and best.hits else []
    quality = best.quality if best else 0.0
    return hits, {
        "quality": quality,
        "degraded": quality < GOOD_QUALITY,
        "fallback": "kb" if not hits else None,
        "attempts": [
            {
                "name": a.name,
                "provider": a.provider,
                "quality": a.quality,
                "n": len(a.hits),
                "error": a.error,
            }
            for a in history
        ],
    }


def is_transient_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    needles = (
        "timeout",
        "timed out",
        "connection",
        "reset",
        "temporarily",
        "429",
        "503",
        "502",
        "unavailable",
        "connecterror",
    )
    if any(n in name for n in ("timeout", "connection", "unavailable")):
        return True
    return any(n in msg for n in needles)


def call_with_transient_retry(fn: Callable[[], Any], *, max_tries: int = MAX_NETWORK_TRIES) -> tuple[Any, str | None, int]:
    """Retry only when the error looks transient. Permanent errors stop immediately."""
    tries = max(1, int(max_tries))
    last_err: str | None = None
    for i in range(tries):
        try:
            return fn(), None, i + 1
        except FileNotFoundError as exc:
            return None, str(exc), i + 1
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            if not is_transient_error(exc) or i + 1 >= tries:
                return None, last_err, i + 1
    return None, last_err, tries


def alt_web_query(query: str) -> str | None:
    """Drop the year if Bing latched onto it; otherwise add a year cue."""
    q = (query or "").strip()
    if not q:
        return None
    stripped = re.sub(r"20\d{2}\s*年?", " ", q)
    stripped = re.sub(r"\s+", " ", stripped).strip(" ，,")
    if stripped and stripped != q:
        return stripped
    low = q.lower()
    if any(x in low for x in ("新闻", "news", "最新")):
        return None
    return f"{q} 2024"


_KB_FLUFF = re.compile(
    r"(的核心卖点是什么|大概多少|是什么|如何|怎样|请分析|请|一下|[？?？])"
)


def kb_query_variants(query: str) -> list[str]:
    """Different query strings so a retry is not the same embedding."""
    q = (query or "").strip()
    out: list[str] = []
    for cand in (q, _KB_FLUFF.sub(" ", q).strip(), q[:12].strip() if len(q) > 12 else ""):
        c = re.sub(r"\s+", " ", cand).strip()
        if c and c not in out:
            out.append(c)
    return out[:MAX_PARAM_ROUNDS] or [q]


def kb_param_schedule(query: str, top_k: int) -> list[tuple[str, int]]:
    k0 = max(1, min(int(top_k), 8))
    ks = [k0, min(8, k0 + 2), min(8, max(k0 + 4, 6))]
    qs = kb_query_variants(query)
    rows: list[tuple[str, int]] = []
    for i, q in enumerate(qs):
        rows.append((q, ks[min(i, len(ks) - 1)]))
    while len(rows) < min(MAX_PARAM_ROUNDS, 3):
        q, k = rows[-1]
        nk = min(8, k + 2)
        if nk == k:
            break
        rows.append((q, nk))
    return rows[:MAX_PARAM_ROUNDS]


def _hit_id(h: Any) -> str:
    ch = getattr(h, "chunk", None)
    if ch is not None:
        return str(getattr(ch, "id", "") or "")
    if isinstance(h, dict):
        return str(h.get("id") or "")
    return ""


def _hit_source(h: Any) -> str:
    ch = getattr(h, "chunk", None)
    if ch is not None:
        return str(getattr(ch, "source", "") or "")
    if isinstance(h, dict):
        return str(h.get("source") or "")
    return ""


def _hit_score(h: Any) -> float:
    if isinstance(h, dict):
        return float(h.get("score") or 0)
    val = getattr(h, "score", None)
    try:
        return float(val or 0)
    except (TypeError, ValueError):
        return 0.0


def merge_kb_hits(acc: list[Any], new: list[Any]) -> list[Any]:
    by_id: dict[str, Any] = {}
    for h in acc + list(new or []):
        hid = _hit_id(h)
        if not hid:
            continue
        old = by_id.get(hid)
        if old is None or _hit_score(h) > _hit_score(old):
            by_id[hid] = h
    return sorted(by_id.values(), key=_hit_score, reverse=True)


def run_kb_search(
    query: str,
    top_k: int,
    search_fn: Callable[[str, int, set[str], set[str]], list[Any]],
    *,
    network_budget: int = MAX_NETWORK_TRIES,
    param_rounds: int = MAX_PARAM_ROUNDS,
) -> tuple[list[Any], dict[str, Any]]:
    """Vary query/top_k/excludes on retry; cap embedding/network calls."""
    budget = max(1, int(network_budget))
    schedule = kb_param_schedule(query, top_k)
    exclude_ids: set[str] = set()
    exclude_sources: set[str] = set()
    merged: list[Any] = []
    attempts: list[dict[str, Any]] = []
    calls = 0
    last_err: str | None = None

    for q, k in schedule[: max(1, int(param_rounds))]:
        if calls >= budget:
            attempts.append({"query": q, "top_k": k, "stop": "network_budget"})
            break
        batch: list[Any] | None = None
        while calls < budget:
            calls += 1
            try:
                batch = list(search_fn(q, k, set(exclude_ids), set(exclude_sources)) or [])
                last_err = None
                break
            except FileNotFoundError as exc:
                return [], {
                    "error": str(exc),
                    "tries": calls,
                    "degraded": True,
                    "fallback": "stop",
                    "attempts": attempts,
                }
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
                if not is_transient_error(exc):
                    return merged, {
                        "error": last_err,
                        "tries": calls,
                        "degraded": True,
                        "attempts": attempts,
                    }
                continue
        if batch is None:
            attempts.append(
                {"query": q, "top_k": k, "error": last_err, "stop": "network_budget"}
            )
            break

        ids = {_hit_id(h) for h in batch if _hit_id(h)}
        srcs = {_hit_source(h) for h in batch if _hit_source(h)}
        already = {_hit_id(h) for h in merged}
        new_ids = ids - already
        attempts.append(
            {
                "query": q,
                "top_k": k,
                "n": len(batch),
                "sources": sorted(srcs),
                "new_chunks": len(new_ids),
            }
        )
        merged = merge_kb_hits(merged, batch)
        unique_src = {_hit_source(h) for h in merged if _hit_source(h)}
        if len(unique_src) >= 2:
            break
        if not ids:
            break
        exclude_ids |= ids
        if len(unique_src) == 1:
            exclude_sources |= unique_src
        if not new_ids:
            continue

    unique_src = {_hit_source(h) for h in merged if _hit_source(h)}
    k_out = max(1, min(int(top_k), 8))
    meta: dict[str, Any] = {
        "tries": calls,
        "degraded": len(unique_src) < 2,
        "attempts": attempts,
    }
    if last_err and not merged:
        meta["error"] = last_err
        meta["fallback"] = "stop"
    return merged[:k_out], meta
