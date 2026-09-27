"""S1.5 session (STM) + per-user long-term profile (LTM).

Memory is never evidence. Numbers and citations must come from this turn's tools.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.intent import is_report_query
from agent.paths import MEMORY_DIR

WINDOW_SIZE = 3
INJECT_MAX_CHARS = 800
SUMMARY_MAX_CHARS = 400
DIGEST_MAX_CHARS = 240
EPISODE_CAP = 50
INTEREST_TOP_K = 5
EPISODE_RETRIEVE_K = 2
INTEREST_DECAY = 0.9

FOLLOWUP_UNCLEAR = (
    "无法确定「它」指什么。请写出具体对象，例如：比亚迪刀片电池用在哪些车上？"
)
PREFERENCE_ACK = "已记下你的偏好，之后会按这个风格回答。事实仍会重新检索，不会把记忆当来源。"

_DEIXIS = re.compile(r"(它|它们|这个|那个|上述|刚才|此)")
_DIGITS = re.compile(r"\d+(?:\.\d+)?%?")
_CITE = re.compile(r"（来源:[^）]*）|\(来源:[^)]*\)")
_UNSAFE_USER = re.compile(r"[^a-zA-Z0-9._-]+")

# Extra topic words for interest / entity extraction (not all are NEV domain-gate markers).
_INTEREST_TERMS = (
    "新能源",
    "电动车",
    "电动汽车",
    "混动",
    "插电",
    "增程",
    "刀片电池",
    "刀片",
    "动力电池",
    "电池",
    "电芯",
    "磷酸铁锂",
    "三元锂",
    "充电桩",
    "换电",
    "超充",
    "乘用车",
    "商用车",
    "智驾",
    "自动驾驶",
    "反补贴",
    "比亚迪",
    "特斯拉",
    "宁德",
    "理想汽车",
    "蔚来",
    "小鹏",
    "问界",
    "极氪",
    "海外",
    "扩张",
    "渗透率",
    "供应链",
    "关税",
    "出口",
    "匈牙利",
    "乘联会",
    "BYD",
    "Tesla",
    "CATL",
    "NEV",
)

_PREF_SHORT = ("短答", "简短", "简要", "以后短答", "写短一点")
_PREF_LONG = ("详细", "长文", "写长一点", "以后详细")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def looks_like_followup(text: str) -> bool:
    return bool(_DEIXIS.search(text or ""))


def extract_preferences_from_text(text: str) -> list[dict[str, str]]:
    t = text or ""
    prefs: list[dict[str, str]] = []
    if any(x in t for x in _PREF_SHORT):
        prefs.append({"key": "length", "value": "short", "source": "explicit"})
    if any(x in t for x in _PREF_LONG):
        prefs.append({"key": "length", "value": "long", "source": "explicit"})
    if "乘联会" in t:
        prefs.append({"key": "source_pref", "value": "cpca", "source": "inferred"})
    return prefs


def extract_entities(*texts: str) -> list[str]:
    blob = " ".join(t for t in texts if t)
    found: list[str] = []
    for term in sorted(_INTEREST_TERMS, key=len, reverse=True):
        if term.lower() in blob.lower() and term not in found:
            found.append(term)
    return found[:12]


def extract_session_interests(session: SessionMemory) -> list[tuple[str, int]]:
    blob = " ".join(session.queries)
    for turn in session.turns:
        blob += " " + " ".join(turn.entities)
        blob += " " + turn.query
    counts: dict[str, int] = {}
    for term in extract_entities(blob):
        counts[term] = counts.get(term, 0) + max(1, blob.lower().count(term.lower()))
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:8]


def strip_numbers(text: str) -> str:
    t = _CITE.sub("", text or "")
    t = _DIGITS.sub("", t)
    t = re.sub(r"\s+", " ", t).strip(" ；;，,")
    return t


def make_digest(report: str, query: str) -> str:
    text = report or ""
    if "## 回答" in text:
        text = text.split("## 回答", 1)[1]
    if "## 信息来源" in text:
        text = text.split("## 信息来源", 1)[0]
    text = strip_numbers(text)
    lines = [ln.strip(" #*-") for ln in text.splitlines() if ln.strip()]
    digest = "；".join(lines[:4]) or strip_numbers(query)
    return digest[:DIGEST_MAX_CHARS]


def refs_from_sources(sources: list[dict[str, Any]] | None) -> list[str]:
    out: list[str] = []
    for s in sources or []:
        ref = str(s.get("ref") or s.get("label") or "").strip()
        if ref.startswith("KB/"):
            ref = ref[3:]
        if ref and ref not in out:
            out.append(ref)
    return out[:6]


def resolve_followup_query(query: str, session: SessionMemory | None) -> str:
    q = (query or "").strip()
    if not q or not looks_like_followup(q):
        return q
    ents = session.recent_entities() if session else []
    if not ents:
        return q
    prefix = " ".join(ents[:3])
    if any(e in q for e in ents[:3]):
        return q
    return f"{prefix} {q}".strip()


def user_id_from_env(raw: str | None = None) -> str:
    t = (raw or os.getenv("NEV_USER_ID") or "local").strip() or "local"
    t = _UNSAFE_USER.sub("_", t)[:64]
    return t or "local"


@dataclass
class Turn:
    query: str
    digest: str
    entities: list[str] = field(default_factory=list)
    cited_refs: list[str] = field(default_factory=list)
    kind: str = "qa"


@dataclass
class SessionMemory:
    turns: list[Turn] = field(default_factory=list)
    running_summary: str = ""
    queries: list[str] = field(default_factory=list)
    pending_episodes: list[dict[str, Any]] = field(default_factory=list)
    pending_preferences: list[dict[str, str]] = field(default_factory=list)

    def has_context(self) -> bool:
        return bool(self.turns or self.running_summary)

    def recent_entities(self) -> list[str]:
        seen: list[str] = []
        for turn in reversed(self.turns):
            for e in turn.entities:
                if e not in seen:
                    seen.append(e)
        return seen

    def add_preferences(self, prefs: list[dict[str, str]]) -> None:
        for p in prefs:
            key = p.get("key")
            self.pending_preferences = [x for x in self.pending_preferences if x.get("key") != key]
            self.pending_preferences.append(dict(p))

    def remember_from_result(self, query: str, out: dict[str, Any]) -> None:
        if (out or {}).get("domain") != "nev":
            return
        q = (query or "").strip()
        report = str(out.get("final_report") or "")
        sources = list(out.get("sources") or [])
        entities = extract_entities(q, report)
        refs = refs_from_sources(sources)
        digest = make_digest(report, q)
        kind = "report" if is_report_query(q) else "qa"
        self.queries.append(q)
        self.turns.append(Turn(query=q, digest=digest, entities=entities, cited_refs=refs, kind=kind))
        self.pending_episodes.append(
            {
                "ts": _now_iso(),
                "query": q,
                "summary": digest[:180],
                "entities": entities[:8],
                "source_hints": refs[:4],
                "kind": kind,
            }
        )
        prefs = extract_preferences_from_text(q)
        if prefs:
            self.add_preferences(prefs)
        self.compact()

    def _fold_oldest(self) -> None:
        if not self.turns:
            return
        old = self.turns.pop(0)
        bit = strip_numbers(f"{old.query}（{','.join(old.entities)}）")
        merged = "；".join(x for x in (self.running_summary, bit) if x)
        self.running_summary = merged[:SUMMARY_MAX_CHARS]

    def inject_len(self) -> int:
        return len(self.format_stm())

    def compact(self) -> None:
        while len(self.turns) > WINDOW_SIZE:
            self._fold_oldest()
        while self.inject_len() > INJECT_MAX_CHARS and len(self.turns) > 1:
            self._fold_oldest()
        if self.inject_len() > INJECT_MAX_CHARS and self.turns:
            last = self.turns[-1]
            last.digest = last.digest[:80]
            self.running_summary = self.running_summary[:200]

    def format_stm(self) -> str:
        parts: list[str] = []
        if self.running_summary:
            parts.append(f"更早对话摘要：{self.running_summary}")
        for i, turn in enumerate(self.turns, start=1):
            ents = "、".join(turn.entities) or "无"
            parts.append(f"近窗{i}：问={turn.query}；要点={turn.digest}；实体={ents}")
        refs: list[str] = []
        for turn in self.turns:
            for r in turn.cited_refs:
                if r not in refs:
                    refs.append(r)
        if refs:
            parts.append("本会话查过（提示，不是来源）：" + "、".join(refs[:8]))
        return "\n".join(parts)


class UserMemoryStore:
    def __init__(self, user_id: str = "local", path: Path | None = None) -> None:
        self.user_id = user_id_from_env(user_id)
        self.path = path or (MEMORY_DIR / f"{self.user_id}.json")
        self.interests: list[dict[str, Any]] = []
        self.preferences: list[dict[str, str]] = []
        self.episodes: list[dict[str, Any]] = []

    @classmethod
    def load(cls, user_id: str = "local", path: Path | None = None) -> UserMemoryStore:
        store = cls(user_id=user_id, path=path)
        if store.path.exists():
            raw = json.loads(store.path.read_text(encoding="utf-8"))
            store.interests = list(raw.get("interests") or [])
            store.preferences = list(raw.get("preferences") or [])
            store.episodes = list(raw.get("episodes") or [])
        return store

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "user_id": self.user_id,
            "updated_at": _now_iso(),
            "interests": self.interests,
            "preferences": self.preferences,
            "episodes": self.episodes[-EPISODE_CAP:],
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def top_interests(self, k: int = INTEREST_TOP_K) -> list[str]:
        ranked = sorted(self.interests, key=lambda x: float(x.get("weight") or 0), reverse=True)
        return [str(x.get("term")) for x in ranked[:k] if x.get("term")]

    def pref_value(self, key: str) -> str | None:
        for p in reversed(self.preferences):
            if p.get("key") == key:
                return p.get("value")
        return None

    def merge_interests(self, ranked: list[tuple[str, int]]) -> None:
        by_term = {str(x.get("term")): dict(x) for x in self.interests if x.get("term")}
        for item in by_term.values():
            item["weight"] = round(float(item.get("weight") or 0) * INTEREST_DECAY, 4)
        now = _now_iso()
        for term, count in ranked:
            prev = by_term.get(term, {"term": term, "weight": 0.0})
            prev["weight"] = round(float(prev.get("weight") or 0) + float(count), 4)
            prev["last_seen"] = now
            by_term[term] = prev
        self.interests = sorted(by_term.values(), key=lambda x: float(x.get("weight") or 0), reverse=True)[:40]

    def merge_preferences(self, prefs: list[dict[str, str]]) -> None:
        for p in prefs:
            key = p.get("key")
            if not key:
                continue
            self.preferences = [x for x in self.preferences if x.get("key") != key]
            self.preferences.append(dict(p))

    def add_episodes(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.episodes.append(dict(row))
        self.episodes = self.episodes[-EPISODE_CAP:]

    def retrieve_episodes(self, query: str, k: int = EPISODE_RETRIEVE_K) -> list[dict[str, Any]]:
        q_ents = set(extract_entities(query))
        if not q_ents:
            return list(self.episodes[-k:]) if query else []
        scored: list[tuple[int, dict[str, Any]]] = []
        for ep in self.episodes:
            ents = set(ep.get("entities") or [])
            overlap = len(q_ents & ents)
            if overlap:
                scored.append((overlap, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:k]]

    def ingest_session(self, session: SessionMemory) -> None:
        self.merge_interests(extract_session_interests(session))
        self.merge_preferences(session.pending_preferences)
        self.add_episodes(session.pending_episodes)

    def format_ltm(self, query: str) -> str:
        parts: list[str] = []
        interests = self.top_interests()
        if interests:
            parts.append("该用户近期关注：" + "、".join(interests))
        length = self.pref_value("length")
        if length == "short":
            parts.append("用户偏好：短答（只影响篇幅，不影响必须检索）。")
        elif length == "long":
            parts.append("用户偏好：详细作答（只影响篇幅，不影响必须检索）。")
        source_pref = self.pref_value("source_pref")
        if source_pref == "cpca":
            parts.append("用户偏好：口径优先考虑乘联会（仍须本轮工具确认）。")
        for ep in self.retrieve_episodes(query):
            ents = "、".join(ep.get("entities") or [])
            parts.append(f"过往情景：{ep.get('query')} → {ep.get('summary')}（实体：{ents}）")
        return "\n".join(parts)


def format_memory_block(
    session: SessionMemory | None,
    user_memory: UserMemoryStore | None,
    query: str = "",
) -> str:
    chunks: list[str] = []
    if user_memory:
        ltm = user_memory.format_ltm(query)
        if ltm:
            chunks.append(ltm)
    if session:
        stm = session.format_stm()
        if stm:
            chunks.append(stm)
    text = "\n".join(chunks).strip()
    if len(text) > INJECT_MAX_CHARS:
        text = text[: INJECT_MAX_CHARS - 1] + "…"
    return text
