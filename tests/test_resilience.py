"""Quality-gated retry (no network)."""

from __future__ import annotations

from agent.tools.resilience import (
    GOOD_QUALITY,
    MAX_NETWORK_TRIES,
    Strategy,
    alt_web_query,
    call_with_transient_retry,
    decide_continue,
    execute_strategies,
    filter_web_hits,
    is_transient_error,
    kb_query_variants,
    reset_circuits,
    run_kb_search,
    score_web_hits,
)


def setup_function() -> None:
    reset_circuits()


def _hit(url: str, snippet: str, title: str = "t") -> dict:
    return {"title": title, "url": url, "snippet": snippet, "provider": "x"}


def test_score_prefers_diverse_news_over_jobs():
    junk = [
        _hit("https://job.byd.com/portal/pc/", "招聘", "比亚迪招聘"),
        _hit("https://www.byd.com/cn/index.html", "官网", "首页"),
        _hit("https://baike.baidu.com/item/x", "百科", "百科"),
    ]
    news = [
        _hit(
            "https://www.reuters.com/byd-hungary-2024",
            "BYD will build a plant in Szeged with capacity around 150,000 vehicles.",
            "BYD Hungary plant",
        ),
        _hit(
            "https://www.ft.com/byd-tariffs",
            "EU provisional countervailing duties on Chinese BEVs affect BYD exports.",
            "EU EV tariffs",
        ),
        _hit(
            "https://www.bloomberg.com/byd-thailand",
            "Rayong factory started production in 2024 with a 150,000-unit plan.",
            "Thailand Rayong",
        ),
    ]
    assert score_web_hits(news) > score_web_hits(junk)
    assert score_web_hits(news) >= GOOD_QUALITY
    assert score_web_hits(junk) < GOOD_QUALITY
    assert score_web_hits([]) == 0.0


def test_decide_continue_stops_when_good():
    from agent.tools.resilience import Attempt

    good = Attempt("bing", "bing", "q", hits=[_hit("https://a.com/x", "x" * 80)], quality=0.8)
    nxt = Strategy("ddg", "duckduckgo", "q", lambda: [])
    assert decide_continue([good], nxt) is False


def test_decide_continue_tries_new_provider_if_weak():
    from agent.tools.resilience import Attempt

    weak = Attempt("ddg", "duckduckgo", "q", hits=[], error="timeout", quality=0.0)
    nxt = Strategy("bing", "bing", "q", lambda: [])
    assert decide_continue([weak], nxt) is True


def test_execute_stops_after_good_quality():
    calls = {"n": 0}

    def good():
        calls["n"] += 1
        return [
            _hit("https://reuters.com/a", "word " * 40),
            _hit("https://ft.com/b", "word " * 40),
            _hit("https://bloomberg.com/c", "word " * 40),
        ]

    def should_not_run():
        raise AssertionError("must not run after good quality")

    hits, meta = execute_strategies(
        [
            Strategy("a", "ddg", "q", good),
            Strategy("b", "bing", "q", should_not_run),
        ]
    )
    assert hits
    assert meta["quality"] >= GOOD_QUALITY
    assert meta["degraded"] is False
    assert calls["n"] == 1


def test_execute_skips_dead_provider_and_uses_next():
    def boom():
        raise TimeoutError("timed out")

    def ok():
        return [_hit("https://example.com/news", "snippet " * 20)]

    hits, meta = execute_strategies(
        [
            Strategy("ddg", "duckduckgo", "q", boom),
            Strategy("bing", "bing", "q", ok),
        ]
    )
    assert hits
    assert meta["attempts"][0]["error"]
    assert meta["attempts"][1]["provider"] == "bing"


def test_no_second_reformulation_without_gain():
    from agent.tools.resilience import Attempt

    a1 = Attempt("bing", "bing", "q", hits=[_hit("https://a.com/x", "short")], quality=0.3)
    a2 = Attempt("bing_alt", "bing", "q 新闻", hits=[_hit("https://a.com/y", "short")], quality=0.31)
    nxt = Strategy("bing_alt2", "bing", "q 最新", lambda: [])
    assert decide_continue([a1, a2], nxt) is False


def test_transient_retry_then_ok():
    box = {"n": 0}

    def flaky():
        box["n"] += 1
        if box["n"] == 1:
            raise ConnectionError("reset")
        return "ok"

    val, err, tries = call_with_transient_retry(flaky, max_tries=2)
    assert val == "ok"
    assert err is None
    assert tries == 2


def test_permanent_error_no_retry():
    box = {"n": 0}

    def missing():
        box["n"] += 1
        raise FileNotFoundError("no index")

    val, err, tries = call_with_transient_retry(missing, max_tries=2)
    assert val is None
    assert "no index" in (err or "")
    assert tries == 1
    assert box["n"] == 1


def test_is_transient():
    assert is_transient_error(TimeoutError("x"))
    assert is_transient_error(ConnectionError("reset"))
    assert not is_transient_error(ValueError("bad query"))


def test_alt_web_query():
    assert alt_web_query("BYD overseas") == "BYD overseas 2024"
    assert alt_web_query("BYD 新闻") is None
    assert alt_web_query("2024年中国新能源乘用车渗透率数据") == "中国新能源乘用车渗透率数据"


def test_filter_web_hits_drops_year_and_movies():
    q = "2024年中国新能源乘用车渗透率数据"
    junk = [
        _hit(
            "https://baike.baidu.com/item/2024%E5%B9%B4/7433925",
            "2024年是闰年",
            "2024年_百度百科",
        ),
        _hit(
            "https://www.douban.com/doulist/157417687/",
            "电影列表",
            "2024年公映电影汇总",
        ),
        _hit(
            "https://www.stats.gov.cn/sj/ndsj/2024/indexch.htm",
            "年鉴目录",
            "中国统计年鉴2024",
        ),
    ]
    good = _hit(
        "https://www.cpcaauto.com/news/penetration",
        "2024年新能源乘用车零售渗透率约47.6%",
        "乘联会：新能源乘用车渗透率",
    )
    kept = filter_web_hits(junk + [good], q)
    assert len(kept) == 1
    assert "cpcaauto" in kept[0]["url"]


def test_execute_retries_alt_query_after_offtopic_hits():
    q = "中国新能源乘用车渗透率数据 2024年"
    alt = "中国新能源乘用车渗透率数据"
    calls = {"n": 0}

    def junk():
        calls["n"] += 1
        return [
            _hit(
                "https://baike.baidu.com/item/2024%E5%B9%B4/1",
                "闰年",
                "2024年_百度百科",
            )
        ]

    def nev():
        calls["n"] += 1
        return [
            _hit(
                "https://www.iea.org/reports/global-ev-outlook-2025",
                "China NEV passenger car share near 50% in 2024.",
                "IEA Global EV Outlook",
            )
        ]

    hits, meta = execute_strategies(
        [
            Strategy("bing", "bing", q, junk),
            Strategy("bing_alt", "bing", alt, nev),
        ]
    )
    assert any("iea.org" in (h.get("url") or "") for h in hits)
    assert calls["n"] == 2
    assert meta["quality"] > 0


def test_kb_query_variants_are_distinct():
    vs = kb_query_variants("比亚迪刀片电池的核心卖点是什么？")
    assert vs[0] != vs[1] if len(vs) > 1 else True
    assert len(set(vs)) == len(vs)


def test_kb_retry_changes_params_and_skips_same_file():
    from kb import Chunk, Hit

    log: list[tuple] = []

    def fn(q: str, k: int, ex_ids: set[str], ex_src: set[str]):
        log.append((q, k, frozenset(ex_ids), frozenset(ex_src)))
        if not ex_src:
            return [
                Hit(Chunk("a.md::0", "t", "a.md", "t"), 0.9),
                Hit(Chunk("a.md::1", "t", "a.md", "t"), 0.8),
            ]
        return [
            Hit(Chunk("b.md::0", "t", "b.md", "t"), 0.7),
            Hit(Chunk("b.md::1", "t", "b.md", "t"), 0.6),
        ]

    hits, meta = run_kb_search("比亚迪刀片电池的核心卖点是什么？", 4, fn)
    assert len(log) >= 2
    assert log[0][0] != log[1][0] or log[0][1] != log[1][1] or log[1][3]
    assert {h.chunk.source for h in hits} >= {"a.md", "b.md"}
    assert meta["tries"] <= MAX_NETWORK_TRIES


def test_kb_connection_respects_network_budget():
    n = {"c": 0}

    def fn(*_a, **_k):
        n["c"] += 1
        raise ConnectionError("reset")

    hits, meta = run_kb_search("刀片电池", 4, fn, network_budget=3)
    assert hits == []
    assert n["c"] == 3
    assert meta["tries"] == 3
    assert meta.get("error")
