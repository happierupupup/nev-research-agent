#!/usr/bin/env python3
"""
根据 chunk 正文主题关键词生成 chunk 级金标候选（不是按检索结果回标）。

用法:
  PYTHONPATH=src python scripts/build_chunk_qrels.py \\
    --out evals/datasets/retrieval/qrels_v3.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

# 每题：问句 + 正文须命中的关键词（内容标注）+ 可选 source 白名单
SPECS: list[dict] = [
    {
        "id": "C1",
        "query": "比亚迪刀片电池的核心卖点是什么？",
        "any": ["刀片", "安全性", "空间利用率"],
        "sources": ["byd_blade_battery.md"],
        "max": 3,
        "tags": [ "blade", "seed"],
    },
    {
        "id": "C2",
        "query": "磷酸铁锂和三元锂在成本与能量密度上有什么区别？",
        "any": ["磷酸铁锂", "三元", "LFP", "NMC", "能量密度"],
        "sources": ["battery_lfp_nmc.md", "byd_blade_battery.md"],
        "max": 4,
        "tags": ["chemistry"],
    },
    {
        "id": "C3",
        "query": "宁德时代全球动力电池装车份额大概多少？",
        "any": ["宁德", "装车", "份额", "市占"],
        "sources": ["catl_supply_chain.md"],
        "max": 3,
        "tags": ["catl"],
    },
    {
        "id": "C4",
        "query": "欧盟对中国电动汽车加征关税如何影响出海？",
        "any": ["关税", "反补贴", "欧盟", "壁垒"],
        "sources": ["ev_trade_barriers.md", "byd_overseas_2024.md"],
        "max": 4,
        "tags": ["trade"],
    },
    {
        "id": "C5",
        "query": "蔚来换电和 BaaS 是怎么运作的？",
        "any": ["换电", "BaaS", "蔚来"],
        "sources": [
            "nio_battery_swap.md",
            "crawl/nio_swap_2026_1_www_nio_cn.md",
            "crawl/nio_swap_2026_2_www_nio_cn.md",
            "crawl/nio_swap_2026_3_www_nio_cn.md",
            "charging_and_swap.md",
        ],
        "max": 4,
        "tags": ["nio", "diversify"],
    },
    {
        "id": "C6",
        "query": "小鹏智驾 XNGP 或端到端路线强调什么？",
        "any": ["小鹏", "智驾", "XNGP", "端到端"],
        "sources": [
            "xpeng_adas.md",
            "crawl/xpeng_ad_2026_1_www_xiaopeng_com.md",
            "crawl/xpeng_ad_2026_2_www_xiaopeng_com.md",
        ],
        "max": 4,
        "tags": ["xpeng", "diversify"],
    },
    {
        "id": "C7",
        "query": "理想汽车为什么坚持增程式？",
        "any": ["理想", "增程", "REEV"],
        "sources": [
            "li_auto_range_extender.md",
            "crawl/li_auto_er_2026_1_www_chnfund_com.md",
            "crawl/li_auto_er_2026_2_nev_ofweek_com.md",
            "crawl/li_auto_er_2026_3_m_gasgoo_com.md",
            "china_new_forces.md",
        ],
        "max": 4,
        "tags": ["li_auto", "diversify"],
    },
    {
        "id": "C8",
        "query": "LG 新能源在动力电池供应里扮演什么角色？",
        "any": ["LG", "LGES", "LG新能源", "LG 新能源"],
        "sources": [
            "lg_energy_solution.md",
            "crawl/lges_battery_2026_1_www_3c-ate_com.md",
            "crawl/lges_battery_2026_3_www_lgensol_com.md",
        ],
        "max": 4,
        "tags": ["lges", "diversify"],
    },
    {
        "id": "C9",
        "query": "松下给特斯拉供哪些电芯规格？",
        "any": ["松下", "2170", "4680", "特斯拉"],
        "sources": [
            "panasonic_tesla_cells.md",
            "crawl/panasonic_cell_2026_1_www_guancha_cn.md",
            "crawl/panasonic_cell_2026_3_auto_news18a_com.md",
        ],
        "max": 4,
        "tags": ["panasonic", "diversify"],
    },
    {
        "id": "C10",
        "query": "大众 MEB 平台和 ID 系列是什么关系？",
        "any": ["MEB", "大众", "ID."],
        "sources": [
            "vw_meb_platform.md",
            "crawl/vw_meb_2026_1_www_jiemian_com.md",
            "crawl/vw_meb_2026_2_nev_ofweek_com.md",
            "crawl/vw_meb_2026_3_volkswagengroupchina_com_cn.md",
        ],
        "max": 4,
        "tags": ["vw", "diversify"],
    },
    {
        "id": "C11",
        "query": "2024 年中国新能源乘用车渗透率大概多少？",
        "any": ["渗透率", "乘联会", "新能源乘用车"],
        "sources": ["nev_penetration_2024.md", "iea_global_ev_2024_2025.md"],
        "max": 4,
        "tags": ["penetration"],
    },
    {
        "id": "C12",
        "query": "以旧换新买新能源车能拿多少补贴？",
        "any": ["以旧换新", "补贴", "报废"],
        "sources": [
            "china_nev_policy_2024.md",
            "crawl/china_nev_policy_2026_1_chejiahao_autohome_com_cn.md",
            "crawl/china_nev_policy_2026_2_www_ndrc_gov_cn.md",
        ],
        "max": 4,
        "tags": ["policy"],
    },
    {
        "id": "C13",
        "query": "特斯拉和比亚迪在成本与电池技术路线上如何对比？",
        "any": ["特斯拉", "比亚迪", "成本", "4680", "刀片"],
        "sources": ["tesla_vs_byd_2024.md", "battery_lfp_nmc.md", "byd_blade_battery.md"],
        "max": 5,
        "tags": ["compete"],
    },
    {
        "id": "C14",
        "query": "锂镍钴上游资源集中带来哪些供应链风险？",
        "any": ["锂", "镍", "钴", "上游", "集中"],
        "sources": ["upstream_lithium_nickel_cobalt.md"],
        "max": 3,
        "tags": ["upstream"],
    },
    {
        "id": "C15",
        "query": "三星 SDI 圆柱电池和装车布局有何看点？",
        "any": ["三星", "SDI", "圆柱"],
        "sources": [
            "crawl/samsung_sdi_2026_1_mp_ofweek_com.md",
            "crawl/samsung_sdi_2026_2_www_leiphone_com.md",
            "crawl/samsung_sdi_2026_3_www_aibanglib_com.md",
        ],
        "max": 4,
        "tags": ["samsung", "diversify"],
    },
    {
        "id": "C16",
        "query": "极氪或吉利纯电品牌销量与出口情况如何？",
        "any": ["极氪", "吉利"],
        "sources": [
            "crawl/geely_zeekr_2026_1_www_21jingji_com.md",
            "crawl/geely_zeekr_2026_2_m_gasgoo_com.md",
        ],
        "max": 4,
        "tags": ["zeekr", "diversify"],
    },
    # 易混：问刀片，金标只能是刀片相关 chunk，干扰多品牌库
    {
        "id": "H1",
        "query": "只问刀片电池安全性，不要蔚来换电或小鹏智驾",
        "any": ["刀片", "安全"],
        "sources": ["byd_blade_battery.md"],
        "max": 2,
        "tags": ["hard", "blade"],
    },
    {
        "id": "H2",
        "query": "宁德时代市占，不要和 LG、三星装车混为一谈",
        "any": ["宁德", "份额", "装车"],
        "sources": ["catl_supply_chain.md"],
        "max": 2,
        "tags": ["hard", "catl"],
    },
    {
        "id": "H3",
        "query": "换电网络密度，不是增程也不是刀片电池",
        "any": ["换电"],
        "sources": ["nio_battery_swap.md", "charging_and_swap.md", "crawl/nio_swap_2026_1_www_nio_cn.md"],
        "max": 3,
        "tags": ["hard", "swap"],
    },
    {
        "id": "H4",
        "query": "大众 MEB 电动平台，不是中国新势力增程",
        "any": ["MEB", "大众"],
        "sources": ["vw_meb_platform.md", "crawl/vw_meb_2026_1_www_jiemian_com.md"],
        "max": 3,
        "tags": ["hard", "vw"],
    },
]


def _load_chunks(index_dir: Path) -> list[dict]:
    path = index_dir / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run ingest first")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _match(chunk: dict, spec: dict) -> bool:
    text = chunk.get("text") or ""
    src = chunk.get("source") or ""
    sources = spec.get("sources") or []
    if sources and src not in sources:
        return False
    keys = spec.get("any") or []
    return any(k in text for k in keys)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, default=ROOT / "data" / "kb_index")
    p.add_argument("--out", type=Path, default=ROOT / "evals" / "datasets" / "retrieval" / "qrels_v3.yaml")
    p.add_argument("--k", type=int, default=8)
    args = p.parse_args()

    chunks = _load_chunks(args.index)
    queries = []
    for spec in SPECS:
        hits = [c for c in chunks if _match(c, spec)]
        # 稳定：按 chunk_index
        hits.sort(key=lambda c: (c.get("source") or "", int(c.get("chunk_index") or 0)))
        picked = hits[: int(spec.get("max") or 3)]
        if not picked:
            print(f"WARN: {spec['id']} 无匹配 chunk", flush=True)
            continue
        queries.append(
            {
                "id": spec["id"],
                "query": spec["query"],
                "relevant_chunks": [c["id"] for c in picked],
                "tags": spec.get("tags") or [],
                "note": f"content-match any={spec.get('any')}; n={len(picked)}/{len(hits)}",
            }
        )

    doc = {
        "version": "3.0",
        "level": "chunk",
        "k": args.k,
        "domain": "新能源汽车",
        "notes": (
            "chunk 级金标；按正文关键词+source 白名单标注，非检索回标。"
            "语料含多品牌 seed + diversify crawl。评测默认 expand_neighbors=0。"
        ),
        "queries": queries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Wrote {args.out}  queries={len(queries)}  chunks_index={len(chunks)}")


if __name__ == "__main__":
    main()
