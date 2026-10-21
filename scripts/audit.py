#!/usr/bin/env python3
"""Exercise real database rules across every template and a large random draw."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trash_talk.generator import GrammarGenerator
from trash_talk.storage import CorpusDatabase, matches


def audit(count: int) -> dict:
    database = CorpusDatabase(ROOT / "data/trash_talk.sqlite3")
    stats = database.initialize(ROOT / "data/corpus.json")
    generator = GrammarGenerator(database, seed=20261020)
    fragment_map = {f["id"]: f for f in generator.fragments}
    template_map = {t["id"]: t for t in generator.templates}
    reachable_fragments = {f["id"] for pool in generator.pools.values() for f in pool}

    def check(result: dict) -> None:
        template = template_map[result["template_id"]]
        drawn = {d["slot"]: d for d in result["draws"]}
        assert len(drawn) == len(result["draws"]) == result["block_count"]
        assert len({d["id"] for d in result["draws"]}) == result["block_count"]
        rebuilt = []
        for segment in template["segments"]:
            if isinstance(segment, str):
                rebuilt.append(segment)
            elif segment["slot"] in drawn:
                fragment = fragment_map[drawn[segment["slot"]]["id"]]
                assert matches(fragment, segment), (template["id"], segment["slot"])
                rebuilt.extend((segment.get("prefix", ""), fragment["text"], segment.get("suffix", "")))
            else:
                assert "optional" in segment, (template["id"], segment["slot"])
        expected = "".join(rebuilt).strip()
        if not expected.endswith(("。", "！", "？", "!", "?")):
            expected += "。"
        assert result["text"] == expected
        assert 6 <= len(result["text"]) <= 90
        assert not re.search(r"[{}\n\r]|[，、；：]{2}|[。！？]{2}", result["text"])

    examples = []
    for template in generator.templates:
        for include in (False, True):
            result = generator.draw(debug=True, template_id=template["id"], include_optional=include)
            check(result)
            examples.append({"template": template["id"], "name": template["name"], "optional": include, "text": result["text"], "block_count": result["block_count"]})
    unique = set()
    templates = Counter()
    themes = Counter()
    blocks = Counter()
    lengths = []
    immediate_duplicates = 0
    profanity_count = 0
    profanity_terms = ("他妈", "妈的", "傻逼", "放屁", "狗屎", "装逼", "屁用", "废物")
    previous = None
    for _ in range(count):
        result = generator.draw(debug=True)
        check(result)
        unique.add(result["text"])
        templates[result["template_id"]] += 1
        themes[result["theme"]] += 1
        blocks[result["block_count"]] += 1
        lengths.append(len(result["text"]))
        immediate_duplicates += result["text"] == previous
        previous = result["text"]
        profanity_count += any(term in result["text"] for term in profanity_terms)
    assert len(templates) == len(generator.templates), "随机抽样未覆盖全部句型"
    assert len(blocks) >= 3, "语料块数量缺少变化"
    assert immediate_duplicates == 0, "出现连续重复句子"
    assert len(unique) / count > 0.95, "句子多样性不足"
    assert profanity_count / count >= 0.6, "语料口味没有达到已选定的嘴臭强度"
    return {
        "status": "passed",
        "seed": 20261020,
        "database": stats,
        "reachable_fragments": len(reachable_fragments),
        "unused_fragment_ids": sorted(set(fragment_map) - reachable_fragments),
        "random_draws": count,
        "unique_sentences": len(unique),
        "unique_ratio": round(len(unique) / count, 5),
        "immediate_duplicates": immediate_duplicates,
        "profanity_ratio": round(profanity_count / count, 5),
        "block_count_distribution": dict(sorted(blocks.items())),
        "length": {"min": min(lengths), "max": max(lengths), "mean": round(sum(lengths) / count, 2)},
        "template_distribution": dict(sorted(templates.items())),
        "theme_distribution": dict(themes),
        "template_extremes": examples,
        "checks": ["every_required_slot", "optional_slots_both_extremes", "category_and_semantic_tags", "no_repeated_fragment_id", "exact_reconstruction", "punctuation_and_length", "all_templates_covered", "variable_block_count", "recent_sentence_deduplication"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=ROOT / "output/generation-audit.json")
    args = parser.parse_args()
    if args.count < 1000:
        parser.error("完整抽样审计至少需要 1000 条")
    result = audit(args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("template_extremes", "template_distribution", "unused_fragment_ids")}, ensure_ascii=False, indent=2))
