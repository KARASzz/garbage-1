"""Weighted grammar expansion over a validated SQLite corpus snapshot."""

from __future__ import annotations

from collections import Counter, deque
import hashlib
import random
import re
import threading

from .storage import CorpusDatabase, CorpusError, matches


class GrammarGenerator:
    def __init__(self, database: CorpusDatabase, *, seed: int | None = None):
        self.fragments, self.templates, self.stats = database.snapshot()
        if not self.templates or not self.fragments:
            raise CorpusError("数据库没有可抽取的语料")
        self.random = random.Random(seed)
        self.lock = threading.Lock()
        self.recent_texts: deque[str] = deque(maxlen=128)
        self.recent_fragments: deque[str] = deque(maxlen=80)
        self.recent_templates: deque[str] = deque(maxlen=3)
        self.pools: dict[tuple[str, str], list[dict]] = {}
        for template in self.templates:
            for segment in template["segments"]:
                if not isinstance(segment, dict):
                    continue
                pool = [f for f in self.fragments if matches(f, segment)]
                if not pool:
                    raise CorpusError(f"空语料池：{template['id']}/{segment['slot']}")
                self.pools[template["id"], segment["slot"]] = pool

    def _pick_template(self) -> dict:
        weights = [t["weight"] * (0.18 if t["id"] in self.recent_templates else 1) for t in self.templates]
        return self.random.choices(self.templates, weights=weights, k=1)[0]

    def _expand(self, template: dict, *, include_optional: bool | None = None) -> dict:
        parts: list[str] = []
        draws: list[dict] = []
        used_ids: set[str] = set()
        recent = Counter(self.recent_fragments)
        for segment in template["segments"]:
            if isinstance(segment, str):
                parts.append(segment)
                continue
            probability = segment.get("optional")
            if probability is not None:
                include = include_optional if include_optional is not None else self.random.random() < probability
                if not include:
                    continue
            pool = [f for f in self.pools[template["id"], segment["slot"]] if f["id"] not in used_ids]
            if not pool:
                raise CorpusError(f"句型槽位无法无重复填充：{template['id']}/{segment['slot']}")
            weights = [f["weight"] / (1 + 2 * recent[f["id"]]) for f in pool]
            fragment = self.random.choices(pool, weights=weights, k=1)[0]
            used_ids.add(fragment["id"])
            parts.extend((segment.get("prefix", ""), fragment["text"], segment.get("suffix", "")))
            draws.append({"slot": segment["slot"], "id": fragment["id"], "category": fragment["category"], "text": fragment["text"], "tags": fragment["tags"]})
        text = "".join(parts).strip()
        if not text.endswith(("。", "！", "？", "!", "?")):
            text += "。"
        if not 6 <= len(text) <= 90 or re.search(r"[，、；：]{2}|[。！？]{2}|[{}\n\r]", text):
            raise CorpusError(f"句型输出长度或标点无效：{template['id']}")
        return {"text": text, "id": hashlib.sha256(text.encode()).hexdigest()[:12], "theme": template["theme"], "block_count": len(draws), "template_id": template["id"], "draws": draws}

    def draw(self, *, debug: bool = False, template_id: str | None = None, include_optional: bool | None = None) -> dict:
        """Return a fresh sentence; seeded construction supports repeatable audits."""
        with self.lock:
            fixed_template = None
            if template_id is not None:
                fixed_template = next((t for t in self.templates if t["id"] == template_id), None)
                if fixed_template is None:
                    raise CorpusError(f"未知句型：{template_id}")
            failure = None
            result = None
            for _ in range(48):
                template = fixed_template or self._pick_template()
                try:
                    candidate = self._expand(template, include_optional=include_optional)
                except CorpusError as error:
                    failure = error
                    continue
                result = candidate
                if candidate["text"] not in self.recent_texts:
                    break
            else:
                # A tiny user-edited pool may exhaust recent uniqueness. Still
                # return a grammatical candidate rather than fabricate a string.
                if result is None:
                    raise CorpusError("语料无法组成完整句子，请检查句型和语料长度") from failure
            self.recent_texts.append(result["text"])
            self.recent_templates.append(result["template_id"])
            self.recent_fragments.extend(d["id"] for d in result["draws"])
            if debug:
                return result
            return {key: result[key] for key in ("text", "id", "theme", "block_count")}
