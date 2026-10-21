#!/usr/bin/env python3
"""Build a deterministic, sentence-disjoint corpus for garbage-1 training.

All grammar examples are drawn through the production SQLite-backed generator.
The SQLite database is created under a temporary directory and is discarded
when this script exits; the application's data/trash_talk.sqlite3 is untouched.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unicodedata
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trash_talk.generator import GrammarGenerator
from trash_talk.garbage_model import CharacterTokenizer, solve_config
from trash_talk.storage import CorpusDatabase


DEFAULT_SEED = 20261021
DEFAULT_GENERATED_COUNT = 16_000
REPLAY_COUNT = 512
SPLIT_DESCRIPTION = (
    "Normalize each complete sentence to NFC, hash UTF-8 with SHA-256, sort by "
    "(digest, normalized text), then assign the first floor(90%) to train, the "
    "next floor(5%) to validation, and the remainder to test. A sentence occurs "
    "in exactly one split. Shared templates and grammar fragments cross splits, "
    "so evaluation is in-domain and does not measure novel-template generalization."
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sentence_digest(text: str) -> str:
    return sha256_bytes(unicodedata.normalize("NFC", text).encode("utf-8"))


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        with temp_path.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def draw_with_seed(generator: GrammarGenerator, seed: int) -> dict:
    """Make one independent, replayable draw using the real grammar generator."""
    generator.random.seed(seed)
    generator.recent_texts.clear()
    generator.recent_fragments.clear()
    generator.recent_templates.clear()
    return generator.draw(debug=True)


def load_curated(path: Path) -> list[dict]:
    rows = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"curated.jsonl 第 {line_no} 行不是有效 JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"curated.jsonl 第 {line_no} 行必须是 JSON 对象")
        text = row.get("text")
        theme = row.get("theme")
        row_id = row.get("id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"curated.jsonl 第 {line_no} 行缺少有效 text")
        if theme not in {"game", "logic", "boast", "daily"}:
            raise ValueError(f"curated.jsonl 第 {line_no} 行 theme 无效")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"curated.jsonl 第 {line_no} 行缺少有效 id")
        text = unicodedata.normalize("NFC", text)
        digest = sentence_digest(text)
        if digest in seen:
            raise ValueError(f"curated.jsonl 包含重复完整句：{row_id}")
        seen.add(digest)
        rows.append(
            {
                "id": row_id,
                "text": text,
                "theme": theme,
                "source": "curated_seed",
                "template_id": None,
                "draws": [],
            }
        )
    if len(rows) != 120:
        raise ValueError(f"预期 120 条 curated 人工种子，实际 {len(rows)} 条")
    return rows


def build_dataset(
    *, root: Path, generated_count: int, seed: int, max_attempts: int | None = None
) -> dict:
    corpus_path = root / "data/corpus.json"
    curated_path = root / "data/garbage-1/curated.jsonl"
    corpus_raw = corpus_path.read_bytes()
    curated_raw = curated_path.read_bytes()
    curated = load_curated(curated_path)
    reserved = {sentence_digest(row["text"]) for row in curated}
    if max_attempts is None:
        max_attempts = generated_count * 100

    generated: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="garbage-1-corpus-") as temporary:
        database = CorpusDatabase(Path(temporary) / "temporary.sqlite3")
        database_stats = database.initialize(corpus_path)
        generator = GrammarGenerator(database, seed=seed)
        attempt = 0
        while len(generated) < generated_count and attempt < max_attempts:
            attempt += 1
            seed_material = f"garbage-1-v1:{seed}:{attempt}".encode("ascii")
            draw_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
            result = draw_with_seed(generator, draw_seed)
            text = unicodedata.normalize("NFC", result["text"])
            digest = sentence_digest(text)
            if digest in reserved or digest in generated:
                continue
            generated[digest] = {
                "id": digest,
                "text": text,
                "theme": result["theme"],
                "source": "grammar_generated",
                "template_id": result["template_id"],
                "draws": result["draws"],
                "seed": draw_seed,
                "draw_index": attempt,
            }
        if len(generated) != generated_count:
            raise RuntimeError(
                f"在 {max_attempts} 次真实语法抽取后只得到 {len(generated)} 条唯一生成句"
            )

    rows = curated + list(generated.values())
    rows.sort(key=lambda row: (sentence_digest(row["text"]), row["text"]))
    total = len(rows)
    train_end = total * 90 // 100
    validation_end = total * 95 // 100
    splits = {
        "train": rows[:train_end],
        "validation": rows[train_end:validation_end],
        "test": rows[validation_end:],
    }

    # A full-text digest may appear in only one output file.
    seen_by_split: dict[str, str] = {}
    for split_name, split_rows in splits.items():
        for row in split_rows:
            digest = sentence_digest(row["text"])
            previous = seen_by_split.setdefault(digest, split_name)
            if previous != split_name:
                raise AssertionError(f"完整句泄漏到 {previous} 和 {split_name}")
    if len(seen_by_split) != total:
        raise AssertionError("句子哈希存在跨 split 重复")

    tokenizer = CharacterTokenizer.fit([row["text"] for row in splits["train"]])
    model_config = solve_config(tokenizer.vocab_size).to_dict()

    def distribution(split_rows: list[dict], field: str) -> dict:
        counter = Counter(row[field] for row in split_rows)
        size = len(split_rows)
        return {
            key: {"count": counter[key], "ratio": counter[key] / size if size else 0.0}
            for key in sorted(counter)
        }

    grammar_train = [
        row for row in splits["train"] if row["source"] == "grammar_generated"
    ]
    if len(grammar_train) < REPLAY_COUNT:
        raise RuntimeError("train 中真实生成句不足 512 条，无法建立 seed replay")
    replay_rows = sorted(grammar_train, key=lambda row: row["draw_index"])[:REPLAY_COUNT]
    replay = {
        "format_version": 1,
        "description": "每条均由 SQLite GrammarGenerator 按独立 seed 生成，全部来自 train。",
        "seed_policy": "int.from_bytes(SHA256('garbage-1-v1:{base_seed}:{attempt}')[:8], 'big')",
        "base_seed": seed,
        "count": len(replay_rows),
        "examples": [
            {
                "seed": row["seed"],
                "draw_index": row["draw_index"],
                "theme": row["theme"],
                "text": row["text"],
            }
            for row in replay_rows
        ],
    }

    # Prefixes are short, direct prefixes of train sentences only. Each stays
    # shorter than its source sentence and carries the original theme label.
    prefixes = []
    for row in splits["train"]:
        text = row["text"]
        if len(text) <= 3:
            continue
        prefix_length = 4
        prefix = text[:prefix_length]
        if len(prefix) >= len(text):
            raise AssertionError("prefix 不能覆盖完整句")
        prefixes.append(
            {
                "source_id": row["id"],
                "theme": row["theme"],
                "prefix": prefix,
            }
        )

    split_payloads = {
        name: b"".join(canonical_json(row) + b"\n" for row in split_rows)
        for name, split_rows in splits.items()
    }
    split_hashes = {name: sha256_bytes(payload) for name, payload in split_payloads.items()}
    dataset_digest = sha256_bytes(
        canonical_json({"splits": split_hashes, "total_count": total})
    )
    source_material = {
        "corpus_sha256": sha256_bytes(corpus_raw),
        "curated_sha256": sha256_bytes(curated_raw),
        "base_seed": seed,
        "generated_count": generated_count,
        "dataset_sha256": dataset_digest,
        "generator": "trash_talk.generator.GrammarGenerator",
    }
    source_sha = sha256_bytes(canonical_json(source_material))

    manifest = {
        "format": "garbage-1-dataset-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "corpus_sha256": source_material["corpus_sha256"],
        "curated_sha256": source_material["curated_sha256"],
        "dataset_source_sha256": source_sha,
        "source_inputs": {
            "corpus.json": source_material["corpus_sha256"],
            "curated.jsonl": source_material["curated_sha256"],
        },
        "dataset_sha256": dataset_digest,
        "base_seed": seed,
        "generated_sentence_count": len(generated),
        "curated_sentence_count": len(curated),
        "total_sentence_count": total,
        "generation_attempts": attempt,
        "temporary_database": {
            "format": "SQLite initialized from data/corpus.json",
            "path": "temporary directory; removed after preparation",
            "database_source_sha256": database_stats["source_sha256"],
            "fragments": database_stats["fragments"],
            "templates": database_stats["templates"],
            "categories": database_stats["categories"],
        },
        "split_method": SPLIT_DESCRIPTION,
        "split_counts": {name: len(split_rows) for name, split_rows in splits.items()},
        "split_sha256": split_hashes,
        "split_distributions": {
            name: {
                "theme": distribution(split_rows, "theme"),
                "source": distribution(split_rows, "source"),
            }
            for name, split_rows in splits.items()
        },
        "overall_distributions": {
            "theme": distribution(rows, "theme"),
            "source": distribution(rows, "source"),
        },
        "training_only_tokenizer": {
            "fit_split": "train",
            "vocab_size": tokenizer.vocab_size,
            "exact_100k_parameter_config": model_config,
        },
        "seed_replay_count": len(replay_rows),
        "prefix_count": len(prefixes),
        "evaluation_scope": "域内评估；模板与语料块跨 split 共享，不能据此声称未见模板泛化。",
    }

    output_dir = root / "data/garbage-1"
    for name, payload in split_payloads.items():
        atomic_write(output_dir / f"{name}.jsonl", payload)
    atomic_write(output_dir / "seed_replay.json", canonical_json(replay) + b"\n")
    atomic_write(
        output_dir / "prefixes.json",
        canonical_json({"format_version": 1, "prefix_length": 4, "count": len(prefixes), "examples": prefixes})
        + b"\n",
    )
    atomic_write(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--generated-count", type=int, default=DEFAULT_GENERATED_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-attempts", type=int, default=None)
    args = parser.parse_args()
    if args.generated_count <= 0:
        parser.error("--generated-count 必须为正数")
    if args.max_attempts is not None and args.max_attempts < args.generated_count:
        parser.error("--max-attempts 必须不少于 --generated-count")
    result = build_dataset(
        root=args.root.resolve(),
        generated_count=args.generated_count,
        seed=args.seed,
        max_attempts=args.max_attempts,
    )
    print(
        json.dumps(
            {
                "status": "prepared",
        "dataset_source_sha256": result["dataset_source_sha256"],
                "generated": result["generated_sentence_count"],
                "curated": result["curated_sentence_count"],
                "splits": result["split_counts"],
                "tokenizer_vocab_size": result["training_only_tokenizer"]["vocab_size"],
                "config": result["training_only_tokenizer"]["exact_100k_parameter_config"],
                "seed_replay": result["seed_replay_count"],
                "prefixes": result["prefix_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
