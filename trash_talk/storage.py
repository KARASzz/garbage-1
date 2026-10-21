"""Validate the editable corpus and keep its normalized SQLite database in sync."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import apsw

from . import SQLITE_TARGET

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.sql"
RULE_KEYS = {"tags_all": "all", "tags_any": "any", "exclude_tags": "exclude"}


class CorpusError(ValueError):
    """The corpus cannot safely produce complete sentences."""


def matches(fragment: dict, slot: dict) -> bool:
    tags = set(fragment["tags"])
    return (
        fragment["category"] == slot["category"]
        and set(slot.get("tags_all", [])) <= tags
        and (not slot.get("tags_any") or bool(tags & set(slot["tags_any"])))
        and not (tags & set(slot.get("exclude_tags", [])))
    )


def _positive(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def validate_source(data: dict) -> None:
    if data.get("version") != 1:
        raise CorpusError("语料格式版本必须为 1")
    for key in ("categories", "fragments", "templates"):
        if not isinstance(data.get(key), list) or not data[key]:
            raise CorpusError(f"缺少非空列表：{key}")
        ids = [item.get("id") for item in data[key]]
        if any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
            raise CorpusError(f"{key} 的 ID 为空或重复")
    categories = {c["id"] for c in data["categories"]}
    for fragment in data["fragments"]:
        if fragment.get("category") not in categories:
            raise CorpusError(f"未知语法类别：{fragment['id']}")
        value = fragment.get("text")
        if not isinstance(value, str) or not value.strip() or any(c in value for c in "{}\n\r"):
            raise CorpusError(f"语料为空、含换行或未解析占位符：{fragment['id']}")
        tags = fragment.get("tags")
        if not isinstance(tags, list) or any(not isinstance(t, str) or not t for t in tags) or len(set(tags)) != len(tags):
            raise CorpusError(f"语料标签无效：{fragment['id']}")
        if not _positive(fragment.get("weight", 1)):
            raise CorpusError(f"语料权重必须为正数：{fragment['id']}")
    for template in data["templates"]:
        if not _positive(template.get("weight", 1)) or not template.get("name") or not template.get("theme"):
            raise CorpusError(f"句型信息无效：{template['id']}")
        segments = template.get("segments")
        if not isinstance(segments, list) or not segments:
            raise CorpusError(f"句型为空：{template['id']}")
        slot_names = set()
        required_count = 0
        for segment in segments:
            if isinstance(segment, str):
                if any(c in segment for c in "{}\n\r"):
                    raise CorpusError(f"固定结构包含占位符：{template['id']}")
                continue
            if not isinstance(segment, dict) or not segment.get("slot") or segment.get("category") not in categories:
                raise CorpusError(f"句型槽位无效：{template['id']}")
            if segment["slot"] in slot_names:
                raise CorpusError(f"槽位名称重复：{template['id']}/{segment['slot']}")
            slot_names.add(segment["slot"])
            for key in RULE_KEYS:
                if key in segment and (not isinstance(segment[key], list) or any(not isinstance(t, str) or not t for t in segment[key])):
                    raise CorpusError(f"槽位标签规则无效：{template['id']}")
            for key in ("prefix", "suffix"):
                if key in segment and not isinstance(segment[key], str):
                    raise CorpusError(f"槽位标点无效：{template['id']}")
            probability = segment.get("optional")
            if probability is not None:
                if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
                    raise CorpusError(f"可选槽概率无效：{template['id']}")
            else:
                required_count += 1
            if not any(matches(f, segment) for f in data["fragments"]):
                raise CorpusError(f"槽位没有兼容语料：{template['id']}/{segment['slot']}")
        if required_count < 3:
            raise CorpusError(f"句型至少需要三个必选语料槽：{template['id']}")


class CorpusDatabase:
    def __init__(self, path: Path):
        self.path = Path(path)

    def connect(self, *, readonly: bool = False) -> apsw.Connection:
        flags = apsw.SQLITE_OPEN_READONLY if readonly else apsw.SQLITE_OPEN_READWRITE | apsw.SQLITE_OPEN_CREATE
        connection = apsw.Connection(str(self.path), flags=flags)
        connection.set_busy_timeout(5000)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def initialize(self, source: Path) -> dict:
        if tuple(map(int, apsw.sqlitelibversion().split("."))) < tuple(map(int, SQLITE_TARGET.split("."))):
            raise RuntimeError(f"需要 SQLite >= {SQLITE_TARGET}，实际 {apsw.sqlitelibversion()}")
        raw = Path(source).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        validate_source(data)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self.connect()
        try:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_schema WHERE type='table'")}
            if tables and "metadata" not in tables:
                raise CorpusError("目标数据库不属于本项目，拒绝覆盖")
            if "metadata" in tables:
                meta = dict(connection.execute("SELECT key,value FROM metadata"))
                if meta and meta.get("format") != "trash-talk-corpus-v1":
                    raise CorpusError("数据库格式未知，拒绝覆盖")
                if meta.get("source_sha256") == digest:
                    self._check(connection)
                    return self._stats(connection)
            with connection:
                connection.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
                for table in ("compatibility_rules", "template_slots", "templates", "fragment_tags", "fragments", "tags", "categories", "metadata"):
                    connection.execute(f"DELETE FROM {table}")
                connection.executemany("INSERT INTO categories VALUES (?,?,?)", ((c["id"], c["label"], c.get("description", "")) for c in data["categories"]))
                all_tags = {tag for f in data["fragments"] for tag in f["tags"]}
                all_tags.update(tag for t in data["templates"] for s in t["segments"] if isinstance(s, dict) for key in RULE_KEYS for tag in s.get(key, []))
                connection.executemany("INSERT INTO tags VALUES (?)", ((t,) for t in sorted(all_tags)))
                connection.executemany("INSERT INTO fragments VALUES (?,?,?,?)", ((f["id"], f["category"], f["text"], float(f.get("weight", 1))) for f in data["fragments"]))
                connection.executemany("INSERT INTO fragment_tags VALUES (?,?)", ((f["id"], tag) for f in data["fragments"] for tag in f["tags"]))
                for template in data["templates"]:
                    connection.execute("INSERT INTO templates VALUES (?,?,?,?)", (template["id"], template["name"], template["theme"], float(template.get("weight", 1))))
                    for position, segment in enumerate(template["segments"]):
                        if isinstance(segment, str):
                            connection.execute("INSERT INTO template_slots(template_id,position,kind,literal) VALUES (?,?,'literal',?)", (template["id"], position, segment))
                        else:
                            connection.execute("INSERT INTO template_slots VALUES (?,?,'slot',?,NULL,?,?,?,?)", (template["id"], position, segment["slot"], segment["category"], segment.get("optional"), segment.get("prefix", ""), segment.get("suffix", "")))
                            for key, mode in RULE_KEYS.items():
                                connection.executemany("INSERT INTO compatibility_rules VALUES (?,?,?,?)", ((template["id"], position, tag, mode) for tag in segment.get(key, [])))
                connection.executemany("INSERT INTO metadata VALUES (?,?)", [("format", "trash-talk-corpus-v1"), ("source_sha256", digest), ("schema_version", "1"), ("sqlite_import_version", apsw.sqlitelibversion())])
                self._check(connection)
            return self._stats(connection)
        finally:
            connection.close()

    @staticmethod
    def _check(connection: apsw.Connection) -> None:
        if list(connection.execute("PRAGMA quick_check")) != [("ok",)]:
            raise CorpusError("SQLite 完整性检查失败")
        if list(connection.execute("PRAGMA foreign_key_check")):
            raise CorpusError("SQLite 外键检查失败")

    @staticmethod
    def _stats(connection: apsw.Connection) -> dict:
        return {
            "sqlite_version": connection.execute("SELECT sqlite_version()").get,
            "source_sha256": connection.execute("SELECT value FROM metadata WHERE key='source_sha256'").get,
            "categories": connection.execute("SELECT count(*) FROM categories").get,
            "fragments": connection.execute("SELECT count(*) FROM fragments").get,
            "templates": connection.execute("SELECT count(*) FROM templates").get,
            "rules": connection.execute("SELECT count(*) FROM compatibility_rules").get,
        }

    def snapshot(self) -> tuple[list[dict], list[dict], dict]:
        connection = self.connect(readonly=True)
        try:
            tag_map: dict[str, list[str]] = {}
            for fragment_id, tag in connection.execute("SELECT fragment_id,tag_id FROM fragment_tags ORDER BY tag_id"):
                tag_map.setdefault(fragment_id, []).append(tag)
            fragments = [{"id": i, "category": c, "text": t, "weight": w, "tags": tag_map.get(i, [])} for i, c, t, w in connection.execute("SELECT id,category_id,text,weight FROM fragments ORDER BY id")]
            templates = []
            for identifier, name, theme, weight in connection.execute("SELECT id,name,theme,weight FROM templates ORDER BY id"):
                segments = []
                for position, kind, slot, literal, category, probability, prefix, suffix in connection.execute("SELECT position,kind,slot_name,literal,category_id,optional_probability,prefix,suffix FROM template_slots WHERE template_id=? ORDER BY position", (identifier,)):
                    if kind == "literal":
                        segments.append(literal)
                        continue
                    segment = {"slot": slot, "category": category, "prefix": prefix, "suffix": suffix}
                    if probability is not None:
                        segment["optional"] = probability
                    for tag, mode in connection.execute("SELECT tag_id,mode FROM compatibility_rules WHERE template_id=? AND position=? ORDER BY tag_id", (identifier, position)):
                        key = {"all": "tags_all", "any": "tags_any", "exclude": "exclude_tags"}[mode]
                        segment.setdefault(key, []).append(tag)
                    segments.append(segment)
                templates.append({"id": identifier, "name": name, "theme": theme, "weight": weight, "segments": segments})
            return fragments, templates, self._stats(connection)
        finally:
            connection.close()
