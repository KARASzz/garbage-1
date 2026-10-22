"""Human annotations live separately from the replaceable synthetic corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unicodedata
import uuid

import apsw

SCHEMA = """
CREATE TABLE IF NOT EXISTS app_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT;
CREATE TABLE IF NOT EXISTS samples (
    id TEXT PRIMARY KEY,
    content_sha256 TEXT NOT NULL,
    text TEXT NOT NULL CHECK(length(trim(text)) > 0),
    source TEXT NOT NULL CHECK(source IN ('rule','garbage-1')),
    theme TEXT NOT NULL,
    model_step INTEGER NOT NULL CHECK(model_step >= 0),
    model_run_id TEXT NOT NULL DEFAULT 'uninitialized',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
) STRICT;
CREATE INDEX IF NOT EXISTS samples_content ON samples(content_sha256);
CREATE TABLE IF NOT EXISTS corpus_feeds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id TEXT NOT NULL UNIQUE REFERENCES samples(id),
    content_sha256 TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
) STRICT;
CREATE TABLE IF NOT EXISTS ratings (
    content_sha256 TEXT PRIMARY KEY,
    sample_id TEXT NOT NULL REFERENCES samples(id),
    rating INTEGER NOT NULL CHECK(rating IN (-1,1)),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
) STRICT;
CREATE TABLE IF NOT EXISTS feedback_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id TEXT NOT NULL REFERENCES samples(id),
    previous_rating INTEGER CHECK(previous_rating IN (-1,1)),
    rating INTEGER NOT NULL CHECK(rating IN (-1,1)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
) STRICT;
CREATE TABLE IF NOT EXISTS training_events (
    sample_id TEXT PRIMARY KEY REFERENCES samples(id),
    step_before INTEGER NOT NULL,
    step_after INTEGER NOT NULL CHECK(step_after >= step_before),
    model_run_id TEXT NOT NULL DEFAULT 'uninitialized',
    trained INTEGER NOT NULL CHECK(trained IN (0,1)),
    examples INTEGER NOT NULL CHECK(examples >= 0),
    loss REAL,
    weight_sha256 TEXT,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
) STRICT;
CREATE TABLE IF NOT EXISTS training_jobs (
    id INTEGER PRIMARY KEY,
    event_id INTEGER NOT NULL UNIQUE REFERENCES feedback_events(id),
    sample_id TEXT NOT NULL REFERENCES samples(id),
    action TEXT NOT NULL CHECK(action IN ('positive','exclude')),
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','running','done','error')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
    error TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
) STRICT;
CREATE INDEX IF NOT EXISTS training_jobs_status_id ON training_jobs(status, id);
CREATE TABLE IF NOT EXISTS background_training_events (
    job_id INTEGER PRIMARY KEY REFERENCES training_jobs(id),
    model_run_id TEXT NOT NULL DEFAULT 'uninitialized',
    step_before INTEGER NOT NULL CHECK(step_before >= 0),
    step_after INTEGER NOT NULL CHECK(step_after >= step_before),
    trained INTEGER NOT NULL CHECK(trained IN (0,1)),
    examples INTEGER NOT NULL CHECK(examples >= 0),
    loss REAL,
    weight_sha256 TEXT,
    error TEXT
) STRICT;
CREATE TABLE IF NOT EXISTS generation_contexts (
    sample_id TEXT PRIMARY KEY REFERENCES samples(id),
    context_json TEXT NOT NULL
) STRICT;
CREATE TABLE IF NOT EXISTS preference_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id TEXT NOT NULL REFERENCES samples(id),
    chosen_sample_id TEXT NOT NULL REFERENCES samples(id),
    original_text TEXT NOT NULL,
    chosen_text TEXT NOT NULL,
    context_json TEXT NOT NULL,
    prompt_mode TEXT NOT NULL CHECK(prompt_mode IN ('prefix','theme_only')),
    prompt_ids_json TEXT NOT NULL,
    chosen_completion TEXT NOT NULL,
    rejected_completion TEXT NOT NULL,
    previous_ratings_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','running','done','error')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
    result_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE(sample_id, chosen_text)
) STRICT;
CREATE INDEX IF NOT EXISTS preference_jobs_status_id ON preference_jobs(status, id);
"""


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FeedbackStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self.connect()
        try:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_schema WHERE type='table'")}
            if tables and "app_metadata" not in tables:
                raise ValueError("目标文件不是 garbage-1 标注库，拒绝修改")
            if "app_metadata" in tables:
                meta = dict(connection.execute("SELECT key,value FROM app_metadata"))
                if meta.get("format") not in (None, "garbage-1-feedback-v1"):
                    raise ValueError("标注数据库格式未知，保留原数据")
            with connection:
                connection.execute(SCHEMA)
                connection.execute("INSERT OR IGNORE INTO app_metadata VALUES ('format','garbage-1-feedback-v1')")
        finally:
            connection.close()

    def connect(self) -> apsw.Connection:
        connection = apsw.Connection(str(self.path))
        connection.set_busy_timeout(5000)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def add_sample(
        self,
        text: str,
        source: str,
        theme: str,
        model_step: int,
        model_run_id: str = "uninitialized",
        context: dict | None = None,
    ) -> str:
        if not isinstance(text, str) or not text.strip() or source not in ("rule", "garbage-1"):
            raise ValueError("样本文本或来源无效")
        context_json = None
        if context is not None:
            if not isinstance(context, dict):
                raise ValueError("生成上下文必须为 JSON 对象")
            try:
                context_json = json.dumps(context, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError("生成上下文无法保存为 JSON") from error
        identifier = uuid.uuid4().hex
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("INSERT INTO samples(id,content_sha256,text,source,theme,model_step,model_run_id) VALUES (?,?,?,?,?,?,?)", (identifier, text_digest(text), text, source, theme, model_step, model_run_id))
                if context_json is not None:
                    connection.execute(
                        "INSERT INTO generation_contexts(sample_id,context_json) VALUES (?,?)",
                        (identifier, context_json),
                    )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            return identifier
        finally:
            connection.close()

    @staticmethod
    def _normalized_sentence(text: str) -> str:
        if not isinstance(text, str):
            raise ValueError("请输入完整改写句子")
        normalized = unicodedata.normalize("NFC", text.strip())
        if len(normalized) < 2:
            raise ValueError("请输入完整句子，不要只提交单个字")
        if len(normalized) > 800:
            raise ValueError("单句最多 800 个字符")
        if any(char in "\r\n\u2028\u2029" for char in normalized):
            raise ValueError("一次提交一整句，请不要换行")
        return normalized

    @staticmethod
    def _context_prompt_ids(context: dict) -> tuple[list[int], list[int], str]:
        prompt_ids = context.get("prompt_ids")
        neutral_prompt_ids = context.get("neutral_prompt_ids")
        prefix = context.get("prefix")
        if (
            not isinstance(prompt_ids, list)
            or any(type(token_id) is not int or token_id < 0 for token_id in prompt_ids)
            or not isinstance(neutral_prompt_ids, list)
            or any(type(token_id) is not int or token_id < 0 for token_id in neutral_prompt_ids)
            or not isinstance(prefix, str)
        ):
            raise ValueError("生成上下文缺少有效的 prompt_ids、neutral_prompt_ids 或 prefix，请重新生成一句")
        return prompt_ids, neutral_prompt_ids, prefix

    def correct(self, sample_id: str, text: str) -> dict:
        """Save a corrected sentence as a chosen sample and queue a preference pair."""
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("样本编号无效")
        chosen_text = self._normalized_sentence(text)

        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                original = connection.execute(
                    "SELECT id,content_sha256,text,source,theme,model_step,model_run_id "
                    "FROM samples WHERE id=?",
                    (sample_id,),
                ).get
                if original is None:
                    raise LookupError("这条样本不存在，请先生成一句")
                original_text = original[2]
                if chosen_text == unicodedata.normalize("NFC", original_text.strip()):
                    raise ValueError("改写内容不能与原句相同")

                context_row = connection.execute(
                    "SELECT context_json FROM generation_contexts WHERE sample_id=?", (sample_id,)
                ).get
                if context_row is None:
                    raise ValueError("这句话没有保存生成上下文，请重新生成一句后再纠正")
                try:
                    context = json.loads(context_row)
                except (TypeError, json.JSONDecodeError) as error:
                    raise ValueError("生成上下文已损坏，请重新生成一句后再纠正") from error
                if not isinstance(context, dict):
                    raise ValueError("生成上下文已损坏，请重新生成一句后再纠正")
                prompt_ids, neutral_prompt_ids, prefix = self._context_prompt_ids(context)

                existing = connection.execute(
                    "SELECT id,status,chosen_sample_id FROM preference_jobs WHERE sample_id=? AND chosen_text=?",
                    (sample_id, chosen_text),
                ).get
                if existing is not None:
                    training_status = existing[1]
                    if training_status == "error":
                        connection.execute(
                            "UPDATE preference_jobs SET status='pending',error=NULL "
                            "WHERE id=? AND status='error'",
                            (existing[0],),
                        )
                        training_status = "pending"
                    counts = self._counts(connection)
                    queued = training_status in ("pending", "running")
                    connection.execute("COMMIT")
                    return {
                        "sample_id": sample_id,
                        "chosen_sample_id": existing[2],
                        "preference_id": existing[0],
                        "duplicate": True,
                        "queued": queued,
                        "training_status": training_status,
                        "counts": counts,
                        "message": "这条改写已经记录。" if not queued else "这条改写已经在偏好队列中。",
                    }

                use_prefix = bool(prefix) and chosen_text.startswith(prefix) and original_text.startswith(prefix)
                prompt_mode = "prefix" if use_prefix else "theme_only"
                pair_prompt_ids = prompt_ids if use_prefix else neutral_prompt_ids
                chosen_completion = chosen_text[len(prefix):] if use_prefix else chosen_text
                rejected_completion = original_text[len(prefix):] if use_prefix else original_text

                chosen_sample_id = uuid.uuid4().hex
                chosen_digest = text_digest(chosen_text)
                original_digest = original[1]
                previous_rejected_rating = connection.execute(
                    "SELECT rating FROM ratings WHERE content_sha256=?", (original_digest,)
                ).get
                previous_chosen_rating = connection.execute(
                    "SELECT rating FROM ratings WHERE content_sha256=?", (chosen_digest,)
                ).get

                connection.execute(
                    "INSERT INTO samples(id,content_sha256,text,source,theme,model_step,model_run_id) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (chosen_sample_id, chosen_digest, chosen_text, original[3], original[4], original[5], original[6]),
                )
                connection.execute(
                    "INSERT INTO ratings(content_sha256,sample_id,rating) VALUES (?,?,1) "
                    "ON CONFLICT(content_sha256) DO UPDATE SET sample_id=excluded.sample_id,"
                    "rating=1,updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')",
                    (chosen_digest, chosen_sample_id),
                )
                connection.execute(
                    "INSERT INTO ratings(content_sha256,sample_id,rating) VALUES (?,?, -1) "
                    "ON CONFLICT(content_sha256) DO UPDATE SET sample_id=excluded.sample_id,"
                    "rating=-1,updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')",
                    (original_digest, sample_id),
                )
                previous_ratings_json = json.dumps(
                    {
                        "rejected_content_sha256": original_digest,
                        "chosen_content_sha256": chosen_digest,
                        "previous_rejected_rating": previous_rejected_rating,
                        "previous_chosen_rating": previous_chosen_rating,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                preference_id = connection.execute(
                    "INSERT INTO preference_jobs "
                    "(sample_id,chosen_sample_id,original_text,chosen_text,context_json,prompt_mode,"
                    "prompt_ids_json,chosen_completion,rejected_completion,previous_ratings_json) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?) RETURNING id",
                    (
                        sample_id,
                        chosen_sample_id,
                        original_text,
                        chosen_text,
                        context_row,
                        prompt_mode,
                        json.dumps(pair_prompt_ids, separators=(",", ":")),
                        chosen_completion,
                        rejected_completion,
                        previous_ratings_json,
                    ),
                ).get
                counts = self._counts(connection)
                connection.execute("COMMIT")
                return {
                    "sample_id": sample_id,
                    "chosen_sample_id": chosen_sample_id,
                    "preference_id": preference_id,
                    "duplicate": False,
                    "queued": True,
                    "training_status": "pending",
                    "counts": counts,
                    "message": "改写已记录，正在等待偏好训练。",
                }
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        finally:
            connection.close()

    def feed_sentence(self, text: str, model_step: int, model_run_id: str) -> dict:
        """Persist a complete user sentence as a positive example and queue it."""
        if not isinstance(text, str):
            raise ValueError("请输入一句中文语料")
        normalized = unicodedata.normalize("NFC", text.strip())
        if not normalized or len(normalized) < 2:
            raise ValueError("请输入完整句子，不要只投喂单个字")
        if len(normalized) > 800:
            raise ValueError("单句最多 800 个字符")
        if any(char in "\r\n\u2028\u2029" for char in normalized):
            raise ValueError("一次投喂一整句，请不要换行")
        if type(model_step) is not int or model_step < 0 or not isinstance(model_run_id, str):
            raise ValueError("模型状态无效，暂时无法投喂")

        digest = text_digest(normalized)
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT sample_id FROM corpus_feeds WHERE content_sha256=?",
                    (digest,),
                ).get
                if existing is not None:
                    counts = self._counts(connection)
                    feed_count = connection.execute("SELECT count(*) FROM corpus_feeds").get
                    connection.execute("COMMIT")
                    return {
                        "sample_id": existing,
                        "duplicate": True,
                        "job_id": None,
                        "counts": counts,
                        "fed_sentences": feed_count,
                        "message": "这句已经投喂过了。",
                    }

                sample_id = uuid.uuid4().hex
                connection.execute(
                    "INSERT INTO samples(id,content_sha256,text,source,theme,model_step,model_run_id) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (sample_id, digest, normalized, "garbage-1", "manual", model_step, model_run_id),
                )
                previous = connection.execute(
                    "SELECT rating FROM ratings WHERE content_sha256=?", (digest,)
                ).get
                connection.execute(
                    "INSERT INTO ratings(content_sha256,sample_id,rating) VALUES (?,?,1) "
                    "ON CONFLICT(content_sha256) DO UPDATE SET sample_id=excluded.sample_id,"
                    "rating=1,updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')",
                    (digest, sample_id),
                )
                event_id = connection.execute(
                    "INSERT INTO feedback_events(sample_id,previous_rating,rating) "
                    "VALUES (?,?,1) RETURNING id",
                    (sample_id, previous),
                ).get
                job_id = connection.execute(
                    "INSERT INTO training_jobs(event_id,sample_id,action) "
                    "VALUES (?,?,'positive') RETURNING id",
                    (event_id, sample_id),
                ).get
                connection.execute(
                    "INSERT INTO corpus_feeds(sample_id,content_sha256) VALUES (?,?)",
                    (sample_id, digest),
                )
                counts = self._counts(connection)
                feed_count = connection.execute("SELECT count(*) FROM corpus_feeds").get
                connection.execute("COMMIT")
                return {
                    "sample_id": sample_id,
                    "duplicate": False,
                    "job_id": job_id,
                    "counts": counts,
                    "fed_sentences": feed_count,
                    "message": "整句已收下，正在后台学习。",
                }
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        finally:
            connection.close()

    def corpus_feed_count(self) -> int:
        connection = self.connect()
        try:
            return connection.execute("SELECT count(*) FROM corpus_feeds").get
        finally:
            connection.close()

    @staticmethod
    def _counts(connection: apsw.Connection) -> dict:
        votes = dict(connection.execute("SELECT rating,count(*) FROM ratings GROUP BY rating"))
        return {"up": votes.get(1, 0), "down": votes.get(-1, 0), "pending": connection.execute("SELECT count(DISTINCT content_sha256) FROM samples WHERE content_sha256 NOT IN (SELECT content_sha256 FROM ratings)").get, "total_samples": connection.execute("SELECT count(*) FROM samples").get}

    def counts(self) -> dict:
        connection = self.connect()
        try:
            return self._counts(connection)
        finally:
            connection.close()

    def rate(self, sample_id: str, rating: int) -> dict:
        if not isinstance(sample_id, str) or type(rating) is not int or rating not in (1, -1):
            raise ValueError("评价必须为点赞 1 或点踩 -1")
        connection = self.connect()
        try:
            with connection:
                job_id = None
                row = connection.execute("SELECT content_sha256 FROM samples WHERE id=?", (sample_id,)).get
                if row is None:
                    raise LookupError("这条样本不存在，请先生成一句")
                digest = row
                previous = connection.execute("SELECT rating FROM ratings WHERE content_sha256=?", (digest,)).get
                changed = previous != rating
                if changed:
                    connection.execute("INSERT INTO ratings(content_sha256,sample_id,rating) VALUES (?,?,?) ON CONFLICT(content_sha256) DO UPDATE SET sample_id=excluded.sample_id,rating=excluded.rating,updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')", (digest, sample_id, rating))
                    event_id = connection.execute(
                        "INSERT INTO feedback_events(sample_id,previous_rating,rating) VALUES (?,?,?) RETURNING id",
                        (sample_id, previous, rating),
                    ).get
                    action = "positive" if rating == 1 else "exclude"
                    job_id = connection.execute(
                        "INSERT INTO training_jobs(event_id,sample_id,action) VALUES (?,?,?) RETURNING id",
                        (event_id, sample_id, action),
                    ).get
                counts = self._counts(connection)
            return {
                "sample_id": sample_id,
                "rating": rating,
                "changed": changed,
                "counts": counts,
                "message": "评价已保存。",
                "job_id": job_id,
            }
        finally:
            connection.close()

    def positive_examples(self, limit: int = 512) -> list[dict[str, str]]:
        if type(limit) is not int or limit < 0:
            raise ValueError("limit 必须为非负整数")
        connection = self.connect()
        try:
            return [
                {"text": row[0], "theme": row[1]}
                for row in connection.execute(
                    "SELECT s.text,s.theme FROM ratings r JOIN samples s ON s.id=r.sample_id "
                    "WHERE r.rating=1 ORDER BY r.updated_at DESC,r.content_sha256 LIMIT ?",
                    (limit,),
                )
            ]
        finally:
            connection.close()

    def positive_texts(self, limit: int = 512) -> list[str]:
        return [example["text"] for example in self.positive_examples(limit)]

    def recover_jobs(self) -> int:
        """Return abandoned running jobs to the pending queue after a worker restart."""
        connection = self.connect()
        try:
            with connection:
                connection.execute("UPDATE training_jobs SET status='pending' WHERE status='running'")
                return connection.changes()
        finally:
            connection.close()

    def claim_job(self) -> dict | None:
        """Atomically claim one pending job and include the latest rating for its content."""
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT j.id,j.sample_id,j.action,s.text,s.theme,r.rating,j.event_id "
                    "FROM training_jobs j "
                    "JOIN samples s ON s.id=j.sample_id "
                    "LEFT JOIN ratings r ON r.content_sha256=s.content_sha256 "
                    "WHERE j.status='pending' ORDER BY j.id LIMIT 1"
                ).get
                if row is None:
                    connection.execute("COMMIT")
                    return None

                connection.execute(
                    "UPDATE training_jobs SET status='running',attempts=attempts+1,error=NULL "
                    "WHERE id=? AND status='pending'",
                    (row[0],),
                )
                connection.execute("COMMIT")
                return {
                    "id": row[0],
                    "sample_id": row[1],
                    "action": row[2],
                    "text": row[3],
                    "theme": row[4],
                    "current_rating": row[5],
                    "event_id": row[6],
                }
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        finally:
            connection.close()

    def finish_job(self, job_id: int, result: dict, error: str | None = None) -> None:
        if type(job_id) is not int or job_id <= 0:
            raise ValueError("job_id 必须为正整数")
        if not isinstance(result, dict):
            raise ValueError("result 必须为字典")
        effective_error = error if error is not None else result.get("error")
        step_before = result.get("step_before", 0)
        step_after = result.get("step_after", step_before)
        trained = bool(result.get("trained", False))
        examples = result.get("samples", 0)

        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                status = connection.execute("SELECT status FROM training_jobs WHERE id=?", (job_id,)).get
                if status is None:
                    raise LookupError("训练任务不存在")
                if status != "running":
                    raise ValueError("只能完成正在运行的训练任务")

                connection.execute(
                    "INSERT INTO background_training_events "
                    "(job_id,model_run_id,step_before,step_after,trained,examples,loss,weight_sha256,error) "
                    "VALUES (?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(job_id) DO UPDATE SET "
                    "model_run_id=excluded.model_run_id,step_before=excluded.step_before,"
                    "step_after=excluded.step_after,trained=excluded.trained,examples=excluded.examples,"
                    "loss=excluded.loss,weight_sha256=excluded.weight_sha256,error=excluded.error",
                    (
                        job_id,
                        result.get("run_id", "uninitialized"),
                        step_before,
                        step_after,
                        int(trained),
                        examples,
                        result.get("loss"),
                        result.get("weight_sha256"),
                        effective_error,
                    ),
                )
                final_status = "error" if effective_error is not None else "done"
                connection.execute(
                    "UPDATE training_jobs SET status=?,error=? WHERE id=? AND status='running'",
                    (final_status, effective_error, job_id),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        finally:
            connection.close()

    def queue_counts(self) -> dict[str, int]:
        connection = self.connect()
        try:
            counts = dict(connection.execute("SELECT status,count(*) FROM training_jobs GROUP BY status"))
            return {status: counts.get(status, 0) for status in ("pending", "running", "done", "error")}
        finally:
            connection.close()

    def recover_preferences(self) -> int:
        """Return abandoned running preferences to the pending queue after a restart."""
        connection = self.connect()
        try:
            with connection:
                connection.execute("UPDATE preference_jobs SET status='pending' WHERE status='running'")
                return connection.changes()
        finally:
            connection.close()

    def claim_preference(self) -> dict | None:
        """Atomically claim one pending pair and persist that it is being processed."""
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT id,sample_id,chosen_sample_id,prompt_ids_json,chosen_completion,"
                    "rejected_completion,context_json,original_text,chosen_text,prompt_mode "
                    "FROM preference_jobs WHERE status='pending' ORDER BY id LIMIT 1"
                ).get
                if row is None:
                    connection.execute("COMMIT")
                    return None

                try:
                    prompt_ids = json.loads(row[3])
                    context = json.loads(row[6])
                except (TypeError, json.JSONDecodeError) as error:
                    raise ValueError("偏好队列中的生成上下文已损坏") from error
                claimed = {
                    "id": row[0],
                    "sample_id": row[1],
                    "chosen_sample_id": row[2],
                    "prompt_ids": prompt_ids,
                    "chosen_completion": row[4],
                    "rejected_completion": row[5],
                    "context": context,
                    "original_text": row[7],
                    "chosen_text": row[8],
                    "prompt_mode": row[9],
                }
                connection.execute(
                    "UPDATE preference_jobs SET status='running',attempts=attempts+1,error=NULL "
                    "WHERE id=? AND status='pending'",
                    (row[0],),
                )
                connection.execute("COMMIT")
                return claimed
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        finally:
            connection.close()

    def finish_preference(self, preference_id: int, result: dict, error: str | None = None) -> None:
        if type(preference_id) is not int or preference_id <= 0:
            raise ValueError("preference_id 必须为正整数")
        if not isinstance(result, dict):
            raise ValueError("result 必须为字典")
        effective_error = error if error is not None else result.get("error")
        if effective_error is not None and not isinstance(effective_error, str):
            raise ValueError("error 必须为字符串或 None")
        try:
            result_json = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("result 无法保存为 JSON") from error

        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                status = connection.execute(
                    "SELECT status FROM preference_jobs WHERE id=?", (preference_id,)
                ).get
                if status is None:
                    raise LookupError("偏好训练任务不存在")
                if status != "running":
                    raise ValueError("只能完成正在运行的偏好训练任务")

                final_status = "error" if effective_error is not None else "done"
                connection.execute(
                    "UPDATE preference_jobs SET status=?,result_json=?,error=? "
                    "WHERE id=? AND status='running'",
                    (final_status, result_json, effective_error, preference_id),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        finally:
            connection.close()

    def preference_counts(self) -> dict[str, int]:
        connection = self.connect()
        try:
            counts = dict(connection.execute("SELECT status,count(*) FROM preference_jobs GROUP BY status"))
            return {status: counts.get(status, 0) for status in ("pending", "running", "done", "error")}
        finally:
            connection.close()

    def negative_hashes(self) -> set[str]:
        connection = self.connect()
        try:
            return {row[0] for row in connection.execute("SELECT content_sha256 FROM ratings WHERE rating=-1")}
        finally:
            connection.close()

    def record_training(self, sample_id: str, result: dict) -> None:
        connection = self.connect()
        try:
            connection.execute("INSERT INTO training_events(sample_id,step_before,step_after,model_run_id,trained,examples,loss,weight_sha256,error) VALUES (?,?,?,?,?,?,?,?,?)", (sample_id, result["step_before"], result["step_after"], result.get("run_id", "uninitialized"), int(result["trained"]), result.get("samples", 0), result.get("loss"), result.get("weight_sha256"), result.get("error")))
        finally:
            connection.close()
