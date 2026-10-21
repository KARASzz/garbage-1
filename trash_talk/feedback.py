"""Human annotations live separately from the replaceable synthetic corpus."""

from __future__ import annotations

import hashlib
from pathlib import Path
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

    def add_sample(self, text: str, source: str, theme: str, model_step: int, model_run_id: str = "uninitialized") -> str:
        if not isinstance(text, str) or not text.strip() or source not in ("rule", "garbage-1"):
            raise ValueError("样本文本或来源无效")
        identifier = uuid.uuid4().hex
        connection = self.connect()
        try:
            connection.execute("INSERT INTO samples(id,content_sha256,text,source,theme,model_step,model_run_id) VALUES (?,?,?,?,?,?,?)", (identifier, text_digest(text), text, source, theme, model_step, model_run_id))
            return identifier
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
