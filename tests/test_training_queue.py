"""Persistence and transaction tests for the silent background training queue."""

from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import unittest

import apsw

from trash_talk.feedback import FeedbackStore


class TrainingQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "feedback.sqlite3"
        self.store = FeedbackStore(self.path)

    def query(self, sql: str, bindings: tuple = ()) -> list[tuple]:
        connection = self.store.connect()
        try:
            return list(connection.execute(sql, bindings))
        finally:
            connection.close()

    def test_changed_ratings_enqueue_once_per_event_in_the_same_transaction(self) -> None:
        sample_id = self.store.add_sample("静默队列样本", "rule", "测试主题", 0)

        first = self.store.rate(sample_id, 1)
        repeated = self.store.rate(sample_id, 1)
        reversed_vote = self.store.rate(sample_id, -1)

        self.assertTrue(first["changed"])
        self.assertEqual(first["job_id"], 1)
        self.assertFalse(repeated["changed"])
        self.assertIsNone(repeated["job_id"])
        self.assertEqual(reversed_vote["job_id"], 2)
        self.assertEqual(first["message"], "评价已保存。")
        self.assertNotIn("下次", first["message"])
        self.assertEqual(
            self.query("SELECT event_id,action,status FROM training_jobs ORDER BY id"),
            [(1, "positive", "pending"), (2, "exclude", "pending")],
        )
        self.assertEqual(self.store.queue_counts(), {"pending": 2, "running": 0, "done": 0, "error": 0})

    def test_claim_reads_latest_content_rating_instead_of_stale_event_rating(self) -> None:
        first_id = self.store.add_sample("重复内容", "rule", "甲主题", 0)
        second_id = self.store.add_sample("重复内容", "garbage-1", "乙主题", 2)
        self.store.rate(first_id, 1)
        self.store.rate(second_id, -1)

        claimed = self.store.claim_job()

        self.assertEqual(
            claimed,
            {
                "id": 1,
                "sample_id": first_id,
                "action": "positive",
                "text": "重复内容",
                "theme": "甲主题",
                "current_rating": -1,
                "event_id": 1,
            },
        )
        self.assertEqual(self.store.queue_counts(), {"pending": 1, "running": 1, "done": 0, "error": 0})
        self.assertEqual(self.query("SELECT attempts FROM training_jobs WHERE id=1"), [(1,)])

    def test_concurrent_claims_cannot_take_the_same_pending_job_twice(self) -> None:
        sample_id = self.store.add_sample("并发领取样本", "rule", "测试", 0)
        self.store.rate(sample_id, 1)

        with ThreadPoolExecutor(max_workers=2) as executor:
            claims = list(executor.map(lambda _: self.store.claim_job(), range(2)))

        self.assertEqual(sum(claim is not None for claim in claims), 1)
        self.assertEqual(self.store.queue_counts(), {"pending": 0, "running": 1, "done": 0, "error": 0})
        self.assertEqual(self.query("SELECT attempts FROM training_jobs"), [(1,)])

    def test_recovery_requeues_interrupted_work_and_finish_persists_result(self) -> None:
        sample_id = self.store.add_sample("需要训练", "rule", "轻松", 12)
        job_id = self.store.rate(sample_id, 1)["job_id"]

        self.assertEqual(self.store.claim_job()["id"], job_id)
        self.assertEqual(self.store.recover_jobs(), 1)
        self.assertEqual(self.store.queue_counts(), {"pending": 1, "running": 0, "done": 0, "error": 0})
        self.assertEqual(self.store.claim_job()["id"], job_id)
        self.store.finish_job(
            job_id,
            {
                "run_id": "worker-run-9",
                "step_before": 12,
                "step_after": 15,
                "trained": True,
                "samples": 8,
                "loss": 0.125,
                "weight_sha256": "a" * 64,
            },
        )

        self.assertEqual(self.store.queue_counts(), {"pending": 0, "running": 0, "done": 1, "error": 0})
        self.assertEqual(
            self.query("SELECT model_run_id,step_before,step_after,trained,examples,loss,weight_sha256,error "
                       "FROM background_training_events WHERE job_id=?", (job_id,)),
            [("worker-run-9", 12, 15, 1, 8, 0.125, "a" * 64, None)],
        )
        self.assertEqual(self.query("SELECT attempts,status,error FROM training_jobs WHERE id=?", (job_id,)), [(2, "done", None)])

    def test_failed_finish_is_recorded_as_error_and_cannot_be_completed_twice(self) -> None:
        sample_id = self.store.add_sample("需要重试", "rule", "测试", 3)
        job_id = self.store.rate(sample_id, -1)["job_id"]
        self.store.claim_job()

        self.store.finish_job(job_id, {"run_id": "failed-run", "step_before": 3}, error="worker unavailable")

        self.assertEqual(self.store.queue_counts(), {"pending": 0, "running": 0, "done": 0, "error": 1})
        self.assertEqual(self.query("SELECT status,error FROM training_jobs WHERE id=?", (job_id,)), [("error", "worker unavailable")])
        self.assertEqual(self.query("SELECT model_run_id,error FROM background_training_events WHERE job_id=?", (job_id,)), [("failed-run", "worker unavailable")])
        with self.assertRaises(ValueError):
            self.store.finish_job(job_id, {})

    def test_positive_examples_include_theme_and_legacy_text_api_is_preserved(self) -> None:
        sample_id = self.store.add_sample("正向例句", "rule", "职场", 0)
        self.store.rate(sample_id, 1)

        self.assertEqual(self.store.positive_examples(), [{"text": "正向例句", "theme": "职场"}])
        self.assertEqual(self.store.positive_texts(), ["正向例句"])
        self.assertEqual(self.store.positive_examples(0), [])
        with self.assertRaises(ValueError):
            self.store.positive_examples(-1)

    def test_foreign_keys_constraints_and_enqueue_rollback_are_real(self) -> None:
        sample_id = self.store.add_sample("回滚样本", "rule", "测试", 0)
        connection = self.store.connect()
        try:
            self.assertEqual(connection.execute("PRAGMA foreign_keys").get, 1)
            for table in ("training_jobs", "background_training_events"):
                self.assertTrue(list(connection.execute(f"PRAGMA foreign_key_list('{table}')")))

            with self.assertRaises(apsw.ConstraintError):
                connection.execute(
                    "INSERT INTO training_jobs(event_id,sample_id,action) VALUES (999,'missing','positive')"
                )

            action_event = connection.execute(
                "INSERT INTO feedback_events(sample_id,previous_rating,rating) VALUES (?,NULL,1) RETURNING id",
                (sample_id,),
            ).get
            with self.assertRaises(apsw.ConstraintError):
                connection.execute(
                    "INSERT INTO training_jobs(event_id,sample_id,action) VALUES (?,?,'invalid')",
                    (action_event, sample_id),
                )

            status_event = connection.execute(
                "INSERT INTO feedback_events(sample_id,previous_rating,rating) VALUES (?,NULL,-1) RETURNING id",
                (sample_id,),
            ).get
            with self.assertRaises(apsw.ConstraintError):
                connection.execute(
                    "INSERT INTO training_jobs(event_id,sample_id,action,status) VALUES (?,?,'exclude','invalid')",
                    (status_event, sample_id),
                )

            valid_job_event = connection.execute(
                "INSERT INTO feedback_events(sample_id,previous_rating,rating) VALUES (?,NULL,1) RETURNING id",
                (sample_id,),
            ).get
            connection.execute(
                "INSERT INTO training_jobs(event_id,sample_id,action) VALUES (?,?,'positive')",
                (valid_job_event, sample_id),
            )
            with self.assertRaises(apsw.ConstraintError):
                connection.execute(
                    "INSERT INTO training_jobs(event_id,sample_id,action) VALUES (?,?,'exclude')",
                    (valid_job_event, sample_id),
                )
            with self.assertRaises(apsw.ConstraintError):
                connection.execute(
                    "INSERT INTO background_training_events(job_id,step_before,step_after,trained,examples) "
                    "VALUES (999,0,0,0,0)"
                )

            connection.execute(
                "CREATE TRIGGER fail_training_job BEFORE INSERT ON training_jobs "
                "BEGIN SELECT RAISE(ABORT, 'forced job failure'); END"
            )
        finally:
            connection.close()

        with self.assertRaises(apsw.ConstraintError):
            self.store.rate(sample_id, 1)
        self.assertEqual(self.query("SELECT count(*) FROM ratings"), [(0,)])
        self.assertEqual(self.query("SELECT count(*) FROM feedback_events"), [(3,)])
        self.assertEqual(self.query("SELECT count(*) FROM training_jobs"), [(1,)])

    def test_additive_schema_migration_keeps_existing_feedback_data(self) -> None:
        sample_id = self.store.add_sample("旧反馈保留", "rule", "旧主题", 5)
        self.store.rate(sample_id, 1)

        connection = self.store.connect()
        try:
            connection.execute("DROP TABLE background_training_events")
            connection.execute("DROP TABLE training_jobs")
        finally:
            connection.close()

        reopened = FeedbackStore(self.path)

        self.assertEqual(reopened.positive_texts(), ["旧反馈保留"])
        self.assertEqual(reopened.counts()["up"], 1)
        connection = reopened.connect()
        try:
            self.assertEqual(connection.execute("SELECT count(*) FROM feedback_events").get, 1)
        finally:
            connection.close()
        self.assertEqual(reopened.queue_counts(), {"pending": 0, "running": 0, "done": 0, "error": 0})
        self.assertEqual(reopened.rate(sample_id, -1)["job_id"], 1)


if __name__ == "__main__":
    unittest.main()
