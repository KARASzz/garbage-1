"""Data-integrity tests for the separate human-feedback store."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
import unittest

import apsw

from trash_talk.feedback import FeedbackStore, text_digest


class FeedbackStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.feedback_path = self.root / "feedback.sqlite3"
        self.store = FeedbackStore(self.feedback_path)

    def query(self, sql: str, bindings: tuple = ()) -> list[tuple]:
        connection = self.store.connect()
        try:
            return list(connection.execute(sql, bindings))
        finally:
            connection.close()

    def event_count(self) -> int:
        return self.query("SELECT count(*) FROM feedback_events")[0][0]

    def test_new_sample_is_pending_until_rated_and_downvote_removes_it_from_positive_pool(self) -> None:
        text = "你这句样本"
        sample_id = self.store.add_sample(text, "rule", "测试", 0, "draw-run-1")

        initial = self.store.counts()
        self.assertEqual(initial["pending"], 1)
        self.assertEqual(initial["up"], 0)
        self.assertEqual(initial["down"], 0)
        self.assertEqual(self.store.positive_texts(), [])
        self.assertEqual(self.store.negative_hashes(), set())

        self.store.rate(sample_id, 1)
        self.assertEqual(self.store.counts()["pending"], 0)
        self.assertEqual(self.store.positive_texts(), [text])
        self.assertEqual(self.store.negative_hashes(), set())

        self.store.rate(sample_id, -1)
        self.assertEqual(self.store.positive_texts(), [])
        self.assertEqual(self.store.negative_hashes(), {text_digest(text)})
        self.assertEqual(self.store.counts()["down"], 1)

    def test_duplicate_text_uses_one_rating_and_one_positive_training_example(self) -> None:
        text = "内容相同的重复样本"
        first_id = self.store.add_sample(text, "rule", "甲", 0, "draw-run-a")
        second_id = self.store.add_sample(text, "garbage-1", "乙", 4, "draw-run-b")

        self.assertEqual(self.store.counts()["pending"], 1)
        self.assertEqual(self.store.counts()["total_samples"], 2)
        self.store.rate(first_id, 1)
        repeated = self.store.rate(second_id, 1)

        self.assertFalse(repeated["changed"])
        self.assertEqual(self.store.positive_texts(), [text])
        self.assertEqual(self.query("SELECT count(*) FROM ratings")[0][0], 1)
        self.assertEqual(self.event_count(), 1)

    def test_repeated_rating_is_idempotent_and_changed_rating_can_be_reversed(self) -> None:
        sample_id = self.store.add_sample("可以改评价", "rule", "测试", 0)

        self.assertTrue(self.store.rate(sample_id, 1)["changed"])
        self.assertFalse(self.store.rate(sample_id, 1)["changed"])
        self.assertEqual(self.event_count(), 1)

        self.assertTrue(self.store.rate(sample_id, -1)["changed"])
        self.assertEqual(self.store.positive_texts(), [])
        self.assertTrue(self.store.rate(sample_id, 1)["changed"])
        self.assertEqual(self.store.positive_texts(), ["可以改评价"])
        self.assertEqual(self.event_count(), 3)
        self.assertEqual(
            self.query("SELECT previous_rating, rating FROM feedback_events ORDER BY id"),
            [(None, 1), (1, -1), (-1, 1)],
        )

    def test_unknown_sample_and_invalid_ratings_leave_annotation_data_unchanged(self) -> None:
        sample_id = self.store.add_sample("真实样本", "rule", "测试", 0)
        before = (
            self.query("SELECT count(*) FROM samples")[0][0],
            self.query("SELECT count(*) FROM ratings")[0][0],
            self.event_count(),
        )

        with self.assertRaises(LookupError):
            self.store.rate("missing-sample-id", 1)
        for invalid in (True, False, 0, 2, -2, "1", None):
            with self.subTest(rating=invalid), self.assertRaises(ValueError):
                self.store.rate(sample_id, invalid)  # type: ignore[arg-type]

        after = (
            self.query("SELECT count(*) FROM samples")[0][0],
            self.query("SELECT count(*) FROM ratings")[0][0],
            self.event_count(),
        )
        self.assertEqual(after, before)

    def test_reopen_keeps_samples_and_ratings_when_separate_corpus_db_changes(self) -> None:
        corpus_path = self.root / "corpus.sqlite3"
        corpus = sqlite3.connect(corpus_path)
        try:
            corpus.execute("CREATE TABLE fragments(text TEXT NOT NULL)")
            corpus.execute("INSERT INTO fragments VALUES ('old corpus text')")
            corpus.commit()
        finally:
            corpus.close()

        sample_id = self.store.add_sample("人工标注应保留", "rule", "测试", 2, "persistent-run")
        self.store.rate(sample_id, 1)

        corpus = sqlite3.connect(corpus_path)
        try:
            corpus.execute("DELETE FROM fragments")
            corpus.execute("INSERT INTO fragments VALUES ('replacement corpus text')")
            corpus.commit()
        finally:
            corpus.close()

        reopened = FeedbackStore(self.feedback_path)
        self.assertEqual(reopened.positive_texts(), ["人工标注应保留"])
        self.assertEqual(reopened.counts()["up"], 1)
        connection = reopened.connect()
        try:
            self.assertEqual(
                connection.execute("SELECT id, model_run_id FROM samples WHERE id=?", (sample_id,)).get,
                (sample_id, "persistent-run"),
            )
        finally:
            connection.close()

    def test_training_event_keeps_run_id_and_untrained_result_has_no_fake_examples(self) -> None:
        sample_id = self.store.add_sample("候选样本", "rule", "测试", 7, "sample-run-7")
        self.store.rate(sample_id, 1)

        self.store.record_training(
            sample_id,
            {
                "step_before": 7,
                "step_after": 7,
                "run_id": "training-run-8",
                "trained": False,
                "samples": 0,
                "error": "no trainable examples",
            },
        )

        self.assertEqual(self.store.positive_texts(), ["候选样本"])
        self.assertEqual(self.store.counts()["up"], 1)
        self.assertEqual(
            self.query("SELECT model_run_id, model_step FROM samples WHERE id=?", (sample_id,))[0],
            ("sample-run-7", 7),
        )
        training = self.query(
            "SELECT model_run_id, step_before, step_after, trained, examples "
            "FROM training_events WHERE sample_id=?",
            (sample_id,),
        )[0]
        self.assertEqual(training, ("training-run-8", 7, 7, 0, 0))

    def test_foreign_keys_are_enabled_and_failed_rating_transaction_rolls_back(self) -> None:
        sample_id = self.store.add_sample("触发回滚", "rule", "测试", 0)
        connection = self.store.connect()
        try:
            self.assertEqual(connection.execute("PRAGMA foreign_keys").get, 1)
            self.assertTrue(list(connection.execute("PRAGMA foreign_key_list('feedback_events')")))
            self.assertTrue(list(connection.execute("PRAGMA foreign_key_list('ratings')")))
            self.assertTrue(list(connection.execute("PRAGMA foreign_key_list('training_events')")))
            with self.assertRaises(apsw.ConstraintError):
                connection.execute(
                    "INSERT INTO feedback_events(sample_id, previous_rating, rating) VALUES ('missing-id', NULL, 1)"
                )
            with self.assertRaises(apsw.ConstraintError):
                connection.execute(
                    "INSERT INTO ratings(content_sha256, sample_id, rating) VALUES ('missing-hash', 'missing-id', 1)"
                )
            with self.assertRaises(apsw.ConstraintError):
                connection.execute(
                    "INSERT INTO training_events(sample_id, step_before, step_after, model_run_id, trained, examples) "
                    "VALUES ('missing-id', 0, 0, 'run', 0, 0)"
                )
            connection.execute(
                "CREATE TRIGGER fail_feedback_event BEFORE INSERT ON feedback_events "
                "BEGIN SELECT RAISE(ABORT, 'forced event failure'); END"
            )
        finally:
            connection.close()

        with self.assertRaises(apsw.ConstraintError):
            self.store.rate(sample_id, 1)
        self.assertEqual(self.query("SELECT count(*) FROM ratings")[0][0], 0)
        self.assertEqual(self.event_count(), 0)


if __name__ == "__main__":
    unittest.main()
