"""Persistence, validation, and queue tests for sentence corrections."""

from __future__ import annotations

import json
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import unittest

import apsw

from trash_talk.feedback import FeedbackStore, text_digest


class PreferenceStoreTests(unittest.TestCase):
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

    def context(self, *, prefix: str = "请评价：") -> dict:
        return {
            "prompt_ids": [11, 12, 13],
            "neutral_prompt_ids": [21, 22],
            "prefix": prefix,
            "reference": {"checkpoint": "checkpoints/base/model.safetensors"},
            "theme": "daily",
        }

    def add_original(self, text: str = "请评价：原来的句子", *, context: dict | None = None) -> str:
        return self.store.add_sample(
            text,
            "garbage-1",
            "测试主题",
            17,
            "source-run-17",
            context=self.context() if context is None else context,
        )

    def test_additive_migration_keeps_legacy_samples_ratings_and_job_schema(self) -> None:
        sample_id = self.store.add_sample("旧库里的句子", "rule", "旧主题", 4, "old-run")
        self.store.rate(sample_id, 1)
        before_training_sql = self.query(
            "SELECT sql FROM sqlite_schema WHERE type='table' AND name='training_jobs'"
        )[0][0]
        connection = self.store.connect()
        try:
            connection.execute("DROP TABLE preference_jobs")
            connection.execute("DROP TABLE generation_contexts")
        finally:
            connection.close()

        reopened = FeedbackStore(self.path)
        self.store = reopened

        self.assertEqual(reopened.positive_texts(), ["旧库里的句子"])
        self.assertEqual(
            self.query("SELECT text,source,theme,model_step,model_run_id FROM samples WHERE id=?", (sample_id,)),
            [("旧库里的句子", "rule", "旧主题", 4, "old-run")],
        )
        self.assertEqual(
            self.query("SELECT name FROM sqlite_schema WHERE type='table' AND name IN ('generation_contexts','preference_jobs') ORDER BY name"),
            [("generation_contexts",), ("preference_jobs",)],
        )
        self.assertEqual(
            self.query("SELECT sql FROM sqlite_schema WHERE type='table' AND name='training_jobs'")[0][0],
            before_training_sql,
        )

    def test_add_sample_saves_optional_json_context_without_changing_return_value(self) -> None:
        context = self.context()
        sample_id = self.store.add_sample("存有上下文的样本", "garbage-1", "日常", 1, "run", context=context)

        self.assertIsInstance(sample_id, str)
        self.assertEqual(json.loads(self.query("SELECT context_json FROM generation_contexts WHERE sample_id=?", (sample_id,))[0][0]), context)
        without_context = self.store.add_sample("没有上下文的样本", "rule", "日常", 1)
        self.assertEqual(self.query("SELECT count(*) FROM generation_contexts WHERE sample_id=?", (without_context,)), [(0,)])

    def test_prefix_pair_strips_prefix_and_keeps_original_context_and_source_text(self) -> None:
        original_text = "请评价：原来的句子"
        context = self.context()
        sample_id = self.add_original(original_text, context=context)

        corrected = self.store.correct(sample_id, "  请评价：更好的改写  ")
        claimed = self.store.claim_preference()

        self.assertEqual(corrected["sample_id"], sample_id)
        self.assertFalse(corrected["duplicate"])
        self.assertTrue(corrected["queued"])
        self.assertEqual(claimed["prompt_mode"], "prefix")
        self.assertEqual(claimed["prompt_ids"], [11, 12, 13])
        self.assertEqual(claimed["chosen_completion"], "更好的改写")
        self.assertEqual(claimed["rejected_completion"], "原来的句子")
        self.assertEqual(claimed["original_text"], original_text)
        self.assertEqual(claimed["chosen_text"], "请评价：更好的改写")
        self.assertEqual(claimed["context"], context)
        self.assertEqual(
            self.query("SELECT text FROM samples WHERE id=?", (sample_id,)),
            [(original_text,)],
        )

    def test_prefix_change_uses_theme_only_prompt_and_full_sentences(self) -> None:
        sample_id = self.add_original()

        self.store.correct(sample_id, "改写句子不再带前缀")
        claimed = self.store.claim_preference()

        self.assertEqual(claimed["prompt_mode"], "theme_only")
        self.assertEqual(claimed["prompt_ids"], [21, 22])
        self.assertEqual(claimed["chosen_completion"], "改写句子不再带前缀")
        self.assertEqual(claimed["rejected_completion"], "请评价：原来的句子")

    def test_duplicate_normalized_rewrite_is_idempotent(self) -> None:
        sample_id = self.add_original()
        composed = "改写 café 句子"
        decomposed = unicodedata.normalize("NFD", composed)

        first = self.store.correct(sample_id, composed)
        second = self.store.correct(sample_id, decomposed)

        self.assertFalse(first["duplicate"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(second["preference_id"], first["preference_id"])
        self.assertEqual(second["chosen_sample_id"], first["chosen_sample_id"])
        self.assertEqual(self.query("SELECT count(*) FROM preference_jobs"), [(1,)])
        self.assertEqual(self.query("SELECT count(*) FROM samples"), [(2,)])
        self.assertEqual(self.query("SELECT text FROM samples WHERE id=?", (sample_id,)), [("请评价：原来的句子",)])

    def test_ratings_and_metadata_are_updated_without_training_jobs(self) -> None:
        sample_id = self.add_original()
        self.store.rate(sample_id, 1)
        training_jobs_before = self.query("SELECT count(*) FROM training_jobs")[0][0]

        corrected = self.store.correct(sample_id, "请评价：这是全新的改写句子")
        original_digest = text_digest("请评价：原来的句子")
        chosen_digest = text_digest("请评价：这是全新的改写句子")

        self.assertEqual(self.store.positive_texts(), ["请评价：这是全新的改写句子"])
        self.assertEqual(self.store.negative_hashes(), {original_digest})
        self.assertEqual(self.query("SELECT sample_id,rating FROM ratings WHERE content_sha256=?", (original_digest,)), [(sample_id, -1)])
        self.assertEqual(
            self.query("SELECT sample_id,rating FROM ratings WHERE content_sha256=?", (chosen_digest,)),
            [(corrected["chosen_sample_id"], 1)],
        )
        self.assertEqual(
            self.query("SELECT source,theme,model_step,model_run_id FROM samples WHERE id=?", (corrected["chosen_sample_id"],)),
            [("garbage-1", "测试主题", 17, "source-run-17")],
        )
        self.assertEqual(self.query("SELECT count(*) FROM training_jobs"), [(training_jobs_before,)])
        audit = json.loads(self.query("SELECT previous_ratings_json FROM preference_jobs WHERE id=?", (corrected["preference_id"],))[0][0])
        self.assertEqual(audit["previous_rejected_rating"], 1)
        self.assertIsNone(audit["previous_chosen_rating"])

    def test_validation_requires_saved_context_and_preserves_database(self) -> None:
        no_context_id = self.store.add_sample("没有上下文的原句", "rule", "主题", 0)
        with self.assertRaisesRegex(ValueError, "重新生成"):
            self.store.correct(no_context_id, "新的改写句子")

        context_id = self.add_original()
        invalid_texts = (None, "", "甲", " x\ny ", "字" * 801)
        for invalid in invalid_texts:
            with self.subTest(text=invalid), self.assertRaises(ValueError):
                self.store.correct(context_id, invalid)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "相同"):
            self.store.correct(context_id, "  请评价：原来的句子  ")

        self.assertEqual(self.query("SELECT count(*) FROM preference_jobs"), [(0,)])
        self.assertEqual(self.query("SELECT count(*) FROM samples"), [(2,)])
        self.assertEqual(self.query("SELECT count(*) FROM ratings"), [(0,)])

    def test_queue_claim_recovery_finish_and_error_counts_are_durable(self) -> None:
        first_sample = self.add_original()
        first = self.store.correct(first_sample, "请评价：完成的改写句子")
        second_sample = self.add_original("请评价：第二个原句")
        second = self.store.correct(second_sample, "请评价：失败的改写句子")

        claimed = self.store.claim_preference()
        self.assertEqual(claimed["id"], first["preference_id"])
        self.assertEqual(self.store.preference_counts(), {"pending": 1, "running": 1, "done": 0, "error": 0})
        self.assertEqual(self.store.recover_preferences(), 1)
        self.assertEqual(self.store.preference_counts(), {"pending": 2, "running": 0, "done": 0, "error": 0})

        claimed_again = self.store.claim_preference()
        self.assertEqual(claimed_again["id"], first["preference_id"])
        self.assertEqual(self.query("SELECT attempts FROM preference_jobs WHERE id=?", (first["preference_id"],)), [(2,)])
        result = {"metrics": {"loss": 0.25}, "run_id": "pref-run", "step": 19, "weight_sha256": "abc123"}
        self.store.finish_preference(first["preference_id"], result)
        self.assertEqual(json.loads(self.query("SELECT result_json FROM preference_jobs WHERE id=?", (first["preference_id"],))[0][0]), result)
        with self.assertRaisesRegex(ValueError, "正在运行"):
            self.store.finish_preference(first["preference_id"], {})

        self.assertEqual(self.store.claim_preference()["id"], second["preference_id"])
        self.store.finish_preference(second["preference_id"], {"run_id": "pref-run-2"}, error="backend stopped")
        self.assertEqual(
            self.query("SELECT status,error FROM preference_jobs WHERE id=?", (second["preference_id"],)),
            [("error", "backend stopped")],
        )
        self.assertEqual(self.store.preference_counts(), {"pending": 0, "running": 0, "done": 1, "error": 1})

    def test_correct_requeues_failed_pair_in_place_and_reports_each_status(self) -> None:
        sample_id = self.add_original()
        chosen_text = "请评价：失败后可以重试的改写"
        created = self.store.correct(sample_id, chosen_text)
        self.assertEqual(created["training_status"], "pending")

        pending_repeat = self.store.correct(sample_id, chosen_text)
        self.assertTrue(pending_repeat["duplicate"])
        self.assertEqual(pending_repeat["training_status"], "pending")
        self.assertEqual(self.store.claim_preference()["id"], created["preference_id"])

        running_repeat = self.store.correct(sample_id, chosen_text)
        self.assertTrue(running_repeat["duplicate"])
        self.assertEqual(running_repeat["training_status"], "running")
        self.store.finish_preference(created["preference_id"], {"partial_metric": 0.4}, error="temporary failure")

        counts_before_retry = {
            "samples": self.query("SELECT count(*) FROM samples")[0][0],
            "ratings": self.query("SELECT content_sha256,sample_id,rating FROM ratings ORDER BY content_sha256"),
            "jobs": self.query("SELECT count(*) FROM training_jobs")[0][0],
        }
        retried = self.store.correct(sample_id, chosen_text)

        self.assertTrue(retried["duplicate"])
        self.assertTrue(retried["queued"])
        self.assertEqual(retried["training_status"], "pending")
        self.assertEqual(retried["preference_id"], created["preference_id"])
        self.assertEqual(retried["chosen_sample_id"], created["chosen_sample_id"])
        self.assertEqual(
            self.query("SELECT status,error FROM preference_jobs WHERE id=?", (created["preference_id"],)),
            [("pending", None)],
        )
        self.assertEqual(self.store.preference_counts(), {"pending": 1, "running": 0, "done": 0, "error": 0})
        self.assertEqual(self.query("SELECT count(*) FROM samples"), [(counts_before_retry["samples"],)])
        self.assertEqual(
            self.query("SELECT content_sha256,sample_id,rating FROM ratings ORDER BY content_sha256"),
            counts_before_retry["ratings"],
        )
        self.assertEqual(self.query("SELECT count(*) FROM training_jobs"), [(counts_before_retry["jobs"],)])

        self.assertEqual(self.store.claim_preference()["id"], created["preference_id"])
        self.store.finish_preference(created["preference_id"], {"run_id": "retry-success"})
        done_repeat = self.store.correct(sample_id, chosen_text)
        self.assertTrue(done_repeat["duplicate"])
        self.assertFalse(done_repeat["queued"])
        self.assertEqual(done_repeat["training_status"], "done")
        self.assertEqual(self.query("SELECT count(*) FROM preference_jobs"), [(1,)])

    def test_concurrent_claims_cannot_take_the_same_preference_twice(self) -> None:
        sample_id = self.add_original()
        corrected = self.store.correct(sample_id, "请评价：并发领取的改写")

        with ThreadPoolExecutor(max_workers=2) as executor:
            claims = list(executor.map(lambda _: self.store.claim_preference(), range(2)))

        self.assertEqual(sum(claim is not None for claim in claims), 1)
        self.assertEqual(
            [claim["id"] for claim in claims if claim is not None],
            [corrected["preference_id"]],
        )
        self.assertEqual(self.query("SELECT status,attempts FROM preference_jobs"), [("running", 1)])

    def test_failed_preference_insert_rolls_back_sample_ratings_and_queue(self) -> None:
        sample_id = self.add_original()
        self.store.rate(sample_id, 1)
        before = {
            "samples": self.query("SELECT count(*) FROM samples")[0][0],
            "ratings": self.query("SELECT content_sha256,sample_id,rating FROM ratings ORDER BY content_sha256"),
            "jobs": self.query("SELECT count(*) FROM training_jobs")[0][0],
        }
        connection = self.store.connect()
        try:
            connection.execute(
                "CREATE TRIGGER fail_preference_insert BEFORE INSERT ON preference_jobs "
                "BEGIN SELECT RAISE(ABORT, 'forced preference failure'); END"
            )
        finally:
            connection.close()

        with self.assertRaises(apsw.ConstraintError):
            self.store.correct(sample_id, "请评价：触发回滚的改写")

        self.assertEqual(self.query("SELECT count(*) FROM samples"), [(before["samples"],)])
        self.assertEqual(self.query("SELECT content_sha256,sample_id,rating FROM ratings ORDER BY content_sha256"), before["ratings"])
        self.assertEqual(self.query("SELECT count(*) FROM preference_jobs"), [(0,)])
        self.assertEqual(self.query("SELECT count(*) FROM training_jobs"), [(before["jobs"],)])


if __name__ == "__main__":
    unittest.main()
