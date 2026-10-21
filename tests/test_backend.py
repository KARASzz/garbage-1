"""Behavioral tests for corpus import, grammar generation, and the local API."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import apsw

from trash_talk import APP_ID
from trash_talk.generator import GrammarGenerator
from trash_talk.server import create_server, project_identity
from trash_talk.storage import CorpusDatabase, CorpusError


def make_corpus() -> dict:
    """Return a small source corpus with valid and adversarial tag combinations."""
    categories = [
        {"id": "subject", "label": "主语", "description": "句子主语"},
        {"id": "verb", "label": "谓语", "description": "动作"},
        {"id": "object", "label": "宾语", "description": "动作对象"},
        {"id": "aside", "label": "补充", "description": "可选补充"},
    ]
    fragments = []
    for index in range(6):
        fragments.extend(
            (
                {"id": f"s{index}", "category": "subject", "text": f"主语{index}", "tags": ["required", "approved"], "weight": 1},
                {"id": f"v{index}", "category": "verb", "text": f"动作{index}", "tags": [], "weight": 1},
                {"id": f"o{index}", "category": "object", "text": f"对象{index}", "tags": [], "weight": 1},
            )
        )
    fragments.extend(
        (
            {"id": "s-missing-all", "category": "subject", "text": "缺少全标签", "tags": ["approved"], "weight": 10000},
            {"id": "s-missing-any", "category": "subject", "text": "缺少任意标签", "tags": ["required", "neutral"], "weight": 10000},
            {"id": "s-excluded", "category": "subject", "text": "带有禁用标签", "tags": ["required", "approved", "forbidden"], "weight": 10000},
            {"id": "aside-0", "category": "aside", "text": "旁边等着", "tags": [], "weight": 1},
        )
    )
    template = {
        "id": "main",
        "name": "三段句型",
        "theme": "测试主题",
        "weight": 1,
        "segments": [
            {
                "slot": "subject",
                "category": "subject",
                "tags_all": ["required"],
                "tags_any": ["approved", "friendly"],
                "exclude_tags": ["forbidden"],
                "suffix": " ",
            },
            {"slot": "verb", "category": "verb", "suffix": " "},
            {"slot": "object", "category": "object", "prefix": "把"},
            {"slot": "aside", "category": "aside", "optional": 0.5, "prefix": "，还"},
        ],
    }
    return {"version": 1, "categories": categories, "fragments": fragments, "templates": [template]}


class BackendTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "data").mkdir()
        self.source_path = self.root / "data" / "corpus.json"
        self.database_path = self.root / "data" / "trash_talk.sqlite3"

    def write_source(self, data: dict, path: Path | None = None) -> Path:
        target = path or self.source_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return target

    def import_corpus(self, data: dict | None = None) -> CorpusDatabase:
        self.write_source(data or make_corpus())
        database = CorpusDatabase(self.database_path)
        database.initialize(self.source_path)
        return database


class GeneratorBehaviorTests(BackendTestCase):
    def test_required_slots_are_always_present_and_optional_slot_can_be_forced(self) -> None:
        database = self.import_corpus()
        generator = GrammarGenerator(database, seed=17)

        without_optional = generator.draw(debug=True, template_id="main", include_optional=False)
        self.assertEqual(without_optional["block_count"], 3)
        self.assertEqual([draw["slot"] for draw in without_optional["draws"]], ["subject", "verb", "object"])
        self.assertIn("主语", without_optional["text"])
        self.assertIn("动作", without_optional["text"])
        self.assertIn("把对象", without_optional["text"])
        self.assertNotIn("旁边等着", without_optional["text"])

        with_optional = generator.draw(debug=True, template_id="main", include_optional=True)
        self.assertEqual(with_optional["block_count"], 4)
        self.assertEqual([draw["slot"] for draw in with_optional["draws"]], ["subject", "verb", "object", "aside"])
        self.assertIn("，还旁边等着", with_optional["text"])

    def test_tag_all_any_and_exclude_rules_do_not_leak_fragments(self) -> None:
        database = self.import_corpus()
        generator = GrammarGenerator(database, seed=241)
        accepted = {f"s{index}" for index in range(6)}

        for _ in range(120):
            result = generator.draw(debug=True, template_id="main", include_optional=False)
            subject = next(draw for draw in result["draws"] if draw["slot"] == "subject")
            self.assertIn(subject["id"], accepted)
            self.assertIn("required", subject["tags"])
            self.assertTrue({"approved", "friendly"} & set(subject["tags"]))
            self.assertNotIn("forbidden", subject["tags"])

    def test_fixed_seed_reproduces_draw_sequence(self) -> None:
        database = self.import_corpus()
        left = GrammarGenerator(database, seed=302)
        right = GrammarGenerator(database, seed=302)

        left_sequence = [left.draw(debug=True, include_optional=None) for _ in range(30)]
        right_sequence = [right.draw(debug=True, include_optional=None) for _ in range(30)]
        self.assertEqual(left_sequence, right_sequence)

    def test_large_fragment_weight_has_a_clear_sampling_effect(self) -> None:
        data = make_corpus()
        data["fragments"] = [f for f in data["fragments"] if f["id"] not in {f"s{index}" for index in range(6)}]
        data["fragments"].extend(
            (
                {"id": "s-heavy", "category": "subject", "text": "高权重主语", "tags": ["required", "approved"], "weight": 1000},
                {"id": "s-light", "category": "subject", "text": "低权重主语", "tags": ["required", "approved"], "weight": 1},
            )
        )
        database = self.import_corpus(data)
        generator = GrammarGenerator(database, seed=409)

        counts = Counter(
            next(draw["id"] for draw in generator.draw(debug=True, template_id="main", include_optional=False)["draws"] if draw["slot"] == "subject")
            for _ in range(400)
        )
        # Recent-use suppression intentionally softens raw weight ratios; this
        # still requires a clear shift from an even 50/50 pool.
        self.assertGreater(counts["s-heavy"] / sum(counts.values()), 0.8, counts)

    def test_large_template_weight_has_a_clear_sampling_effect(self) -> None:
        data = make_corpus()
        data["templates"][0]["weight"] = 1000
        alternate = deepcopy(data["templates"][0])
        alternate.update({"id": "secondary", "name": "備選句型", "theme": "備選", "weight": 1})
        alternate["segments"] = ["另一種：", *alternate["segments"]]
        data["templates"].append(alternate)
        generator = GrammarGenerator(self.import_corpus(data), seed=431)

        counts = Counter(generator.draw(debug=True, include_optional=False)["template_id"] for _ in range(400))
        self.assertGreater(counts["main"] / sum(counts.values()), 0.9, counts)

    def test_recent_sentence_window_avoids_repeating_a_sentence(self) -> None:
        database = self.import_corpus()
        generator = GrammarGenerator(database, seed=521)

        texts = [generator.draw(template_id="main", include_optional=False)["text"] for _ in range(128)]
        self.assertEqual(len(set(texts)), 128)

    def test_unknown_template_is_a_user_facing_corpus_error(self) -> None:
        generator = GrammarGenerator(self.import_corpus(), seed=8)
        with self.assertRaises(CorpusError):
            generator.draw(template_id="does-not-exist")


class StorageBehaviorTests(BackendTestCase):
    def test_import_rejects_a_required_slot_without_candidates(self) -> None:
        data = make_corpus()
        data["templates"][0]["segments"][0]["tags_any"] = ["tag-that-no-fragment-has"]
        self.write_source(data)

        with self.assertRaises(CorpusError):
            CorpusDatabase(self.database_path).initialize(self.source_path)
        self.assertFalse(self.database_path.exists())

    def test_failed_source_update_preserves_the_previous_database(self) -> None:
        database = self.import_corpus()
        before = database.snapshot()
        bad_data = make_corpus()
        # This reaches SQLite's NOT NULL constraint after the importer has
        # started its replacement transaction, exercising rollback protection.
        bad_data["categories"][0]["label"] = None
        bad_source = self.root / "data" / "broken.json"
        self.write_source(bad_data, bad_source)

        with self.assertRaises((CorpusError, apsw.Error)):
            database.initialize(bad_source)
        self.assertEqual(database.snapshot(), before)

    def test_schema_is_strict_foreign_keys_are_enforced_and_engine_is_reported(self) -> None:
        database = self.import_corpus()
        _, _, stats = database.snapshot()
        connection = database.connect()
        try:
            strict_tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM pragma_table_list WHERE schema='main' AND type='table' AND strict=1"
                )
            }
            self.assertTrue(
                {"metadata", "categories", "fragments", "tags", "fragment_tags", "templates", "template_slots", "compatibility_rules"}
                <= strict_tables
            )
            self.assertEqual(connection.execute("PRAGMA foreign_keys").get, 1)
            self.assertTrue(list(connection.execute("PRAGMA foreign_key_list('fragment_tags')")))
            with self.assertRaises(apsw.ConstraintError):
                connection.execute("INSERT INTO fragment_tags(fragment_id, tag_id) VALUES ('missing-fragment', 'approved')")

            actual_sqlite = connection.execute("SELECT sqlite_version()").get
            self.assertEqual(stats["sqlite_version"], actual_sqlite)
            self.assertGreaterEqual(tuple(map(int, actual_sqlite.split("."))), (3, 37, 0))
        finally:
            connection.close()

    def test_same_source_import_returns_same_summary_without_committing_again(self) -> None:
        database = self.import_corpus()
        observer = database.connect(readonly=True)
        try:
            version_before = observer.execute("PRAGMA data_version").get
            summary_before = database.initialize(self.source_path)
            version_after = observer.execute("PRAGMA data_version").get
        finally:
            observer.close()

        self.assertEqual(summary_before, database.snapshot()[2])
        self.assertEqual(version_after, version_before)


class HttpBehaviorTests(BackendTestCase):
    def test_http_api_static_allowlist_and_unknown_host_rejection(self) -> None:
        self.write_source(make_corpus())
        web = self.root / "web"
        web.mkdir()
        (web / "index.html").write_text("<main>test-ui</main>", encoding="utf-8")
        (web / "style.css").write_text("body{}", encoding="utf-8")
        (web / "app.js").write_text("void 0;", encoding="utf-8")
        (web / "favicon.svg").write_text("<svg/>", encoding="utf-8")
        venv_python = self.root / ".venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.write_text("private executable placeholder", encoding="utf-8")

        server = create_server(self.root, port=0, seed=613)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)
        port = server.server_port
        opener = build_opener(ProxyHandler({}))

        def get(path: str) -> tuple[int, bytes, dict[str, str]]:
            request = Request(f"http://127.0.0.1:{port}{path}")
            try:
                with opener.open(request, timeout=2) as response:
                    return response.status, response.read(), dict(response.headers.items())
            except HTTPError as error:
                response = error.code, error.read(), dict(error.headers.items())
                error.close()
                return response

        status, body, _ = get("/api/health")
        health = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(health["app"], APP_ID)
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["fragments"], len(make_corpus()["fragments"]))
        self.assertTrue(health["sqlite_version"])

        status, body, _ = get("/api/draw")
        draw = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(set(draw), {"text", "id", "theme", "block_count"})
        self.assertTrue(draw["text"].endswith("。"))

        status, body, _ = get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"test-ui", body)
        for forbidden_path in (
            "/data/corpus.json",
            "/data/trash_talk.sqlite3",
            "/.venv/bin/python",
            "/../data/corpus.json",
            "/%2e%2e/data/corpus.json",
            "/%2e%2e/.venv/bin/python",
        ):
            with self.subTest(path=forbidden_path):
                self.assertEqual(get(forbidden_path)[0], 404)

        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            connection.request("GET", "/api/health", headers={"Host": "evil.example"})
            response = connection.getresponse()
            self.assertEqual(response.status, 403)
        finally:
            connection.close()


class RunStartupRaceTests(BackendTestCase):
    def test_three_simultaneous_run_main_calls_reuse_one_healthy_server(self) -> None:
        source = make_corpus()
        # Keep the first import busy long enough for all three independently
        # released processes to contend for startup at once.
        source["fragments"].extend(
            {
                "id": f"noise-{index}",
                "category": "subject",
                "text": f"并发测试语料{index}",
                "tags": ["noise"],
                "weight": 1,
            }
            for index in range(1800)
        )
        self.write_source(source)
        web = self.root / "web"
        web.mkdir()
        (web / "index.html").write_text("<main>race-test</main>", encoding="utf-8")
        (web / "style.css").write_text("body{}", encoding="utf-8")
        (web / "app.js").write_text("void 0;", encoding="utf-8")
        (web / "favicon.svg").write_text("<svg/>", encoding="utf-8")

        port = None
        for _ in range(10):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", 0))
                candidate = probe.getsockname()[1]
            if 1024 <= candidate <= 65516:
                port = candidate
                break
        self.assertIsNotNone(port, "could not reserve a suitable temporary loopback port")

        runtime = self.root / ".runtime"
        runtime.mkdir()
        gate = runtime / "release-all-processes"
        child_code = "\n".join(
            (
                "from pathlib import Path",
                "import sys, time",
                "root_arg, port_arg, gate_arg, ready_arg = sys.argv[1:5]",
                "import run",
                "run.ROOT = Path(root_arg)",
                "sys.argv = ['run.py', '--port', port_arg]",
                "Path(ready_arg).touch()",
                "deadline = time.monotonic() + 8",
                "while not Path(gate_arg).exists():",
                "    if time.monotonic() >= deadline:",
                "        raise SystemExit(97)",
                "    time.sleep(0.005)",
                "raise SystemExit(run.main())",
            )
        )
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        processes: list[subprocess.Popen[str]] = []
        results: dict[int, tuple[int, str]] = {}
        try:
            for index in range(3):
                ready = runtime / f"ready-{index}"
                processes.append(
                    subprocess.Popen(
                        [sys.executable, "-c", child_code, str(self.root), str(port), str(gate), str(ready)],
                        cwd=Path(__file__).resolve().parents[1],
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                )

            ready_deadline = time.monotonic() + 3
            ready_files = [runtime / f"ready-{index}" for index in range(3)]
            while time.monotonic() < ready_deadline and not all(path.exists() for path in ready_files):
                time.sleep(0.01)
            self.assertTrue(all(path.exists() for path in ready_files), "all child processes must reach the common start gate")
            gate.touch()

            startup_deadline = time.monotonic() + 7
            while time.monotonic() < startup_deadline:
                for index, process in enumerate(processes):
                    if index not in results and process.poll() is not None:
                        output, _ = process.communicate(timeout=0.2)
                        results[index] = (process.returncode, output)
                live = [index for index, process in enumerate(processes) if process.poll() is None]
                if len(live) == 1 and len(results) == 2:
                    state_path = runtime / "server.json"
                    try:
                        state = json.loads(state_path.read_text(encoding="utf-8"))
                        health_url = f"http://127.0.0.1:{state['port']}/api/health"
                        opener = build_opener(ProxyHandler({}))
                        with opener.open(health_url, timeout=0.25) as response:
                            health = json.load(response)
                        break
                    except (OSError, ValueError, KeyError, HTTPError):
                        pass
                time.sleep(0.02)
            else:
                live = [index for index, process in enumerate(processes) if process.poll() is None]
                self.fail(f"startup did not converge; live={live}, completed={results}")

            self.assertEqual(len(live), 1, f"expected one persistent server process; live={live}")
            self.assertEqual(state["port"], port)
            self.assertEqual(health["app"], APP_ID)
            self.assertEqual(health["project_id"], project_identity(self.root))
            self.assertEqual(health["status"], "ok")
            for index, (returncode, output) in results.items():
                with self.subTest(process=index):
                    self.assertEqual(returncode, 0, output)
                    self.assertIn("已在运行", output)
        finally:
            for process in processes:
                if process.poll() is None:
                    try:
                        process.send_signal(signal.SIGINT)
                    except ProcessLookupError:
                        pass
            stop_deadline = time.monotonic() + 1
            while time.monotonic() < stop_deadline and any(process.poll() is None for process in processes):
                time.sleep(0.02)
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            terminate_deadline = time.monotonic() + 0.75
            while time.monotonic() < terminate_deadline and any(process.poll() is None for process in processes):
                time.sleep(0.02)
            for process in processes:
                if process.poll() is None:
                    process.kill()
                try:
                    process.communicate(timeout=0.5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=0.5)


if __name__ == "__main__":
    unittest.main()
