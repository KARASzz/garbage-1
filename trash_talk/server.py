"""Loopback-only HTTP API and an explicit allowlist of local web assets."""

from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit

from . import APP_ID
from .generator import GrammarGenerator
from .feedback import FeedbackStore
from .model_runtime import BackgroundLearner, ModelRuntime
from .storage import CorpusDatabase, CorpusError

LOGGER = logging.getLogger(__name__)
ASSETS = {"/": ("index.html", "text/html; charset=utf-8"), "/index.html": ("index.html", "text/html; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/favicon.svg": ("favicon.svg", "image/svg+xml")}


class LocalHTTPServer(ThreadingHTTPServer):
    """Keep abandoned local connections from consuming unlimited threads."""

    daemon_threads = True

    def __init__(self, *args, **kwargs):
        self._connections = threading.BoundedSemaphore(32)
        super().__init__(*args, **kwargs)

    def get_request(self):
        request, address = super().get_request()
        request.settimeout(5)
        return request, address

    def process_request(self, request, client_address):
        if not self._connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connections.release()

    def server_close(self):
        learner = getattr(self, "learner", None)
        if learner is not None:
            learner.close()
        super().server_close()


def project_identity(root: Path) -> str:
    return hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]


def create_server(root: Path, port: int = 8765, *, seed: int | None = None, backend: str = "mlx") -> ThreadingHTTPServer:
    root = root.resolve()
    database = CorpusDatabase(root / "data" / "trash_talk.sqlite3")
    stats = database.initialize(root / "data" / "corpus.json")
    generator = GrammarGenerator(database, seed=seed)
    feedback = FeedbackStore(root / "data" / "feedback.sqlite3")

    class Handler(BaseHTTPRequestHandler):
        server_version = "TrashTerminal/2.0"

        def _send(self, status: int, body: bytes, content_type: str, *, head: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            if not head:
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def _json(self, status: int, body: dict, *, head: bool = False) -> None:
            self._send(status, json.dumps(body, ensure_ascii=False).encode(), "application/json; charset=utf-8", head=head)

        def _route(self, *, head: bool = False) -> None:
            # Do not serve alternate hostnames through a loopback service.
            host = self.headers.get("Host", "").split(":", 1)[0].lower()
            if host not in ("127.0.0.1", "localhost"):
                self._json(403, {"error": "仅支持本机访问"}, head=head)
                return
            route = urlsplit(self.path)
            if route.path == "/api/health":
                self._json(200, {"app": APP_ID, "project_id": project_identity(root), "status": "ok", "api_version": 2, "backend": backend, **stats}, head=head)
            elif route.path == "/api/model":
                self._json(200, {**runtime.status(), "counts": feedback.counts(), "queue": feedback.queue_counts()}, head=head)
            elif route.path == "/api/draw":
                if head:
                    self._json(405, {"error": "抽取语料请使用 GET"}, head=True)
                    return
                try:
                    result = generator.draw(debug=parse_qs(route.query).get("debug") == ["1"])
                    self._json(200, result)
                except CorpusError:
                    LOGGER.exception("Grammar expansion failed")
                    self._json(503, {"error": "这台破机器卡壳了，再按一次试试。"})
                except Exception:
                    LOGGER.exception("Unexpected generation failure")
                    self._json(500, {"error": "信号暂时中断，再按一次重试。"})
            elif route.path in ASSETS:
                filename, mime = ASSETS[route.path]
                try:
                    self._send(200, (root / "web" / filename).read_bytes(), mime, head=head)
                except FileNotFoundError:
                    self._json(404, {"error": "页面资源不存在"}, head=head)
            else:
                self._json(404, {"error": "这里没有这个频道"}, head=head)

        def do_GET(self) -> None:
            self._route()

        def do_HEAD(self) -> None:
            self._route(head=True)

        def do_POST(self) -> None:
            host = self.headers.get("Host", "")
            if host.split(":", 1)[0].lower() not in ("127.0.0.1", "localhost"):
                self._json(403, {"error": "仅支持本机访问"})
                return
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"):
                self._json(403, {"error": "评价和生成只接受当前本地页面"})
                return
            route = urlsplit(self.path).path
            if route not in ("/api/draw", "/api/ai", "/api/feedback"):
                self._json(404, {"error": "这里没有这个频道"})
                return
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                self._json(415, {"error": "请提交 JSON"})
                return
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("不接受分块请求")
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 4096:
                    raise ValueError("请求大小无效")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("请求必须为 JSON 对象")
            except (ValueError, OSError):
                self._json(400, {"error": "请求格式无效"})
                return
            try:
                if route == "/api/feedback":
                    result = feedback.rate(data.get("sample_id"), data.get("rating"))
                    # The response never waits for a gradient update or disk checkpoint.
                    learner.notify()
                else:
                    source = "rule" if route == "/api/draw" else "garbage-1"
                    result = generator.draw() if source == "rule" else runtime.generate()
                    model = result.get("model", runtime.status())
                    sample_id = feedback.add_sample(result["text"], source, result["theme"], model["step"], model["run_id"])
                    result = {**result, "sample_id": sample_id, "source": source, "rating": 0, "model": model}
                self._json(200, result)
            except LookupError:
                self._json(404, {"error": "样本不存在，请先生成一句"})
            except ValueError as error:
                self._json(400, {"error": str(error)})
            except RuntimeError:
                self._json(503, {"error": "garbage-1 暂未就绪，请检查本地权重和 MLX 依赖。规则抽卡仍可用。"})
            except Exception:
                LOGGER.exception("Generation or annotation failed")
                self._json(500, {"error": "本地服务暂时卡壳了，请再试一次。"})

        def log_message(self, format: str, *args: object) -> None:
            LOGGER.info(format, *args)

    server = LocalHTTPServer(("127.0.0.1", port), Handler)
    try:
        runtime = ModelRuntime(root, backend=backend)
        learner = BackgroundLearner(runtime, feedback)
        server.learner = learner
        server.model_runtime = runtime
        server.feedback_store = feedback
    except BaseException:
        server.server_close()
        raise
    return server
