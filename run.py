#!/usr/bin/env python3
"""Start (or reopen) this project's local CRT terminal."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

try:
    from trash_talk import APP_ID
    from trash_talk.server import create_server, project_identity
except ImportError as error:
    print(f"缺少项目依赖：{error}\n请先运行 ./setup.sh，或双击 启动.command。", file=sys.stderr)
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parent


def existing_url(port: int, backend: str = "mlx") -> str | None:
    url = f"http://127.0.0.1:{port}"
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url + "/api/health", timeout=0.5) as response:
            data = json.load(response)
        if data.get("app") == APP_ID and data.get("project_id") == project_identity(ROOT) and data.get("api_version") == 2 and data.get("backend") == backend:
            return url
    except (OSError, ValueError, urllib.error.URLError):
        pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="本地 CRT 垃圾话生成器")
    parser.add_argument("--port", type=int, default=8765, help="首选端口，默认 8765")
    parser.add_argument("--open", action="store_true", help="启动后打开默认浏览器")
    parser.add_argument("--backend", choices=("mlx", "cuda", "torch-cpu"), default="mlx", help="默认本机 MLX；CUDA 需要另一台具有 CUDA 的机器")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65525:
        parser.error("端口必须在 1024 到 65525 之间")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True)
    state_path = runtime / "server.json"
    server = None
    thread = None
    url = None
    # Serialize startup until the first service is actually accepting health
    # requests, so two near-simultaneous double clicks reuse the same process.
    with (runtime / "startup.lock").open("a") as startup_lock:
        fcntl.flock(startup_lock, fcntl.LOCK_EX)
        known_ports = [args.port]
        try:
            state = json.loads(state_path.read_text())
            if isinstance(state.get("port"), int) and state["port"] not in known_ports:
                known_ports.append(state["port"])
        except (OSError, ValueError):
            pass
        for port in known_ports:
            url = existing_url(port, args.backend)
            if url:
                break
        if url is None:
            for port in range(args.port, args.port + 10):
                try:
                    server = create_server(ROOT, port, backend=args.backend)
                    break
                except OSError as error:
                    if error.errno not in (48, 98, 10048):
                        raise
            if server is not None:
                port = server.server_port
                url = f"http://127.0.0.1:{port}"
                state_path.write_text(json.dumps({"port": port, "url": url}), encoding="utf-8")
                thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.3}, daemon=True)
                thread.start()
                deadline = time.monotonic() + 3
                while existing_url(port, args.backend) is None:
                    if time.monotonic() >= deadline:
                        server.shutdown()
                        server.server_close()
                        raise RuntimeError("本地服务未能通过启动健康检查")
                    time.sleep(0.03)
    if url is None:
        print("附近端口均被占用，请用 --port 指定其他端口。", file=sys.stderr)
        return 1
    if server is None:
        print(f"垃圾话生成器已在运行：{url}", flush=True)
        if args.open:
            webbrowser.open(url)
        return 0
    print(f"\n垃圾话生成器已启动：{url}\n保持此终端窗口开启。按 Control+C 关闭服务。\n", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        thread.join()
    except KeyboardInterrupt:
        print("\n显像管已关闭。")
    finally:
        server.shutdown()
        server.server_close()
        try:
            current = json.loads(state_path.read_text())
            if current.get("port") == port:
                state_path.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"启动失败：{error}", file=sys.stderr)
        raise SystemExit(1)
