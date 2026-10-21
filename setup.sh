#!/bin/bash
set -euo pipefail

TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$TASK_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "没有找到 Python 3。请安装 Python 3.12 或更新版本后重新启动。"
  exit 1
fi
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else "需要 Python 3.12 或更新版本")'

if [ ! -x "$TASK_DIR/.venv/bin/python" ]; then
  echo "正在创建本项目专用的 Python 环境……"
  python3 -m venv "$TASK_DIR/.venv"
fi

if ! "$TASK_DIR/.venv/bin/python" -c 'from importlib.metadata import version; import platform; assert version("apsw") == "3.54.1.0"; assert version("numpy") == "2.5.4"; assert version("torch") == "2.14.1"; assert platform.system() != "Darwin" or platform.machine() != "arm64" or version("mlx") == "0.32.4"' >/dev/null 2>&1; then
  echo "正在安装本项目专用的 SQLite、MLX 和模型依赖（首次安装需要联网）……"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$TASK_DIR/.venv/bin/python" --system-certs --requirement "$TASK_DIR/requirements.txt"
  else
    "$TASK_DIR/.venv/bin/python" -m pip install --disable-pip-version-check --requirement "$TASK_DIR/requirements.txt"
  fi
fi

"$TASK_DIR/.venv/bin/python" - <<'PY'
from pathlib import Path
from trash_talk.storage import CorpusDatabase
root = Path.cwd()
stats = CorpusDatabase(root / 'data/trash_talk.sqlite3').initialize(root / 'data/corpus.json')
print(f"准备完成 · SQLite {stats['sqlite_version']} · {stats['fragments']} 条语料 · {stats['templates']} 个句型")
PY
