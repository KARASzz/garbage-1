#!/bin/bash
set -euo pipefail

TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$TASK_DIR"

if [ ! -x "$TASK_DIR/.venv/bin/python" ] || ! "$TASK_DIR/.venv/bin/python" -c 'from importlib.metadata import version; assert version("apsw") == "3.54.1.0"; assert version("numpy") == "2.5.4"; assert version("torch") == "2.14.1"; assert version("mlx") == "0.32.4"' >/dev/null 2>&1; then
  if ! /bin/bash "$TASK_DIR/setup.sh"; then
    echo "安装未完成。请检查上方错误，联网后再次双击此文件。"
    read -r -p "按回车关闭窗口……" ignored
    exit 1
  fi
fi

if ! "$TASK_DIR/.venv/bin/python" "$TASK_DIR/run.py" --open; then
  echo "启动失败，请查看上方提示。"
  read -r -p "按回车关闭窗口……" ignored
  exit 1
fi
