#!/usr/bin/env bash
# 板端启动脚本
set -e
cd "$(dirname "$0")/.."

ASCEND_HOME="${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit}"
if [ -f "$ASCEND_HOME/set_env.sh" ]; then
  # shellcheck disable=SC1090
  source "$ASCEND_HOME/set_env.sh"
elif [ -f "$ASCEND_HOME/latest/set_env.sh" ]; then
  # shellcheck disable=SC1090
  source "$ASCEND_HOME/latest/set_env.sh"
fi

export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONIOENCODING=utf-8
mkdir -p data/events

# 选择解释器：优先项目 venv，其次 $PYTHON，其次板端自带 conda base，最后 python3
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif [ -x "/usr/local/miniconda3/bin/python" ]; then
  PY="/usr/local/miniconda3/bin/python"
else
  PY=python3
fi

echo "Atlas Fire Watch 启动中…"
exec "$PY" app/main.py --config configs/app.json "$@"
