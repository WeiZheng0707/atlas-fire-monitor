#!/usr/bin/env bash
# 在板端安装运行依赖（不安装训练框架 torch/ultralytics）
set -e
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
echo "[1/4] Python: $($PY -V 2>&1)"

if [ ! -d ".venv" ]; then
  echo "[2/4] 创建 venv"
  $PY -m venv .venv
fi
PIP=".venv/bin/pip"
$PIP install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple || true

echo "[3/4] 安装板端依赖"
$PIP install -r requirements/board.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
  || $PIP install -r requirements/board.txt

echo "[4/4] 环境检查"
.venv/bin/python - <<'PY'
import cv2, flask, numpy
print("opencv :", cv2.__version__)
print("flask  :", flask.__version__)
print("numpy  :", numpy.__version__)
try:
    import onnxruntime
    print("onnxruntime:", onnxruntime.__version__)
except Exception as e:
    print("onnxruntime 未安装（如只使用 OM 后端可忽略）：", e)
PY
echo "依赖安装完成。下一步：bash deployment/convert_om.sh"
