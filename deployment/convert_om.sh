#!/usr/bin/env bash
# 在 Atlas 200I DK A2 板端把 ONNX 转成 Ascend 310B4 的 OM。
# 需要 CANN Toolkit（镜像一般已预装）。
set -e

cd "$(dirname "$0")/.."

ASCEND_HOME="${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit}"
SOC_VERSION="${SOC_VERSION:-Ascend310B4}"
ONNX="${ONNX:-models/fire_yolov8n.onnx}"
OUT="${OUT:-models/fire_yolov8n}"
INPUT_NAME="${INPUT_NAME:-images}"
INPUT_SIZE="${INPUT_SIZE:-640}"

if [ -f "$ASCEND_HOME/set_env.sh" ]; then
  # shellcheck disable=SC1090
  source "$ASCEND_HOME/set_env.sh"
elif [ -f "$ASCEND_HOME/latest/set_env.sh" ]; then
  # shellcheck disable=SC1090
  source "$ASCEND_HOME/latest/set_env.sh"
fi

if ! command -v atc >/dev/null 2>&1; then
  echo "错误：未找到 atc。请确认 CANN Toolkit 已安装并执行："
  echo "  source /usr/local/Ascend/ascend-toolkit/set_env.sh"
  exit 1
fi
if [ ! -f "$ONNX" ]; then
  echo "错误：ONNX 不存在：$ONNX"
  exit 1
fi

echo "=============================================================="
echo " Atlas ATC 模型转换"
echo " ONNX       : $ONNX"
echo " 输出       : $OUT.om"
echo " SOC        : $SOC_VERSION"
echo " 输入       : $INPUT_NAME:1,3,${INPUT_SIZE},${INPUT_SIZE}"
echo "=============================================================="

npu-smi info 2>/dev/null | head -n 12 || true

atc \
  --framework=5 \
  --model="$ONNX" \
  --output="$OUT" \
  --input_shape="${INPUT_NAME}:1,3,${INPUT_SIZE},${INPUT_SIZE}" \
  --input_format=NCHW \
  --soc_version="$SOC_VERSION" \
  --output_type=FP32 \
  --precision_mode=allow_fp32_to_fp16 \
  --log=error

if [ -f "${OUT}.om" ]; then
  echo "--------------------------------------------------------------"
  ls -lh "${OUT}.om"
  echo "转换完成。运行：bash deployment/start.sh"
else
  echo "转换失败：未生成 ${OUT}.om"
  exit 1
fi
