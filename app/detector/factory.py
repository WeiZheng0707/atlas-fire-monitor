# -*- coding: utf-8 -*-
"""推理后端工厂：OM -> ONNX -> 不可用（页面仍可启动并提示）。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class UnavailableDetector(object):
    backend_name = "unavailable"

    def __init__(self, error):
        self.error = str(error)

    def infer(self, image):
        return [], 0.0

    def info(self):
        return {
            "available": False,
            "backend": self.backend_name,
            "device": "未就绪",
            "model": None,
            "input_size": None,
            "labels": [],
            "error": self.error,
        }

    def close(self):
        pass


def read_labels(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    labels = []
    if p.exists():
        with open(p, "r", encoding="utf-8-sig") as f:
            labels = [line.strip() for line in f if line.strip()]
    return labels or ["fire"]


def _resolve(path):
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return str(p)


def build_detector(config):
    """根据 config['model'] 创建检测器；绝不因缺模型让 Web 服务崩溃。"""
    m = config.get("model", {})
    labels = read_labels(m.get("labels", "models/labels.txt"))
    common = dict(
        labels=labels,
        input_size=int(m.get("input_size", 640)),
        conf_threshold=float(m.get("conf_threshold", 0.35)),
        iou_threshold=float(m.get("iou_threshold", 0.5)),
        max_detections=int(m.get("max_detections", 30)),
        class_agnostic=bool(m.get("class_agnostic_nms", True)),
    )
    backend = str(m.get("backend", "auto")).lower()
    om_path = _resolve(m.get("om_path", "models/fire_yolov8n.om"))
    onnx_path = _resolve(m.get("onnx_path", "models/fire_yolov8n.onnx"))
    errors = []

    def try_om():
        from .om_backend import AscendOMDetector
        return AscendOMDetector(om_path, device_id=int(m.get("device_id", 0)), **common)

    def try_onnx():
        from .onnx_backend import OnnxDetector
        return OnnxDetector(onnx_path, **common)

    if backend in ("auto", "om", "ascend", "npu") and om_path and os.path.exists(om_path):
        try:
            detector = try_om()
            print("[model] 已加载 Ascend OM：%s" % om_path, flush=True)
            return detector
        except Exception as e:
            errors.append("OM 加载失败：%s" % e)
            print("[model] %s" % errors[-1], flush=True)

    if backend in ("auto", "om", "ascend", "npu", "onnx") and onnx_path and os.path.exists(onnx_path):
        try:
            detector = try_onnx()
            print("[model] 已加载 ONNX Runtime CPU：%s" % onnx_path, flush=True)
            return detector
        except Exception as e:
            errors.append("ONNX 加载失败：%s" % e)
            print("[model] %s" % errors[-1], flush=True)

    if backend == "auto":
        errors.append(
            "未找到可用模型。训练后把 fire_yolov8n.onnx / .om 放到 models/；"
            "或在 configs/app.json 中检查 model.om_path / model.onnx_path。")
    else:
        errors.append("指定的后端 %s 不可用。" % backend)
    msg = "\n".join(errors)
    print("[model] 模型未就绪：%s" % msg, flush=True)
    return UnavailableDetector(msg)
