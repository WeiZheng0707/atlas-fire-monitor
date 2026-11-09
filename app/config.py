# -*- coding: utf-8 -*-
"""运行配置加载（JSON，避免板端额外依赖 PyYAML）。"""
import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


DEFAULTS = {
    "server": {"host": "0.0.0.0", "port": 8000},
    "camera": {"source": "0", "width": 640, "height": 480, "fps": 25, "fallback": ""},
    "model": {
        "backend": "auto",
        "om_path": "models/fire_yolov8n.om",
        "onnx_path": "models/fire_yolov8n.onnx",
        "labels": "models/labels.txt",
        "input_size": 640,
        "device_id": 0,
        "conf_threshold": 0.35,
        "iou_threshold": 0.5,
        "max_detections": 30,
        "class_agnostic_nms": True,
    },
    "alarm": {
        "enabled": True,
        "class_names": ["fire", "smoke"],
        "threshold": 0.45,                          # 兼容字段：火焰阈值
        "thresholds": {"fire": 0.45, "smoke": 0.35},
        "confirm_frames": {"fire": 1, "smoke": 3},  # 连续帧确认，抑制误报
        "cooldown_seconds": 10,
        "hold_seconds": 6,
        "escalate": True,
        "save_snapshots": True,
        "save_dir": "data/events",
        "max_events": 100,
    },
    "stream": {"infer_interval_ms": 0, "max_fps": 15, "jpeg_quality": 75},
    "system": {"npu_status_cache_seconds": 10,
               "ascend_home": "/usr/local/Ascend/ascend-toolkit"},
}


def deep_update(base, extra):
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=None):
    path = path or os.environ.get("ATLAS_FIRE_CONFIG") or str(ROOT / "configs" / "app.json")
    cfg = json.loads(json.dumps(DEFAULTS))
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            deep_update(cfg, json.load(f))
    # 环境变量便于板端临时覆盖
    if os.environ.get("ATLAS_FIRE_BACKEND"):
        cfg["model"]["backend"] = os.environ["ATLAS_FIRE_BACKEND"]
    if os.environ.get("ATLAS_FIRE_PORT"):
        cfg["server"]["port"] = int(os.environ["ATLAS_FIRE_PORT"])
    if os.environ.get("ATLAS_FIRE_DEVICE_ID"):
        cfg["model"]["device_id"] = int(os.environ["ATLAS_FIRE_DEVICE_ID"])
    return cfg


def resolve_path(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
