# -*- coding: utf-8 -*-
"""Atlas 200I DK A2 视觉火情监测 · Web 服务入口。

启动：
    python3 app/main.py --config configs/app.json
    python3 app/main.py --camera 0 --port 8000
"""
import os
import sys
import json
import time
import socket
import argparse
from pathlib import Path

# 允许直接 `python3 app/main.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from flask import Flask, Response, request, jsonify, send_from_directory

from app.config import load_config, resolve_path, DEFAULTS, deep_update
from app.camera import Camera, PlaceholderCamera
from app.detector import build_detector
from app.monitor import FireMonitor
from app.stream import InferenceService
from app.system_info import SystemInfo

VERSION = "1.0.0"
WEB_DIR = PROJECT_ROOT / "app" / "web"


def create_app(config=None, detector=None, camera=None, start_service=False):
    """应用工厂；测试时可注入 fake detector/camera。"""
    if isinstance(config, dict):
        cfg = json.loads(json.dumps(DEFAULTS))
        deep_update(cfg, config)
    else:
        cfg = load_config(config)

    if detector is None:
        detector = build_detector(cfg)
    if camera is None:
        cam_cfg = cfg["camera"]
        try:
            camera = Camera(source=cam_cfg.get("source", "0"),
                            width=cam_cfg.get("width", 640),
                            height=cam_cfg.get("height", 480),
                            fps=cam_cfg.get("fps", 25),
                            fallback=cam_cfg.get("fallback"))
            camera.open()
        except Exception as e:
            print("[camera] 打开失败：%s" % e, flush=True)
            camera = PlaceholderCamera(str(e))

    monitor = FireMonitor(cfg.get("alarm", {}),
                          resolve_path(cfg.get("alarm", {}).get("save_dir", "data/events")))
    service = InferenceService(cfg, detector, camera, monitor)
    if start_service:
        service.start()

    app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    try:
        app.json.ensure_ascii = False
    except Exception:
        pass
    app.extensions["inference_service"] = service
    app.extensions["monitor"] = monitor
    app.extensions["detector"] = detector
    app.extensions["camera"] = camera
    app.extensions["config"] = cfg
    sysinfo = SystemInfo(cfg.get("system", {}).get("ascend_home",
                                                   "/usr/local/Ascend/ascend-toolkit"),
                         cfg.get("system", {}).get("npu_status_cache_seconds", 10))
    app.extensions["system_info"] = sysinfo

    def _model_summary():
        info = detector.info()
        return {
            "available": bool(info.get("available")),
            "backend": info.get("backend"),
            "device": info.get("device"),
            "model": info.get("model"),
            "input_size": info.get("input_size", cfg["model"].get("input_size")),
            "labels": info.get("labels", []),
            "error": info.get("error"),
            "conf_threshold": cfg["model"].get("conf_threshold"),
            "alarm_threshold": cfg["alarm"].get("threshold"),
        }

    @app.after_request
    def no_cache_api(resp):
        if request.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.route("/")
    def index():
        return send_from_directory(str(WEB_DIR), "index.html")

    @app.route("/favicon.ico")
    def favicon():
        return send_from_directory(str(WEB_DIR), "favicon.ico")

    @app.route("/api/config")
    def api_config():
        return jsonify({
            "version": VERSION,
            "model": _model_summary(),
            "alarm": {
                "class_names": cfg["alarm"].get("class_names", ["fire", "smoke"]),
                "threshold": cfg["alarm"].get("threshold"),
                "thresholds": cfg["alarm"].get("thresholds", {}),
                "confirm_frames": cfg["alarm"].get("confirm_frames", {}),
                "cooldown": cfg["alarm"].get("cooldown_seconds"),
                "hold_seconds": cfg["alarm"].get("hold_seconds"),
            },
            "stream": {"infer_interval_ms": cfg["stream"].get("infer_interval_ms", 0)},
            "camera": camera.info(),
        })

    @app.route("/api/health")
    def api_health():
        sysinfo_data = app.extensions["system_info"].snapshot()
        svc = service.health()
        return jsonify({
            "ok": True,
            "version": VERSION,
            "time": time.time(),
            "model": _model_summary(),
            "service": svc,
            "system": sysinfo_data,
            "camera": camera.info(),
            "alarm": monitor.snapshot(),
            "events": len(monitor.events()),
        })

    @app.route("/api/stream")
    def api_stream():
        try:
            start_seq = int(request.args.get("seq", 0))
        except ValueError:
            start_seq = 0
        resp = Response(service.sse(start_seq), mimetype="text/event-stream")
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    @app.route("/api/detect", methods=["POST"])
    def api_detect():
        file_obj = request.files.get("file") or request.files.get("file_name")
        if file_obj is None:
            return jsonify({"ok": False, "error": "缺少 file 字段"}), 400
        raw = file_obj.read()
        if not raw:
            return jsonify({"ok": False, "error": "图片为空"}), 400
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"ok": False, "error": "图片无法解码"}), 400
        try:
            result = service.analyze_image(image)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        result["ok"] = True
        result["filename"] = file_obj.filename
        return jsonify(result)

    @app.route("/api/snapshot", methods=["POST"])
    def api_snapshot():
        file_obj = request.files.get("file")
        if file_obj is not None:
            raw = file_obj.read()
            item = monitor.save_manual(raw)
        else:
            item = service.save_current()
        if item is None:
            return jsonify({"ok": False, "error": "当前没有可保存的画面"}), 400
        return jsonify({"ok": True, "event": item})

    @app.route("/api/events")
    def api_events():
        return jsonify({"ok": True, "events": monitor.events()})

    @app.route("/api/events/<path:name>")
    def api_event_image(name):
        resp = send_from_directory(str(monitor.save_dir), name, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.route("/api/alarm/clear", methods=["POST", "GET"])
    def api_alarm_clear():
        return jsonify({"ok": True, "alarm": monitor.clear()})

    @app.route("/api/board")
    def api_board():
        return jsonify({"ok": True,
                        "system": app.extensions["system_info"].snapshot(force=True)})

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"ok": False, "error": "图片超过 20MB"}), 413

    return app


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main():
    parser = argparse.ArgumentParser(description="Atlas 200I DK A2 视觉火情监测")
    parser.add_argument("--config", default=None)
    parser.add_argument("--camera", default=None, help="0 / dir:path / file:path")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--backend", default=None, choices=["auto", "om", "onnx"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.camera is not None:
        cfg["camera"]["source"] = args.camera
    if args.port is not None:
        cfg["server"]["port"] = args.port
    if args.host is not None:
        cfg["server"]["host"] = args.host
    if args.backend is not None:
        cfg["model"]["backend"] = args.backend

    app = create_app(cfg, start_service=False)
    service = app.extensions["inference_service"]
    service.start()

    ip = local_ip()
    port = int(cfg["server"]["port"])
    sysinfo = app.extensions["system_info"].snapshot()
    model = app.extensions["detector"].info()
    print("=" * 78)
    print(" Atlas 视觉火情监测系统 v%s" % VERSION)
    print(" 平台      : %s（%s）" % (sysinfo["platform"], sysinfo["arch"]))
    print(" NPU       : %s" % ("可用" if sysinfo["npu_available"] else "未检测到 npu-smi"))
    print(" 模型      : %s / %s" % (model.get("backend"), model.get("model") or "未就绪"))
    print(" 摄像头    : %s" % app.extensions["camera"].info().get("label"))
    print(" 访问地址  : http://%s:%d/" % (ip, port))
    print(" 健康检查  : http://%s:%d/api/health" % (ip, port))
    if not model.get("available"):
        print(" 提示      : 模型未就绪，请先训练并放入 models/，详见 README.md")
    print("=" * 78)
    try:
        app.run(host=cfg["server"]["host"], port=port, threaded=True,
                debug=False, use_reloader=False)
    finally:
        service.stop()


if __name__ == "__main__":
    main()
