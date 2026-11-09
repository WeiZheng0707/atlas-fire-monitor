# -*- coding: utf-8 -*-
"""持续推理线程 + SSE 结果广播。"""
import time
import json
import base64
import threading

import cv2

from .annotate import annotate


class InferenceService(object):
    def __init__(self, config, detector, camera, monitor):
        self.cfg = config
        self.detector = detector
        self.camera = camera
        self.monitor = monitor
        self.jpeg_quality = int(config.get("stream", {}).get("jpeg_quality", 75))
        self.interval = max(0.0, float(
            config.get("stream", {}).get("infer_interval_ms", 0)) / 1000.0)
        self.max_fps = max(1.0, float(config.get("stream", {}).get("max_fps", 15)))

        self._cond = threading.Condition()
        self._latest = None
        self._seq = 0
        self._running = False
        self._thread = None
        self._last_ts = 0.0
        self._fps = 0.0
        self._last_error = None
        self._started_at = time.time()
        self._frames = 0

    # ------------------------------------------------------------------
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="inference", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        try:
            self.camera.close()
        except Exception:
            pass
        try:
            self.detector.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _encode(self, image):
        ok, buf = cv2.imencode(".jpg", image,
                               [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return None, None
        raw = buf.tobytes()
        return raw, base64.b64encode(raw).decode("ascii")

    def _loop(self):
        while self._running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.05)
                continue
            try:
                st = time.time()
                dets, infer_ms = self.detector.infer(frame)
                info = self.detector.info()
                image = annotate(frame, dets, backend=info.get("backend", ""))
                jpeg, b64 = self._encode(image)
                alarm = self.monitor.observe(dets, jpeg)
                counts = {"fire": 0, "smoke": 0}
                for d in dets:
                    cls = str(d.get("label", "")).lower()
                    if cls in counts:
                        counts[cls] += 1
                now = time.time()
                if self._last_ts:
                    dt = max(1e-3, now - self._last_ts)
                    self._fps = 0.85 * self._fps + 0.15 * (1.0 / dt) if self._fps else 1.0 / dt
                self._last_ts = now
                self._frames += 1
                payload = {
                    "seq": self._seq + 1,
                    "image": b64,
                    "detections": dets,
                    "detection_count": len(dets),
                    "counts": counts,
                    "max_score": round(max([d.get("score", 0) for d in dets] or [0.0]), 4),
                    "alarm": alarm,
                    "alarm_level": alarm.get("level", 0),
                    "stats": {
                        "infer_ms": round(float(infer_ms), 1),
                        "total_ms": round((time.time() - st) * 1000.0, 1),
                        "fps": round(self._fps, 1),
                        "frames": self._frames,
                        "uptime_s": round(time.time() - self._started_at, 1),
                    },
                    "model": info,
                    "frame": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                with self._cond:
                    self._seq += 1
                    self._latest = payload
                    self._last_error = None
                    self._cond.notify_all()
            except Exception as e:
                self._last_error = str(e)
                print("[infer] 推理异常：%s" % e, flush=True)
                time.sleep(0.2)
            # 控制循环上限；模型未就绪时降频，避免空转占 CPU
            if not self.detector.info().get("available"):
                time.sleep(0.2)
            else:
                target = max(self.interval, 1.0 / self.max_fps)
                elapsed = time.time() - st
                if elapsed < target:
                    time.sleep(target - elapsed)

    # ------------------------------------------------------------------
    def latest(self):
        with self._cond:
            return self._latest

    def wait_after(self, seq, timeout=20.0):
        with self._cond:
            if self._latest is None or self._latest["seq"] <= seq:
                self._cond.wait(timeout=timeout)
            return self._latest

    def sse(self, start_seq=0):
        last = int(start_seq)
        while True:
            data = self.wait_after(last)
            if data is None:
                continue
            if data["seq"] <= last:
                # 心跳，防止代理断开
                yield ": keepalive\n\n"
                continue
            last = data["seq"]
            yield "data: %s\n\n" % json.dumps(data, ensure_ascii=False)

    # ------------------------------------------------------------------
    def analyze_image(self, image):
        """单图检测（不触发告警），供上传接口使用。"""
        if image is None or getattr(image, "size", 0) == 0:
            raise ValueError("图像为空")
        info = self.detector.info()
        if not info.get("available"):
            raise RuntimeError(info.get("error") or "模型未就绪，无法执行单图检测")
        st = time.time()
        dets, infer_ms = self.detector.infer(image)
        info = self.detector.info()
        annotated = annotate(image, dets, backend=info.get("backend", ""))
        jpeg, b64 = self._encode(annotated)
        return {
            "image": b64,
            "detections": dets,
            "detection_count": len(dets),
            "max_score": round(max([d.get("score", 0) for d in dets] or [0.0]), 4),
            "stats": {"infer_ms": round(infer_ms, 1),
                      "total_ms": round((time.time() - st) * 1000.0, 1)},
            "model": info,
            "frame": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        }

    def save_current(self):
        latest = self.latest()
        if not latest or not latest.get("image"):
            return None
        return self.monitor.save_manual(base64.b64decode(latest["image"]))

    def health(self):
        latest = self.latest()
        return {
            "running": self._running,
            "seq": self._seq,
            "fps": round(self._fps, 1),
            "last_error": self._last_error,
            "last_frame": latest.get("time") if latest else None,
            "model": self.detector.info(),
            "camera": self.camera.info(),
        }
