# -*- coding: utf-8 -*-
"""摄像头/演示图片源。

source:
  0, 1 ...        USB 摄像头（/dev/video0）
  dir:/path       图片目录循环
  file:/path      单张图片
"""
import os
import glob
import time
import threading
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# dir: 源一次性载入内存的图片张数上限（约 200 x 640x480 x 3B ≈ 180MB）
MAX_DIR_IMAGES = 200


def _log(msg):
    print("[camera] %s" % msg, flush=True)


def _imread_u(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(str(path))


def resolve_source(source):
    """把 dir:/file: 的相对路径解析到工程根目录。

    注意两种前缀长度不同（dir: 4 字符、file: 5 字符），不能统一按 5 截取，
    否则 "dir:images" 会被解析成 "dir:." + "images"。
    """
    source = str(source)
    for prefix in ("dir:", "file:"):
        if source.startswith(prefix):
            raw = source[len(prefix):]
            if not raw:
                return source
            p = Path(raw)
            if not p.is_absolute():
                p = ROOT / p
            return prefix + str(p)
    return source


class Camera(object):
    def __init__(self, source="0", width=640, height=480, fps=25, fallback=None):
        self.source = resolve_source(source)
        self.fallback = resolve_source(fallback) if fallback else None
        self.active_source = self.source
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1, int(fps))
        self.fallback_used = False
        self.last_error = None

        self._cap = None
        self._images = []
        self._mock_index = 0
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    # ------------------------------------------------------------------
    def open(self):
        try:
            self._open_source(self.source)
        except Exception as e:
            self.last_error = str(e)
            if self.fallback and self.fallback != self.source:
                _log("视频源打开失败（%s），回退：%s" % (e, self.fallback))
                self._images, self._cap = [], None
                self._open_source(self.fallback)
                self.fallback_used = True
            else:
                raise
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        t0 = time.time()
        while self._frame is None and time.time() - t0 < 5:
            time.sleep(0.02)
        if self._frame is None:
            raise RuntimeError("视频源已打开但 5 秒内没有取到画面")
        return True

    def _open_source(self, source):
        self.active_source = source
        if source.startswith("dir:"):
            path = source[4:]
            files = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                files.extend(glob.glob(os.path.join(path, ext)))
            files.sort()
            # 图片是一次性全部读进内存的，指向 dataset/images/val 这种几千张的
            # 目录会直接把 4GB 的板子撑爆（4306 张 640x640 约 4~5GB），所以设上限。
            # 需要更多样本就改 MAX_DIR_IMAGES，但别真的整个数据集目录指过来。
            if len(files) > MAX_DIR_IMAGES:
                _log("目录图片过多（%d 张），只加载前 %d 张：%s"
                     % (len(files), MAX_DIR_IMAGES, path))
                files = files[:MAX_DIR_IMAGES]
            images = []
            for f in files:
                im = _imread_u(f)
                if im is not None:
                    images.append(im)
            self._images = images
            if not self._images:
                raise RuntimeError("目录中没有可读图片：%s" % path)
            _log("使用图片目录：%s（%d 张）" % (path, len(self._images)))
            return
        if source.startswith("file:"):
            path = source[5:]
            im = _imread_u(path)
            if im is None:
                raise RuntimeError("图片无法读取：%s" % path)
            self._images = [im]
            _log("使用单张图片：%s" % path)
            return

        if source.startswith("/dev/video"):
            index = int(source.replace("/dev/video", ""))
        else:
            index = int(source)
        if os.name != "nt":
            self._cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if not self._cap.isOpened():
                self._cap.release()
                self._cap = cv2.VideoCapture(index)
        else:
            self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError("摄像头 %s 打开失败（检查 /dev/video0）" % source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _log("摄像头 %s 已打开：%dx%d @%dfps" % (source, w, h, self.fps))

    # ------------------------------------------------------------------
    def _grab(self):
        if self._images:
            im = self._images[self._mock_index % len(self._images)]
            self._mock_index += 1
            return im.copy()
        if self._cap is not None:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                return frame
        return None

    def _loop(self):
        interval = 1.0 / self.fps
        while self._running:
            st = time.time()
            try:
                frame = self._grab()
            except Exception as e:
                self.last_error = str(e)
                frame = None
            if frame is not None:
                with self._lock:
                    self._frame = frame
                if self._images:
                    dt = time.time() - st
                    if dt < interval:
                        time.sleep(interval - dt)
            else:
                time.sleep(0.05)

    # ------------------------------------------------------------------
    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def info(self):
        src = self.active_source
        if src.startswith("dir:"):
            label = "图片目录 · %s" % (os.path.basename(src[4:].rstrip("/\\")) or "demo")
        elif src.startswith("file:"):
            label = "单张图片 · %s" % os.path.basename(src[5:])
        elif src.startswith("/dev/video"):
            label = "USB 摄像头 · %s" % src
        else:
            label = "USB 摄像头 /dev/video%s" % src
        if self.fallback_used:
            label = "回退演示 · " + label
        return {
            "source": src,
            "configured_source": self.source,
            "label": label,
            "fallback_used": self.fallback_used,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }

    def close(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class PlaceholderCamera(object):
    def __init__(self, message):
        self.message = str(message)
        self.source = "none"
        self.fallback_used = False
        self._frame = np.full((480, 640, 3), 18, dtype=np.uint8)
        cv2.putText(self._frame, "CAMERA NOT AVAILABLE", (100, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 140, 255), 2)
        cv2.putText(self._frame, "check /dev/video0 or configs/app.json",
                    (100, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1)

    def read(self):
        return self._frame.copy()

    def info(self):
        return {"source": "none", "label": "摄像头不可用", "fallback_used": False,
                "width": 640, "height": 480, "fps": 0, "error": self.message}

    def close(self):
        pass
