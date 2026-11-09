# -*- coding: utf-8 -*-
"""昇腾 OM 推理后端：优先使用 CANN 自带 ais_bench 的 InferSession。

Atlas 200I DK A2（Ascend 310B4）上安装 CANN 后，ais_bench 通常位于：
  /usr/local/Ascend/ascend-toolkit/latest/tools/ais_bench
若镜像未提供，可在 CANN samples 中安装；程序也会尝试常见路径。

重要：ACL context 是「线程私有」的。InferSession 在 Flask 主线程创建，
而真正推理发生在 InferenceService 的 inference 工作线程（以及单图接口的
请求线程），这些线程里 context 为空，首个 rtMalloc 就会报：
  EE1001 / rtMalloc execute failed, reason=[context pointer null]
因此每次推理前都要把主线程捕获到的 context 重新 set 到当前线程。
注意 pyACL 的 acl.rt.get_context() 返回的是 (context_ptr, ret) 元组。
"""
import os
import sys
import time
import threading

import numpy as np

from .preprocess import letterbox
from .postprocess import decode_yolov8


AIS_BENCH_PATHS = [
    "/usr/local/Ascend/ascend-toolkit/latest/tools/ais_bench",
    "/usr/local/Ascend/ascend-toolkit/tools/ais_bench",
    os.path.expanduser("~/Ascend/ascend-toolkit/latest/tools/ais_bench"),
    "/opt/Ascend/ascend-toolkit/latest/tools/ais_bench",
]


def _ensure_ais_bench_path():
    for p in AIS_BENCH_PATHS:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


class AscendOMDetector(object):
    backend_name = "ascend-om"

    def __init__(self, model_path, labels, input_size=640, conf_threshold=0.35,
                 iou_threshold=0.5, max_detections=30, class_agnostic=True,
                 device_id=0):
        if not os.path.exists(model_path):
            raise RuntimeError("OM 模型不存在：%s" % model_path)
        _ensure_ais_bench_path()
        try:
            from ais_bench.infer.interface import InferSession
        except ImportError as e:
            raise RuntimeError(
                "未找到 CANN ais_bench。请确认已在板端执行：\n"
                "  source /usr/local/Ascend/ascend-toolkit/set_env.sh\n"
                "  npu-smi info\n"
                "并将 ais_bench 加入 PYTHONPATH；错误：%s" % e)

        self.model_path = str(model_path)
        self.labels = list(labels)
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.class_agnostic = bool(class_agnostic)
        self.device_id = int(device_id)
        self.session = None
        self._lock = threading.Lock()
        try:
            self.session = InferSession(device_id=self.device_id, model_path=self.model_path)
        except TypeError:
            # 兼容旧版参数顺序
            self.session = InferSession(self.device_id, self.model_path)
        if self.session is None:
            raise RuntimeError("InferSession 初始化失败：%s" % model_path)

        self._acl = None
        self._acl_context = None
        self._bound_tid = None
        self._capture_acl_context()
        self._bind_acl_context()

    # ------------------------------------------------------------------
    def _capture_acl_context(self):
        """在当前（创建 session 的）线程里抓住 ACL context，供其他线程复用。"""
        try:
            import acl
            self._acl = acl
        except Exception:
            self._acl = None
            return
        ctx = None
        # 少数 ais_bench 版本把 context 挂在 session 上
        for attr in ("context", "_context", "context_"):
            v = getattr(self.session, attr, None)
            if v and not isinstance(v, (tuple, list)):
                ctx = v
                break
        if not ctx:
            try:
                got = self._acl.rt.get_context()
                if isinstance(got, (tuple, list)):
                    ctx = got[0] if got and got[0] else None
                else:
                    ctx = got or None
            except Exception:
                ctx = None
        self._acl_context = ctx

    def _bind_acl_context(self):
        """ACL context 线程私有：切到新线程后必须重新 set_device + set_context。"""
        if self._acl is None or self._acl_context is None:
            return
        tid = threading.get_ident()
        if self._bound_tid == tid:
            return
        try:
            self._acl.rt.set_device(self.device_id)
        except Exception:
            pass
        try:
            self._acl.rt.set_context(self._acl_context)
            self._bound_tid = tid
        except Exception:
            self._bound_tid = None

    def _run(self, tensor):
        # ais_bench 不同版本的 infer 签名略有差异，逐级兼容
        try:
            return self.session.infer([tensor], mode="static")
        except TypeError:
            pass
        try:
            return self.session.infer([tensor])
        except TypeError:
            pass
        return self.session.infer([tensor], input_shape="images:%d,3,%d,%d" %
                                  (1, self.input_size, self.input_size))

    def infer(self, image):
        st = time.time()
        blob, ratio, pad = letterbox(image, self.input_size)
        tensor = blob[None, ...].astype(np.float32)
        # 同一个 context 可能被推理线程和单图接口线程同时使用，串行化更稳
        with self._lock:
            self._bind_acl_context()
            outputs = self._run(tensor)
        if isinstance(outputs, dict):
            outputs = list(outputs.values())
        if isinstance(outputs, (list, tuple)):
            if len(outputs) == 0:
                return [], (time.time() - st) * 1000.0
            output = outputs[0]
        else:
            output = outputs
        dets = decode_yolov8(output, image.shape[:2], ratio, pad,
                             conf_thres=self.conf_threshold,
                             iou_thres=self.iou_threshold,
                             labels=self.labels,
                             max_detections=self.max_detections,
                             class_agnostic=self.class_agnostic)
        return dets, (time.time() - st) * 1000.0

    def info(self):
        return {
            "available": True,
            "backend": self.backend_name,
            "device": "Ascend 310B4 NPU",
            "device_id": self.device_id,
            "model": os.path.basename(self.model_path),
            "model_path": self.model_path,
            "input_size": self.input_size,
            "labels": self.labels,
            "provider": "CANN ais_bench",
        }

    def close(self):
        try:
            self.session.free_resource()
        except Exception:
            pass
        self.session = None
