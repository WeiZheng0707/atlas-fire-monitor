# -*- coding: utf-8 -*-
"""ONNX Runtime CPU 推理后端（PC 调试 / 板端兜底）。"""
import os
import time

import numpy as np

from .preprocess import letterbox
from .postprocess import decode_yolov8


class OnnxDetector(object):
    backend_name = "onnx-cpu"

    def __init__(self, model_path, labels, input_size=640, conf_threshold=0.35,
                 iou_threshold=0.5, max_detections=30, class_agnostic=True,
                 num_threads=0):
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError(
                "未安装 onnxruntime。板端执行：\n"
                "  pip3 install onnxruntime\n"
                "或使用 OM 后端（需 CANN 的 ais_bench）。")
        self.model_path = str(model_path)
        self.labels = list(labels)
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.class_agnostic = bool(class_agnostic)

        opts = ort.SessionOptions()
        if num_threads:
            opts.intra_op_num_threads = int(num_threads)
        self.session = ort.InferenceSession(
            self.model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self._model_shape = shape
        self._warmup()

    def _warmup(self):
        dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
        try:
            outputs = self.session.run(None, {self.input_name: dummy})
        except Exception as e:
            raise RuntimeError("ONNX 模型输入尺寸与配置不匹配：%s（模型输入：%s）" %
                               (e, self._model_shape))
        self._check_output_classes(outputs[0])

    def _check_output_classes(self, output):
        """加载时就校验类别数，避免 labels.txt 与模型不一致时静默解码错位。

        本工程导出的是 YOLOv8，输出为 [1, 4+nc, N]（或转置）。取较小的那一维当
        通道维（N 通常是 8400 级别），再反推模型类别数。
        """
        nc = len(self.labels)
        arr = np.asarray(output)
        if arr.ndim < 2:
            return
        a, b = int(arr.shape[-2]), int(arr.shape[-1])
        cols = min(a, b)
        model_nc = cols - 4
        if model_nc != nc:
            raise RuntimeError(
                "模型类别数与 models/labels.txt 不一致：模型输出形状 %s（按 YOLOv8 "
                "推断为 %d 类），labels 共 %d 类。请确认 labels.txt 与训练时的类别"
                "顺序、数量完全一致。" % (tuple(arr.shape), max(model_nc, 0), nc))

    def infer(self, image):
        st = time.time()
        blob, ratio, pad = letterbox(image, self.input_size)
        outputs = self.session.run(None, {self.input_name: blob[None, ...]})
        dets = decode_yolov8(outputs[0], image.shape[:2], ratio, pad,
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
            "device": "CPU",
            "model": os.path.basename(self.model_path),
            "model_path": self.model_path,
            "input_size": self.input_size,
            "labels": self.labels,
            "provider": "onnxruntime/CPUExecutionProvider",
        }

    def close(self):
        self.session = None
