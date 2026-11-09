# -*- coding: utf-8 -*-
"""YOLO 预处理：BGR -> letterbox -> RGB/NCHW/float32。"""
import cv2
import numpy as np


def letterbox(image, new_size=640, color=(114, 114, 114)):
    """等比例缩放并填充到 new_size × new_size。

    返回：
        blob       [3, H, W] float32，RGB，0~1
        ratio      缩放比例
        (dw, dh)   左右/上下填充像素
    """
    if isinstance(new_size, int):
        new_size = (new_size, new_size)
    new_w, new_h = int(new_size[0]), int(new_size[1])
    h, w = image.shape[:2]
    ratio = min(new_w / float(w), new_h / float(h))
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    dw, dh = (new_w - nw) // 2, (new_h - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(rgb, (2, 0, 1))
    return np.ascontiguousarray(blob), ratio, (dw, dh)


def unletterbox_box(box, ratio, pad, orig_shape):
    """把输入尺寸下的 xyxy 框映射回原图坐标。"""
    x1, y1, x2, y2 = [float(v) for v in box]
    dw, dh = pad
    x1 = (x1 - dw) / ratio
    x2 = (x2 - dw) / ratio
    y1 = (y1 - dh) / ratio
    y2 = (y2 - dh) / ratio
    h, w = orig_shape[:2]
    x1 = max(0.0, min(w - 1.0, x1))
    y1 = max(0.0, min(h - 1.0, y1))
    x2 = max(0.0, min(w - 1.0, x2))
    y2 = max(0.0, min(h - 1.0, y2))
    return x1, y1, x2, y2
