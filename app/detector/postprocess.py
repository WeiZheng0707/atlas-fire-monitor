# -*- coding: utf-8 -*-
"""YOLOv8 / YOLOv5 输出解码、NMS 与原图坐标映射。"""
import numpy as np

from .preprocess import unletterbox_box


def _as_2d(output):
    arr = np.asarray(output)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError("不支持的模型输出形状：%s" % (arr.shape,))
    return arr


def _xywh2xyxy(boxes):
    out = np.empty_like(boxes, dtype=np.float32)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def iou_matrix(boxes, box):
    x1 = np.maximum(boxes[:, 0], box[0])
    y1 = np.maximum(boxes[:, 1], box[1])
    x2 = np.minimum(boxes[:, 2], box[2])
    y2 = np.minimum(boxes[:, 3], box[3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_b = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def nms(boxes, scores, iou_threshold=0.5, max_detections=30):
    """纯 numpy NMS，返回保留下标。输入 xyxy。"""
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_detections:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_matrix(boxes[rest], boxes[i])
        order = rest[ious <= iou_threshold]
    return keep


def decode_yolov8(output, orig_shape, ratio, pad, conf_thres=0.35,
                  iou_thres=0.5, labels=None, max_detections=30,
                  class_agnostic=False):
    """把 YOLO 输出解码为 [{label,class_id,score,box:[x1,y1,x2,y2]}]。

    支持：
      - YOLOv8: [1, 4+nc, N] 或 [1, N, 4+nc]
      - YOLOv5: [1, N, 5+nc]（含 objectness）
    """
    arr = _as_2d(output).astype(np.float32)
    labels = labels or ["fire"]
    nc = len(labels)

    # 判断布局：列数 = 4+nc (v8) 或 5+nc (v5)
    if arr.shape[0] in (4 + nc, 5 + nc) and arr.shape[1] not in (4 + nc, 5 + nc):
        arr = arr.T
    cols = arr.shape[1]
    if cols == 4 + nc:
        boxes_xywh = arr[:, :4]
        class_scores = arr[:, 4:]
        objectness = np.ones((arr.shape[0], 1), dtype=np.float32)
    elif cols == 5 + nc:
        boxes_xywh = arr[:, :4]
        objectness = arr[:, 4:5]
        class_scores = arr[:, 5:]
    else:
        raise ValueError("输出列数 %d 与类别数 %d 不匹配（应为 %d 或 %d）" %
                         (cols, nc, 4 + nc, 5 + nc))

    class_ids = np.argmax(class_scores, axis=1)
    scores = objectness[:, 0] * class_scores[np.arange(arr.shape[0]), class_ids]
    mask = scores >= float(conf_thres)
    if not np.any(mask):
        return []
    boxes = _xywh2xyxy(boxes_xywh[mask])
    scores = scores[mask]
    class_ids = class_ids[mask]

    detections = []
    if class_agnostic:
        keep = nms(boxes, scores, iou_thres, max_detections)
        for i in keep:
            detections.append((boxes[i], scores[i], int(class_ids[i])))
    else:
        for cid in np.unique(class_ids):
            idx = np.where(class_ids == cid)[0]
            keep = nms(boxes[idx], scores[idx], iou_thres, max_detections)
            for i in keep:
                j = idx[i]
                detections.append((boxes[j], scores[j], int(class_ids[j])))
        detections.sort(key=lambda t: t[1], reverse=True)
        detections = detections[:max_detections]

    out = []
    for box, score, cid in detections:
        x1, y1, x2, y2 = unletterbox_box(box, ratio, pad, orig_shape)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        label = labels[cid] if cid < len(labels) else "class_%d" % cid
        out.append({
            "label": label,
            "class_id": int(cid),
            "score": round(float(score), 4),
            "box": [round(float(x1), 1), round(float(y1), 1),
                    round(float(x2), 1), round(float(y2), 1)],
        })
    return out
