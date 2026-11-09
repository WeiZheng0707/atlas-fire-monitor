# -*- coding: utf-8 -*-
"""检测结果可视化（中文标签、告警角标）。"""
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
LABEL_CN = {"fire": "火焰", "smoke": "烟雾"}
# 火焰红 / 烟雾蓝，同一画面里两类目标一眼可分
COLORS = {
    "fire": {"line": (255, 78, 35, 255), "glow": (255, 140, 60, 20),
             "text": (255, 227, 196, 255), "bg": (22, 12, 8, 225)},
    "smoke": {"line": (110, 200, 235, 255), "glow": (120, 200, 235, 16),
              "text": (214, 240, 255, 255), "bg": (8, 18, 26, 225)},
}
_FONT_CACHE = {}


def get_font(size):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = [
        ROOT / "app" / "web" / "img" / "wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/msyh.ttc",
    ]
    font = None
    for p in candidates:
        if not os.path.exists(str(p)):
            continue
        try:
            font = ImageFont.truetype(str(p), size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _rounded(draw, box, radius, **kwargs):
    try:
        draw.rounded_rectangle(box, radius=radius, **kwargs)
    except AttributeError:
        draw.rectangle(box, **kwargs)


def annotate(image, detections, backend="", model_note=""):
    """在 BGR 图上画框，返回 BGR numpy 图。"""
    if image is None:
        return image
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil, "RGBA")
    font = get_font(18)
    small = get_font(12)

    counts = {"fire": 0, "smoke": 0}
    for d in detections or []:
        cls = str(d.get("label", "")).lower()
        if cls in counts:
            counts[cls] += 1

    # 先画左上角状态条与计数标签，再画检测框：
    # 否则靠近画面顶部的目标，其类别文字会被状态条盖住。
    if counts["fire"]:
        dot = (255, 78, 35, 255)
    elif counts["smoke"]:
        dot = (110, 200, 235, 255)
    else:
        dot = (120, 190, 150, 255)
    _rounded(draw, [12, 12, 470, 44], 6, fill=(10, 13, 18, 175))
    draw.ellipse([22, 22, 31, 31], fill=dot)
    draw.text((39, 20), "ATLAS 200I DK A2  ·  YOLOv8", font=small,
              fill=(255, 210, 178, 255))
    if backend:
        label = str(backend).upper()
        bbox = draw.textbbox((0, 0), label, font=small)
        tw = bbox[2] - bbox[0]
        draw.text((458 - tw, 20), label, font=small, fill=(140, 190, 210, 225))

    # 类别计数小标签（只有出现目标时才画）
    # 画面够宽就排在状态条右侧，避免和画面顶部目标的类别文字叠在一起
    chip_x = 478 if pil.width >= 620 else 12
    chip_y = 12 if pil.width >= 620 else 50
    for cls in ("fire", "smoke"):
        n = counts[cls]
        if not n:
            continue
        text = "%s %d" % (LABEL_CN[cls], n)
        bbox = draw.textbbox((0, 0), text, font=small)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        right = min(pil.width - 8, chip_x + tw + 16)
        _rounded(draw, [chip_x, chip_y, right, chip_y + th + 9], 4,
                 fill=COLORS[cls]["bg"])
        _rounded(draw, [chip_x, chip_y, right, chip_y + th + 9], 4,
                 outline=COLORS[cls]["line"], width=1)
        draw.text((chip_x + 8, chip_y + 3), text, font=small,
                  fill=COLORS[cls]["text"])
        chip_x = right + 8

    for d in detections or []:
        x1, y1, x2, y2 = [int(round(v)) for v in d.get("box", [0, 0, 0, 0])]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(pil.width - 1, x2), min(pil.height - 1, y2)
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        cls = str(d.get("label", "")).lower()
        palette = COLORS.get(cls, COLORS["fire"])
        color, glow = palette["line"], palette["glow"]
        _rounded(draw, [x1 - 3, y1 - 3, x2 + 3, y2 + 3], 6, outline=glow, width=6)
        _rounded(draw, [x1, y1, x2, y2], 4, outline=color, width=3)
        # 四角
        L = max(10, min(22, min(w, h) // 5))
        for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                               (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            draw.line([cx, cy, cx + dx * L, cy], fill=color, width=4)
            draw.line([cx, cy, cx, cy + dy * L], fill=color, width=4)
        name = LABEL_CN.get(cls, d.get("label", "fire"))
        text = "%s %.0f%%" % (name, float(d.get("score", 0)) * 100)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ty = y1 - th - 12
        if ty < 52:                     # 顶部留给状态条与计数标签
            # 框上沿本来就贴着画面顶部（大面积烟雾常见）时，y1+4 仍会压到状态条，
            # 因此下限钳到 52，保证标签一定落在状态条下方
            ty = max(y1 + 4, 52)
        ty = min(ty, pil.height - th - 12)
        _rounded(draw, [x1, ty, x1 + tw + 16, ty + th + 9], 4,
                 fill=palette["bg"])
        draw.text((x1 + 8, ty + 4), text, font=font, fill=palette["text"])

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
