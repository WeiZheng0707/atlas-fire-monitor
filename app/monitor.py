# -*- coding: utf-8 -*-
"""火情告警与证据管理（双类别：火焰 / 烟雾，分级告警）。

设计要点
--------
1. **分级**：烟雾 → 一级（早期预警），火焰 → 二级（火情告警）。
   同一帧出现火焰时等级直接取二级，烟雾预警自动升级。
2. **按类阈值**：烟雾在真实场景里形态更模糊，通常给更低阈值、
   更严格的连续帧确认，避免蒸汽/雾霾/逆光被误报。
3. **连续帧确认**：`confirm_frames` 要求连续 N 帧都过阈值才算命中，
   单帧抖动不触发告警（板端 5~15 FPS，3 帧约 0.2~0.6 秒）。
4. **保持与冷却**：命中消失后保持 hold_seconds 再回落，冷却期内不重复报。
5. **升级立即上报**：烟雾预警期间出现明火，不受冷却限制直接升到二级。
"""
import os
import json
import time
import threading
import datetime
from pathlib import Path

# 类别 -> 告警等级（数字越大越严重）
CLASS_LEVEL = {"fire": 2, "smoke": 1}
LEVEL_NAME = {0: "安全", 1: "烟雾预警", 2: "火情告警"}
LEVEL_KEY = {0: "safe", 1: "warn", 2: "fire"}

# 已知类别的默认策略：阈值 / 连续确认帧数
CLASS_DEFAULTS = {
    "fire": {"threshold": 0.45, "confirm": 1},
    "smoke": {"threshold": 0.35, "confirm": 3},
}


def _log(msg):
    print("[alarm] %s" % msg, flush=True)


class FireMonitor(object):
    def __init__(self, cfg, save_dir):
        self.enabled = bool(cfg.get("enabled", True))
        self.class_names = [str(x).lower() for x in cfg.get("class_names", ["fire"])]
        self.cooldown = float(cfg.get("cooldown_seconds", 10))
        self.hold_seconds = float(cfg.get("hold_seconds", 6))
        self.escalate = bool(cfg.get("escalate", True))
        self.save_snapshots = bool(cfg.get("save_snapshots", True))
        self.max_events = int(cfg.get("max_events", 100))
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 兼容旧配置：只给 threshold 时，火焰（或唯一类别）沿用该阈值，
        # 其余类别用 CLASS_DEFAULTS 的推荐值。
        base_thr = float(cfg.get("threshold", 0.5))
        thr_cfg = cfg.get("thresholds") or {}
        cf_cfg = cfg.get("confirm_frames") or {}
        self.thresholds = {}
        self.confirm = {}
        for name in self.class_names:
            default = CLASS_DEFAULTS.get(name, {})
            fallback = base_thr if (name == "fire" or len(self.class_names) == 1) \
                else default.get("threshold", base_thr)
            self.thresholds[name] = float(thr_cfg.get(name, fallback))
            self.confirm[name] = max(1, int(
                cf_cfg.get(name, default.get("confirm", 1))))

        self._lock = threading.RLock()
        self._events = []
        self._last_alarm_ts = 0.0
        self._alarm_active = False
        self._alarm_id = None
        self._alarm_level = 0        # 当前处于告警的等级
        self._level_ts = 0.0         # 最后一次命中的时间（用于 hold）
        self._last_score = 0.0
        self._streak = dict((n, 0) for n in self.class_names)
        self._counts = dict((n, 0) for n in self.class_names)
        self._load()

    # ------------------------------------------------------------------
    def _index(self):
        return self.save_dir / "index.json"

    def _load(self):
        try:
            items = json.loads(self._index().read_text(encoding="utf-8"))
            valid = []
            for it in items if isinstance(items, list) else []:
                if not isinstance(it, dict):
                    continue
                image = it.get("image")
                if image and not (self.save_dir / os.path.basename(image)).exists():
                    continue
                valid.append(it)
            self._events = valid[:self.max_events]
        except Exception:
            self._events = []

    def _save_index(self):
        tmp = self._index().with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._events, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(str(tmp), str(self._index()))

    def _new_id(self, prefix=""):
        now = datetime.datetime.now()
        return prefix + now.strftime("%Y%m%d_%H%M%S_") + \
            ("%03d" % (int(time.time() * 1000) % 1000))

    def _write_event(self, jpeg_bytes, score, count, label="火焰", prefix="",
                     level=2, counts=None):
        eid = self._new_id(prefix)
        filename = eid + ".jpg"
        (self.save_dir / filename).write_bytes(jpeg_bytes)
        item = {
            "id": eid,
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": time.time(),
            "score": round(float(score), 4),
            "count": int(count),
            "image": "/api/events/" + filename,
            "label": label,
            "level": int(level),
            "level_name": LEVEL_NAME.get(int(level), "安全"),
            "counts": dict(counts or {}),
        }
        self._events.insert(0, item)
        self._events = self._events[:self.max_events]
        self._save_index()
        return item

    # ------------------------------------------------------------------
    def _event_label(self, hits):
        """根据命中的类别生成证据标签与文件名前缀。"""
        names = [n for n in self.class_names if hits.get(n)]
        if "fire" in names and "smoke" in names:
            return "火焰+烟雾", "fire_smoke_"
        if "fire" in names:
            return "火焰", "fire_"
        if "smoke" in names:
            return "烟雾", "smoke_"
        return "火情", ""

    def observe(self, detections, jpeg_bytes):
        """输入当前帧检测结果与标注图 JPEG，返回告警状态 dict。"""
        now = time.time()
        scores = {}
        counts = dict((n, 0) for n in self.class_names)
        for d in detections or []:
            label = str(d.get("label", "")).lower()
            if label not in counts:
                continue
            counts[label] += 1
            s = float(d.get("score", 0) or 0)
            if s > scores.get(label, 0.0):
                scores[label] = s

        with self._lock:
            # 1) 逐类连续帧确认
            hits = {}
            for name in self.class_names:
                if scores.get(name, 0.0) >= self.thresholds[name]:
                    self._streak[name] += 1
                else:
                    self._streak[name] = 0
                hits[name] = (self._streak[name] >= self.confirm[name]
                              and self._streak[name] > 0)

            # 2) 本帧确认等级
            confirmed = 0
            for name in self.class_names:
                if hits[name]:
                    confirmed = max(confirmed, CLASS_LEVEL.get(name, 2))

            if confirmed > 0:
                self._level_ts = now
            elif self._alarm_active and now - self._level_ts > self.hold_seconds:
                # 持续无命中超过保持时间 -> 回落
                self._alarm_active = False
                self._alarm_level = 0
                for name in self.class_names:
                    self._streak[name] = 0

            display_level = confirmed
            if display_level == 0 and self._alarm_active:
                display_level = self._alarm_level

            # 3) 是否产生新告警
            new_alarm = False
            trigger = None
            event = None
            if self.enabled and confirmed > 0:
                cooled = (now - self._last_alarm_ts) >= self.cooldown
                upgraded = (self.escalate and self._alarm_active
                            and confirmed > self._alarm_level)
                if cooled or upgraded:
                    trigger = "fire" if confirmed >= 2 else "smoke"
                    label, prefix = self._event_label(hits)
                    score = max([scores.get(n, 0.0) for n in self.class_names
                                 if hits.get(n)] or [0.0])
                    if self.save_snapshots and jpeg_bytes:
                        event = self._write_event(
                            jpeg_bytes, score, sum(counts.values()), label, prefix,
                            level=confirmed, counts=counts)
                    self._last_alarm_ts = now
                    self._alarm_active = True
                    self._alarm_level = confirmed
                    self._last_score = score
                    self._alarm_id = event["id"] if event else "alarm_%d" % int(now)
                    new_alarm = True
                    _log("%s #%s：火焰 %d 个 / 烟雾 %d 个，最高分 %.3f，截图=%s" %
                         (LEVEL_NAME[confirmed], self._alarm_id, counts.get("fire", 0),
                          counts.get("smoke", 0), score,
                          event["image"] if event else "关闭"))

            self._counts = counts
            threshold = self.thresholds.get(trigger) if trigger else max(
                self.thresholds.values() or [0.0])
            return {
                "enabled": self.enabled,
                "active": bool(self._alarm_active),
                "new": bool(new_alarm),
                "id": self._alarm_id,
                "level": int(display_level),
                "level_name": LEVEL_NAME.get(display_level, "安全"),
                "level_key": LEVEL_KEY.get(display_level, "safe"),
                "trigger": trigger,
                "label": LEVEL_NAME.get(display_level, "安全"),
                "score": round(float(scores.get(trigger, 0.0) if trigger else
                                     max(scores.values() or [0.0])), 4),
                "threshold": round(float(threshold or 0.0), 4),
                "thresholds": dict(self.thresholds),
                "counts": dict(counts),
                "streak": dict(self._streak),
                "confirm_frames": dict(self.confirm),
                "cooldown": self.cooldown,
                "hold": self.hold_seconds,
                "last_alarm_ts": self._last_alarm_ts,
                "class_names": self.class_names,
            }

    # ------------------------------------------------------------------
    def save_manual(self, jpeg_bytes):
        if not jpeg_bytes:
            return None
        with self._lock:
            return self._write_event(jpeg_bytes, 0.0, 0, "手动截图",
                                     prefix="manual_", level=0)

    def clear(self):
        with self._lock:
            self._alarm_active = False
            self._alarm_id = None
            self._alarm_level = 0
            for name in self.class_names:
                self._streak[name] = 0
            return self.snapshot()

    def snapshot(self):
        with self._lock:
            level = self._alarm_level if self._alarm_active else 0
            return {"active": self._alarm_active,
                    "id": self._alarm_id,
                    "level": int(level),
                    "level_name": LEVEL_NAME.get(level, "安全"),
                    "level_key": LEVEL_KEY.get(level, "safe"),
                    "score": self._last_score,
                    "threshold": self.thresholds.get("fire",
                                                     max(self.thresholds.values() or [0.0])),
                    "thresholds": dict(self.thresholds),
                    "confirm_frames": dict(self.confirm),
                    "counts": dict(self._counts),
                    "cooldown": self.cooldown,
                    "last_alarm_ts": self._last_alarm_ts}

    def events(self):
        with self._lock:
            return list(self._events)
