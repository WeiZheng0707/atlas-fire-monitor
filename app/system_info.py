# -*- coding: utf-8 -*-
"""Atlas 板端环境信息（npu-smi / CANN / 内存），带缓存，避免频繁执行命令。"""
import os
import re
import sys
import time
import shutil
import socket
import platform
import subprocess
import threading
from pathlib import Path


def _run(cmd, timeout=3):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout, text=True, errors="replace")
        return p.returncode, (p.stdout or "").strip()
    except Exception as e:
        return -1, str(e)


class SystemInfo(object):
    def __init__(self, ascend_home="/usr/local/Ascend/ascend-toolkit",
                 cache_seconds=10):
        self.ascend_home = str(ascend_home)
        self.cache_seconds = max(1, int(cache_seconds))
        self._lock = threading.Lock()
        self._cached = None
        self._cached_at = 0.0

    def _cann_version(self):
        candidates = [
            Path(self.ascend_home) / "latest" / "version.cfg",
            Path(self.ascend_home) / "latest" / "acllib" / "version.info",
            Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
        ]
        for p in candidates:
            try:
                if p.exists():
                    text = p.read_text(encoding="utf-8", errors="replace")
                    m = re.search(r"(?:version|Version)\s*[=:]\s*([^\s]+)", text)
                    return m.group(1) if m else text.strip().splitlines()[0][:80]
            except Exception:
                continue
        return None

    def _memory(self):
        if os.name == "nt":
            return None
        rc, out = _run(["free", "-m"])
        if rc != 0:
            return None
        for line in out.splitlines():
            if line.lower().startswith("mem:"):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        return {"total_mb": int(parts[1]), "used_mb": int(parts[2]),
                                "free_mb": int(parts[3])}
                    except ValueError:
                        return None
        return None

    def _collect(self):
        npu_cmd = shutil.which("npu-smi")
        npu_available = bool(npu_cmd)
        npu_output = ""
        if npu_cmd:
            rc, out = _run([npu_cmd, "info"], timeout=4)
            npu_output = out[:1200]
        cann = self._cann_version()
        ascend_dir = Path(self.ascend_home).exists() or Path("/usr/local/Ascend").exists()
        is_atlas = npu_available or ascend_dir
        return {
            "hostname": socket.gethostname(),
            "platform": "Atlas 200I DK A2" if is_atlas else "PC / 无 NPU 环境",
            "is_atlas": is_atlas,
            "arch": platform.machine(),
            "os": platform.platform(),
            "python": sys.version.split()[0],
            "npu_available": npu_available,
            "npu_output": npu_output,
            "cann_version": cann,
            "ascend_dir": str(Path(self.ascend_home)),
            "memory": self._memory(),
            "timestamp": time.time(),
        }

    def snapshot(self, force=False):
        with self._lock:
            now = time.time()
            if force or self._cached is None or now - self._cached_at > self.cache_seconds:
                self._cached = self._collect()
                self._cached_at = now
            return dict(self._cached)
