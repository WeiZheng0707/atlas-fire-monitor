# -*- coding: utf-8 -*-
"""构建 Atlas 板端运行包（显式排除数据集、训练代码和训练产物）。

用法：
    python deployment/package_for_board.py --model models/fire_yolov8n.om
    python deployment/package_for_board.py --allow-missing-model
"""
import os
import sys
import json
import stat
import time
import shutil
import hashlib
import argparse
import tarfile
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = "atlas_fire"
EXCLUDED_NAMES = {
    "dataset", "runs", "training", "tests", "tools", ".venv", "__pycache__",
    ".git", ".idea", ".vscode", "data", "dist", "artifacts",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect():
    files = []

    def add(path):
        p = Path(path)
        if p.is_file():
            files.append(p)

    for d in ("app", "deployment"):
        for dirpath, dirnames, filenames in os.walk(ROOT / d):
            dirnames[:] = [x for x in dirnames if x not in EXCLUDED_NAMES]
            for name in filenames:
                if name.endswith((".pyc", ".pyo")) or name.startswith("."):
                    continue
                add(Path(dirpath) / name)
    add(ROOT / "configs" / "app.json")
    add(ROOT / "requirements" / "board.txt")
    for name in ("README.md",):
        add(ROOT / name)
    for p in (ROOT / "models").glob("*"):
        if p.is_file() and p.name != ".gitkeep":
            add(p)
    for p in (ROOT / "docs").glob("*"):
        if p.is_file():
            add(p)
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="额外加入的 .om/.onnx 模型")
    parser.add_argument("--with-samples", action="store_true",
                        help="加入 samples/ 演示图片（默认不加）")
    parser.add_argument("--allow-missing-model", action="store_true")
    parser.add_argument("--lean", action="store_true",
                        help="精简：去掉 docs/ 里的图片和课程报告，板端只需能查操作说明")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.model:
        src = Path(args.model)
        if not src.is_absolute():
            src = ROOT / src
        if not src.exists():
            raise SystemExit("模型不存在：%s" % src)
        dst = ROOT / "models" / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(str(src), str(dst))
        print("已加入模型：models/%s" % src.name)

    files = collect()
    if args.with_samples:
        for p in (ROOT / "samples").glob("*"):
            if p.is_file():
                files.append(p)
    files = sorted(set(files))

    dropped = []
    if args.lean:
        # 板端只要代码、配置、模型和行为说明；截图与课程报告留在 PC 上
        keep = {"docs/板端部署.md", "docs/操作手册.md"}
        lean_files = []
        for p in files:
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            if rel.startswith("docs/") and rel not in keep:
                dropped.append(rel)
                continue
            lean_files.append(p)
        files = lean_files

    model_files = [p for p in files if p.suffix.lower() in (".om", ".onnx")]
    if not model_files and not args.allow_missing_model:
        raise SystemExit(
            "没有模型文件。请先训练导出 models/fire_yolov8n.onnx，"
            "或在板端生成 .om；调试打包可加 --allow-missing-model。")

    total = sum(p.stat().st_size for p in files)
    manifest = {
        "package": "atlas_fire_board",
        "build_time": datetime.datetime.now().isoformat(timespec="seconds"),
        "version": "1.0.0",
        "model_files": [str(p.relative_to(ROOT)).replace("\\", "/") for p in model_files],
        "total_size": total,
        "dataset_included": False,
        "excluded": ["dataset/", "runs/", "training/", "tests/", "tools/"]
                    + (["docs/ 图片与课程报告"] if dropped else []),
        "files": [{
            "path": str(p.relative_to(ROOT)).replace("\\", "/"),
            "size": p.stat().st_size,
            "sha256": sha256(p),
        } for p in files],
    }

    print("=" * 74)
    print(" Atlas 板端打包")
    print(" 文件数    : %d" % len(files))
    print(" 原始大小  : %.2f MB" % (total / 1024 / 1024))
    print(" 模型      : %s" % (", ".join(manifest["model_files"]) or "无"))
    print(" 排除      : dataset/ runs/ training/ tests/ tools/ data/")
    if dropped:
        print(" 精简      : 已剔除 docs/ 下 %d 个图片与报告文件" % len(dropped))
    print("=" * 74)
    if args.dry_run:
        for p in files:
            print("  %8.1f KB  %s" % (p.stat().st_size / 1024, p.relative_to(ROOT)))
        return 0

    out_dir = ROOT / "dist"
    out_dir.mkdir(exist_ok=True)
    out = Path(args.output) if args.output else out_dir / (
        "atlas_fire_board_%s.tar.gz" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    def _tar_filter(ti):
        # Windows 文件系统会丢可执行位，这里统一补上脚本权限
        if ti.name.endswith(".sh"):
            ti.mode |= stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        return ti

    with tarfile.open(str(out), "w:gz") as tar:
        for p in files:
            arc = PACKAGE_ROOT + "/" + str(p.relative_to(ROOT)).replace("\\", "/")
            tar.add(str(p), arcname=arc, recursive=False, filter=_tar_filter)
        data = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        import io
        info = tarfile.TarInfo(PACKAGE_ROOT + "/manifest.json")
        info.size = len(data)
        info.mtime = time.time()
        tar.addfile(info, io.BytesIO(data))

    print("已生成：%s" % out)
    print("大小  ：%.2f MB" % (out.stat().st_size / 1024 / 1024))
    print("上传  ：scp %s root@192.168.137.100:/home/root/" % out.name)
    print("板端  ：tar -xzf %s && cd %s && bash deployment/install_board.sh"
          % (out.name, PACKAGE_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
