#!/usr/bin/env python3
"""ImageNet 图像分类 — 使用 MobileNetV2 预训练模型"""
# ── 0. 环境变量（必须在 import keras 之前） ──
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["KERAS_BACKEND"] = "tensorflow"

import subprocess
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
# 禁用 TF Python 日志
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ── 1. 自动检测/安装依赖 ──
REQUIRED = {"keras", "pillow", "numpy"}


def ensure_dependencies():
    missing = set()
    for pkg in REQUIRED:
        try:
            __import__(pkg.replace("pillow", "PIL"))
        except ImportError:
            missing.add(pkg)

    if missing:
        print(f"安装缺失依赖: {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing, "-q"]
        )


ensure_dependencies()
import keras

# ── 2. 加载图片 ──
from PIL import Image
import numpy as np

IMAGE_DIR = Path(__file__).parent / "book_images"
IMG_SIZE = (224, 224)

image_paths = sorted({
    p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    for p in IMAGE_DIR.glob(ext)
    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
})

if not image_paths:
    print(f"未在 {IMAGE_DIR}/ 中找到图片文件")
    sys.exit(1)

num_images = len(image_paths)
x = np.empty((num_images, *IMG_SIZE, 3), dtype="float32")

for i, path in enumerate(image_paths):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    x[i] = np.asarray(img, dtype="float32") / 255.0

# MobileNetV2 预处理: [0,1] -> [-1,1]
x = x * 2 - 1

# ── 3. 加载预训练模型 + 预测 + 解码 ──
model = keras.applications.MobileNetV2(weights="imagenet")
preds = model.predict(x, verbose=0)

top5 = keras.applications.mobilenet_v2.decode_predictions(preds, top=5)

for i, path in enumerate(image_paths):
    print(f"[{i + 1}/{num_images}] {path.name}")
    for rank, (_, label, score) in enumerate(top5[i], 1):
        print(f"  {rank:2d}. {label} ({score * 100:.1f}%)")
    print()
