#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 Ultralytics 从 .pt 直接导出 TensorRT .engine（内部：ONNX → TensorRT）。

依赖与本机一致：CUDA、已安装的 tensorrt Python 包、以及能加载权重所需的 torch 等。
导出默认使用 GPU；请保证 ``LD_LIBRARY_PATH`` 已包含 TensorRT / cuDNN 的 lib（若 import tensorrt 需要）。

示例::

    python export_engine_ultralytics.py best.pt --imgsz 640
    python export_engine_ultralytics.py best.pt --imgsz 640 --half
    python export_engine_ultralytics.py best.pt --imgsz 640 --int8 --data data.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(
        description="Ultralytics: .pt → TensorRT .engine（format=engine）",
    )
    p.add_argument(
        "weights",
        type=Path,
        help="YOLO 权重 .pt",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=None,
        metavar="H [W]",
        help="输入尺寸；一个数=方形；两个数=高 宽（默认沿用 ultralytics 模型内 imgsz）",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=1,
        help="导出 batch（默认 1）",
    )
    p.add_argument(
        "--device",
        type=str,
        default="0",
        help="导出所用 GPU，如 0 或 cuda:0",
    )
    p.add_argument(
        "--half",
        action="store_true",
        help="FP16 引擎（与 int8 互斥）",
    )
    p.add_argument(
        "--int8",
        action="store_true",
        help="INT8 引擎（需 --data）",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="INT8 标定用数据集 YAML（如 coco.yaml / 你自己的 data.yaml）",
    )
    p.add_argument(
        "--workspace",
        type=int,
        default=4,
        help="TensorRT workspace（GiB），与 ultralytics 默认一致",
    )
    p.add_argument(
        "--dynamic",
        action="store_true",
        help="动态轴（与 batch/部署方式有关，按需开启）",
    )
    p.add_argument(
        "--nms",
        action="store_true",
        help="导出带 NMS 的端到端图（视任务/版本支持情况而定）",
    )
    p.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ONNX simplify（默认开启）",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=None,
        help="ONNX opset（默认由 ultralytics 决定）",
    )
    p.add_argument(
        "--task",
        type=str,
        default=None,
        choices=("detect", "segment", "classify", "pose", "obb"),
        help="显式任务类型（猜错任务时可填）",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="更详细的导出日志",
    )
    args = p.parse_args()

    w = args.weights.resolve()
    if not w.is_file():
        print(f"错误: 找不到权重: {w}", file=sys.stderr)
        return 1
    if args.int8 and args.data is None:
        print("错误: --int8 需要 --data /path/to/data.yaml", file=sys.stderr)
        return 1
    if args.int8 and args.half:
        print("错误: 不要同时指定 --int8 与 --half", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误: 未安装 ultralytics，请 pip install ultralytics", file=sys.stderr)
        return 1

    load_kw = {}
    if args.task is not None:
        load_kw["task"] = args.task

    model = YOLO(str(w), **load_kw)

    export_kw: dict = {
        "format": "engine",
        "batch": args.batch,
        "device": args.device,
        "half": args.half,
        "int8": args.int8,
        "workspace": args.workspace,
        "dynamic": args.dynamic,
        "nms": args.nms,
        "simplify": args.simplify,
        "verbose": args.verbose,
    }
    if args.imgsz is not None:
        export_kw["imgsz"] = args.imgsz[0] if len(args.imgsz) == 1 else args.imgsz
    if args.data is not None:
        export_kw["data"] = str(args.data.resolve())
    if args.opset is not None:
        export_kw["opset"] = args.opset

    out = model.export(**export_kw)
    print(f"OK: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
