#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 **TensorRT 引擎（.engine）** 在视频上检测，写出带框的可视化视频。

实现与 ``detect_video.py`` 相同，区别为：面向 ``yolo_to_tensorrt.py`` 导出的 ``.engine``、
默认使用 GPU（``--device 0``）。仍通过 **Ultralytics** 加载引擎并 ``predict``。

**imgsz**：须与导出该 ``.engine`` 时使用的 ``--imgsz`` / ``--each-imgsz`` 一致，否则结果可能
异常；请显式传入 ``--imgsz``（如 ``640`` 或 ``640 320``）。

依赖: 见本目录 ``requirements.txt`` 中 **TensorRT** 说明；Python 需能 ``import tensorrt``。

结束时会打印 **全流程 FPS**（墙钟）与 ultralytics 的 **pre/infer/post 毫秒**（均值/帧）。

示例::

    python detect_video_tensorrt.py best_imgsz640.engine clip.mp4 -o out.mp4 --imgsz 640
    python detect_video_tensorrt.py model.engine wide.mp4 --conf 0.35 --imgsz 640 320
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import cv2

_FFMPEG_CENTER_SQUARE_VF = (
    r"crop=min(iw\,ih):min(iw\,ih):(iw-min(iw\,ih))/2:(ih-min(iw\,ih))/2"
)


def _make_center_square_video(
    in_path: Path,
    out_path: Path,
) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "未找到 ffmpeg。请安装: sudo apt install ffmpeg（或自行为 PATH 安装）。",
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(in_path),
        "-vf",
        _FFMPEG_CENTER_SQUARE_VF,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-an",
        str(out_path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        out_path.unlink(missing_ok=True)
        err = (p.stderr or p.stdout or "").strip()
        raise RuntimeError(f"ffmpeg 失败: {err or 'returncode %s' % p.returncode}")


def _print_speed_stats(
    n_frames: int,
    wall_s: float,
    pre: List[float],
    inf: List[float],
    post: List[float],
) -> None:
    if n_frames <= 0 or wall_s <= 0:
        return
    print(
        f"速度: 全流程约 {n_frames / wall_s:.2f} FPS "
        f"（墙钟 {wall_s:.2f} s，含视频解码+推理+画框+写盘）",
        flush=True,
    )
    if not inf:
        return
    p_ms = sum(pre) / len(pre) if pre else 0.0
    i_ms = sum(inf) / len(inf)
    o_ms = sum(post) / len(post) if post else 0.0
    tot = p_ms + i_ms + o_ms
    if tot > 0:
        print(
            f"      模型计时(均值/帧，ultralytics): pre {p_ms:.1f} ms + "
            f"inference {i_ms:.1f} ms + post {o_ms:.1f} ms = {tot:.1f} ms "
            f"（仅推理链路，约 {1000.0 / tot:.1f} FPS）",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TensorRT .engine 视频检测可视化：读入引擎与视频，写出绘制检测框后的视频。",
    )
    parser.add_argument(
        "model",
        type=str,
        help="TensorRT 引擎路径（.engine），如 yolo_to_tensorrt.py 导出的 best_imgsz640.engine",
    )
    parser.add_argument(
        "source",
        type=str,
        help="输入视频路径",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="输出视频路径；默认与输入同目录，文件名为 <原名>_trt_detect.mp4",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值（默认 0.25）",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU 阈值",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="设备：TRT 一般为 GPU 索引，默认 0；若需 CPU 可写 cpu（通常不可用 .engine）",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=None,
        metavar="H [W]",
        help=(
            "与导出该 .engine 时一致：letterbox 目标尺寸。一个正整数=方形边长；"
            "两个正整数=高 宽。**强烈建议显式指定**。"
        ),
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=None,
        help="画框线宽，默认随分辨率自动",
    )
    parser.add_argument(
        "--crop-square",
        action="store_true",
        help=(
            "先用 ffmpeg 将每帧中心裁成最小内接正方形，再送 YOLO；"
            "输出视频也是正方形。需本机可执行 ffmpeg"
        ),
    )
    parser.add_argument(
        "--save-cropped",
        type=str,
        default=None,
        help="与 --crop-square 同用：把裁剪后的中间视频保存到此路径",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"错误：找不到引擎文件: {model_path}", file=sys.stderr)
        return 1
    if model_path.suffix.lower() != ".engine":
        print(
            f"提示：文件扩展名不是 .engine: {model_path}（将仍交给 Ultralytics 尝试加载）",
            file=sys.stderr,
        )
    if args.imgsz is None:
        print(
            "提示：未指定 --imgsz。请与导出 .engine 时用的尺寸一致，否则易出错；"
            "例如: --imgsz 640 或 --imgsz 640 320",
            file=sys.stderr,
        )

    src = Path(args.source)
    if not src.is_file():
        print(f"错误：找不到输入视频: {src}", file=sys.stderr)
        return 1

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = src.parent / f"{src.stem}_trt_detect{src.suffix or '.mp4'}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.save_cropped and not args.crop_square:
        print("提示：--save-cropped 仅在 --crop-square 时有效，已忽略。", file=sys.stderr)
    if args.imgsz is not None:
        if not args.imgsz or any(s <= 0 for s in args.imgsz):
            print("错误：--imgsz 须为至少一个正整数，或 高 宽 两个正整数", file=sys.stderr)
            return 1
        if len(args.imgsz) > 2:
            print("错误：--imgsz 最多两个数（高、宽）", file=sys.stderr)
            return 1

    predict_src = src
    temp_cropped: Path | None = None
    if args.crop_square:
        try:
            if args.save_cropped:
                cro = Path(args.save_cropped)
                _make_center_square_video(src, cro)
                predict_src = cro
                print(f"已用 ffmpeg 裁成中心正方形: {cro.resolve()}", flush=True)
            else:
                fd, tpath = tempfile.mkstemp(suffix=".mp4", prefix="yolo_center_sq_")
                os.close(fd)
                temp_cropped = Path(tpath)
                _make_center_square_video(src, temp_cropped)
                predict_src = temp_cropped
                print("已用 ffmpeg 裁成中心正方形（临时文件）", flush=True)
        except RuntimeError as e:
            print(f"错误: {e}", file=sys.stderr)
            return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("未安装 ultralytics，请执行: pip install ultralytics", file=sys.stderr)
        if temp_cropped is not None:
            temp_cropped.unlink(missing_ok=True)
        return 1

    cap_meta = cv2.VideoCapture(str(predict_src))
    if not cap_meta.isOpened():
        print(f"错误：无法打开视频: {predict_src}", file=sys.stderr)
        if temp_cropped is not None:
            temp_cropped.unlink(missing_ok=True)
        return 1
    fps = float(cap_meta.get(cv2.CAP_PROP_FPS) or 25.0)
    if fps <= 0 or fps > 1e3:
        fps = 25.0
    cap_meta.release()

    model = YOLO(str(model_path))
    pred_kw: dict = {
        "source": str(predict_src),
        "stream": True,
        "conf": args.conf,
        "iou": args.iou,
        "verbose": False,
        "device": args.device,
    }
    if args.imgsz is not None:
        if len(args.imgsz) == 1:
            pred_kw["imgsz"] = args.imgsz[0]
        else:
            pred_kw["imgsz"] = args.imgsz
    if args.line_width is not None:
        pred_kw["line_width"] = args.line_width

    try:
        it = model.predict(**pred_kw)
        writer: cv2.VideoWriter | None = None
        n = 0
        t_pre: List[float] = []
        t_inf: List[float] = []
        t_post: List[float] = []
        t0 = time.perf_counter()
        try:
            for r in it:
                sp = getattr(r, "speed", None)
                if isinstance(sp, dict):
                    t_pre.append(float(sp.get("preprocess", 0.0) or 0.0))
                    t_inf.append(float(sp.get("inference", 0.0) or 0.0))
                    t_post.append(float(sp.get("postprocess", 0.0) or 0.0))
                frame = r.plot()
                h, w = frame.shape[:2]
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(out_path),
                        fourcc,
                        fps,
                        (w, h),
                    )
                    if not writer.isOpened():
                        print(
                            f"错误：无法创建输出视频: {out_path}（可尝试换 -o 为 .avi）",
                            file=sys.stderr,
                        )
                        return 1
                writer.write(frame)
                n += 1
                if n % 100 == 0:
                    print(f"已处理 {n} 帧 …", flush=True)

            if n == 0:
                print(
                    "错误：未处理任何帧，请检查视频是否可读及模型是否兼容。",
                    file=sys.stderr,
                )
                return 1

            wall = time.perf_counter() - t0
            print(f"完成: {n} 帧 -> {out_path.resolve()}")
            _print_speed_stats(n, wall, t_pre, t_inf, t_post)
            return 0
        finally:
            if writer is not None:
                writer.release()
    finally:
        if temp_cropped is not None and temp_cropped.is_file():
            temp_cropped.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
