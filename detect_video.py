#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a YOLO .pt (or .onnx 等) 模型在视频上推理，并写出带框的可视化视频。

依赖: ``pip install -r requirements.txt``（主要为 ultralytics）。

可选：先用 **ffmpeg** 将画面**中心裁成最小内接正方形**再推理（见 ``--crop-square``）。

结束时打印 **全流程 FPS**（墙钟）与 ultralytics 的 **pre/infer/post 毫秒**（均值/帧）。

示例::

    python detect_video.py yolo11n.pt input.mp4 -o out_vis.mp4
    python detect_video.py path/to/best.pt clip.mp4 --conf 0.35 --device 0
    # 与训练/导出一致的实际输入尺寸（letterbox 后送入网络的边长或高×宽）
    python detect_video.py best.engine clip.mp4 --imgsz 640
    python detect_video.py yolo11n.pt wide.mp4 -o out.mp4 --imgsz 640 320
    python detect_video.py yolo11n.pt wide.mp4 -o out.mp4 --crop-square
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


# ffmpeg: 中心裁切，边长 = min(宽, 高)，与常见「正方形化」需求一致
_FFMPEG_CENTER_SQUARE_VF = (
    r"crop=min(iw\,ih):min(iw\,ih):(iw-min(iw\,ih))/2:(ih-min(iw\,ih))/2"
)


def _make_center_square_video(
    in_path: Path,
    out_path: Path,
) -> None:
    """用 ffmpeg 将视频中心裁成最小内接正方形，写到 ``out_path``。"""
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
    """打印全流程墙钟与 ultralytics 返回的 per-frame 计时（见 Results.speed）。"""
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
        description="YOLO 视频检测可视化：读入模型与视频，写出绘制检测框后的视频。",
    )
    parser.add_argument(
        "model",
        type=str,
        help="YOLO 权重路径（如 yolo11n.pt、runs/detect/.../weights/best.pt）",
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
        help="输出视频路径；默认与输入同目录，文件名为 <原名>_detect.mp4",
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
        default=None,
        help="设备：cpu、0、0,1 等；不传则由 ultralytics 自动选择",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=None,
        metavar="H [W]",
        help=(
            "实际送入模型（letterbox 目标）的输入尺寸：一个正整数=方形边长，如 640；"
            "两个正整数=高 宽，如 640 320。导出 TensorRT/ONNX 时用的 imgsz 应与此一致。"
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
        help="与 --crop-square 同用：把裁剪后的中间视频保存到此路径（便于检查或复用）",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"错误：找不到模型文件: {model_path}", file=sys.stderr)
        return 1

    src = Path(args.source)
    if not src.is_file():
        print(f"错误：找不到输入视频: {src}", file=sys.stderr)
        return 1

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = src.parent / f"{src.stem}_detect{src.suffix or '.mp4'}"
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
        print(
            "未安装 ultralytics，请执行: pip install ultralytics",
            file=sys.stderr,
        )
        if temp_cropped is not None:
            temp_cropped.unlink(missing_ok=True)
        return 1

    # 对最终送入 YOLO 的轨道读 fps（与裁剪后文件一致）
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
    }
    if args.device is not None:
        pred_kw["device"] = args.device
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
