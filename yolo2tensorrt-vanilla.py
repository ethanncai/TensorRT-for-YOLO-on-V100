#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNX → TensorRT ``.engine``（不依赖 ultralytics；标定 INT8 仅需本脚本 + data.yaml）。

先用 Ultralytics 从 ``.pt`` 导出 ONNX，再运行本脚本。示例命令（任选其一）::

    # Python
    python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='onnx', imgsz=640, simplify=True)"

    # 官方 CLI
    yolo export model=best.pt format=onnx imgsz=640 simplify=True

    # FP16 追踪（与 TRT FP16 常配套）
    yolo export model=best.pt format=onnx imgsz=640 simplify=True half=True

再编译引擎::

    python yolo_to_tensorrt.py best.onnx --half
    python yolo_to_tensorrt.py best.onnx --int8 --data data.yaml -o out.engine

SPDX-License-Identifier: AGPL-3.0
onnx2engine 衍生自 Ultralytics: https://github.com/ultralytics/ultralytics
（ultralytics/utils/export/engine.py）
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# ONNX → engine（原 trt_onnx2engine.py，合并于此）
# ---------------------------------------------------------------------------


def _is_jetson() -> bool:
    return Path("/etc/nv_tegra_release").exists() or bool(os.environ.get("JETSON_L4T"))


def _trt_log(prefix: str, msg: str) -> None:
    print(f"{prefix}{msg}", flush=True)


def onnx2engine(
    onnx_file: str,
    output_file: Path | str | None = None,
    workspace: int | None = None,
    half: bool = False,
    int8: bool = False,
    dynamic: bool = False,
    shape: tuple[int, int, int, int] = (1, 3, 640, 640),
    dla: int | None = None,
    dataset=None,
    metadata: dict | None = None,
    verbose: bool = False,
    prefix: str = "",
) -> str:
    """ONNX 转 TensorRT engine（与 Ultralytics onnx2engine 行为一致）。"""
    import tensorrt as trt

    output_file = output_file or Path(onnx_file).with_suffix(".engine")

    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    workspace_bytes = int((workspace or 0) * (1 << 30))
    is_trt10 = int(trt.__version__.split(".", 1)[0]) >= 10
    if is_trt10 and workspace_bytes > 0:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    elif workspace_bytes > 0:
        config.max_workspace_size = workspace_bytes
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    half = builder.platform_has_fast_fp16 and half
    int8 = builder.platform_has_fast_int8 and int8

    if dla is not None:
        if not _is_jetson():
            raise ValueError("DLA is only available on NVIDIA Jetson devices")
        _trt_log(prefix, f"enabling DLA on core {dla}...")
        if not half and not int8:
            raise ValueError(
                "DLA requires either 'half=True' (FP16) or 'int8=True' (INT8) to be enabled. "
                "Please enable one of them and try again."
            )
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = int(dla)
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_file):
        raise RuntimeError(f"failed to load ONNX file: {onnx_file}")

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        _trt_log(prefix, f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        _trt_log(prefix, f'output "{out.name}" with shape{out.shape} {out.dtype}')

    if dynamic:
        profile = builder.create_optimization_profile()
        min_shape = (1, shape[1], 32, 32)
        max_shape = (*shape[:2], *(int(max(2, workspace or 2) * d) for d in shape[2:]))
        for inp in inputs:
            profile.set_shape(inp.name, min=min_shape, opt=shape, max=max_shape)
        config.add_optimization_profile(profile)
        if int8 and not is_trt10:
            config.set_calibration_profile(profile)

    _trt_log(
        prefix,
        f"building {'INT8' if int8 else 'FP' + ('16' if half else '32')} engine as {output_file}",
    )
    if int8:
        if dataset is None:
            raise ValueError("int8=True requires a calibration dataset (iterable of dict with key 'img').")
        config.set_flag(trt.BuilderFlag.INT8)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        class EngineCalibrator(trt.IInt8Calibrator):
            def __init__(self, calib_dataset, cache: str = "") -> None:
                trt.IInt8Calibrator.__init__(self)
                self.dataset = calib_dataset
                self.data_iter = iter(calib_dataset)
                self.algo = (
                    trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
                    if dla is not None
                    else trt.CalibrationAlgoType.MINMAX_CALIBRATION
                )
                self.batch = getattr(calib_dataset, "batch_size", None) or 1
                self.cache = Path(cache)

            def get_algorithm(self) -> trt.CalibrationAlgoType:
                return self.algo

            def get_batch_size(self) -> int:
                return self.batch or 1

            def get_batch(self, names) -> list[int] | None:
                try:
                    im0s = next(self.data_iter)["img"] / 255.0
                    im0s = im0s.to("cuda") if im0s.device.type == "cpu" else im0s
                    return [int(im0s.data_ptr())]
                except StopIteration:
                    return None

            def read_calibration_cache(self) -> bytes | None:
                if self.cache.exists() and self.cache.suffix == ".cache":
                    return self.cache.read_bytes()
                return None

            def write_calibration_cache(self, cache: bytes) -> None:
                _ = self.cache.write_bytes(cache)

        config.int8_calibrator = EngineCalibrator(
            dataset,
            cache=str(Path(onnx_file).with_suffix(".cache")),
        )

    elif half:
        config.set_flag(trt.BuilderFlag.FP16)

    if is_trt10:
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("TensorRT engine build failed, check logs for errors")
        with open(output_file, "wb") as t:
            if metadata is not None:
                meta = json.dumps(metadata)
                t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
                t.write(meta.encode())
            t.write(engine)
    else:
        with builder.build_engine(network, config) as engine, open(output_file, "wb") as t:
            if engine is None:
                raise RuntimeError("TensorRT engine build failed, check logs for errors")
            if metadata is not None:
                meta = json.dumps(metadata)
                t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
                t.write(meta.encode())
            t.write(engine.serialize())

    return str(output_file)


# ---------------------------------------------------------------------------
# data.yaml → INT8 标定图像
# ---------------------------------------------------------------------------


def _resolve_val_images(data_yaml: Path) -> List[Path]:
    import yaml

    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("--data YAML 根须为 dict")

    path_s = (raw.get("path") or "").strip()
    val = raw.get("val")
    if val is None:
        raise SystemExit("YAML 中缺少 val 字段")
    if isinstance(val, (list, tuple)):
        val = val[0]

    if path_s:
        rp = Path(path_s)
        root_p = rp.resolve() if rp.is_absolute() else (data_yaml.parent / rp).resolve()
    else:
        root_p = data_yaml.parent
    v = Path(str(val))
    if v.is_absolute():
        val_path = v
    else:
        val_path = (root_p / v).resolve()

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if val_path.is_file():
        return [val_path]
    if not val_path.is_dir():
        raise SystemExit(f"val 路径不存在: {val_path}")

    out: List[Path] = []
    for p in sorted(val_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    seen = set()
    unique: List[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(rp)
    if not unique:
        raise SystemExit(f"在 val 目录中未找到图像: {val_path}")
    return unique


def _letterbox(
    im: np.ndarray,
    new_shape: Tuple[int, int],
) -> np.ndarray:
    import cv2

    h0, w0 = im.shape[:2]
    nh, nw = new_shape
    r = min(nw / w0, nh / h0)
    w1, h1 = int(round(w0 * r)), int(round(h0 * r))
    if (w1, h1) != (w0, h0):
        im = cv2.resize(im, (w1, h1), interpolation=cv2.INTER_LINEAR)
    top = (nh - h1) // 2
    left = (nw - w1) // 2
    out = np.full((nh, nw, 3), 114, dtype=np.uint8)
    out[top : top + h1, left : left + w1] = im
    out = out[:, :, ::-1].astype(np.float32) / 255.0
    out = np.transpose(out, (2, 0, 1))
    return np.ascontiguousarray(out)


def _onnx_static_input_shape(onnx_path: Path) -> Tuple[int, int, int, int]:
    import onnx

    m = onnx.load_model(str(onnx_path))
    if not m.graph.input:
        raise RuntimeError("ONNX 无 input")
    t = m.graph.input[0]
    dims: List[int] = []
    for d in t.type.tensor_type.shape.dim:
        if d.dim_value:
            dims.append(int(d.dim_value))
        else:
            dims.append(1)
    if len(dims) < 4:
        raise RuntimeError(f"无法从 ONNX 解析 NCHW: {dims}")
    return int(dims[0]), int(dims[1]), int(dims[2]), int(dims[3])


class _CalibDataset:
    """与 EngineCalibrator 兼容：可迭代 ``{\"img\": tensor}``，``img`` 为 0–255 浮点。"""

    def __init__(self, batch_tensors: List["torch.Tensor"]) -> None:
        import torch

        if not batch_tensors:
            raise ValueError("INT8 标定 batch 列表为空")
        self.batch_size = int(batch_tensors[0].shape[0])
        self._rows: List[dict] = []
        for b in batch_tensors:
            b = b if isinstance(b, torch.Tensor) else torch.as_tensor(b)
            self._rows.append({"img": (b * 255.0).to(dtype=torch.float32)})

    def __iter__(self):
        return iter(self._rows)


def _build_int8_calib_dataset(
    data_yaml: Path,
    hw: Tuple[int, int],
    batch: int,
    device: str,
    fraction: float,
    max_calib_samples: int,
) -> _CalibDataset:
    import torch

    paths = _resolve_val_images(data_yaml)
    random.shuffle(paths)
    n_take = max(1, int(len(paths) * fraction))
    n_take = min(n_take, max(1, max_calib_samples))
    paths = paths[:n_take]
    h, w = hw
    dev = torch.device(device if device.startswith("cuda") else f"cuda:{device}")
    batches: List[torch.Tensor] = []
    buf: List[np.ndarray] = []
    for p in paths:
        import cv2

        im = cv2.imread(str(p))
        if im is None:
            continue
        t = _letterbox(im, (h, w))
        buf.append(t)
        if len(buf) >= batch:
            x = np.stack(buf, axis=0)
            buf = []
            batches.append(torch.from_numpy(x).to(dev, dtype=torch.float32))
    if buf:
        x = np.stack(buf, axis=0)
        if 0 < x.shape[0] < batch:
            pad = np.stack([x[-1]] * (batch - x.shape[0]), axis=0)
            x = np.vstack([x, pad])
        elif x.shape[0] == 0:
            raise RuntimeError("无有效标定图像")
        batches.append(torch.from_numpy(x).to(dev, dtype=torch.float32))
    if not batches:
        raise RuntimeError("INT8 标定未得到任何 batch")
    return _CalibDataset(batches)


def _apply_cuda_visible(device: str) -> None:
    s = device.strip().lower().replace("gpu:", "")
    if s.startswith("cuda:"):
        s = s[5:]
    if s.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = s


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ONNX -> TensorRT .engine（内置 onnx2engine，不依赖 ultralytics）",
    )
    parser.add_argument(
        "onnx",
        type=Path,
        help="输入 ONNX 路径",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=None,
        metavar="H [W]",
        help="仅用于与 ONNX 静态形状核对时的提示；实际以 ONNX 为准",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--data", type=Path, default=None, help="INT8：data.yaml（读 val）")
    parser.add_argument("--workspace", type=int, default=4)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--fraction", type=float, default=1.0, help="从 val 中抽取的比例 (0,1]")
    parser.add_argument(
        "--max-calib-samples",
        type=int,
        default=1024,
        help="INT8 标定最多使用的图像张数上限（默认 1024，在 --fraction 之后生效）",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    onnx_path = args.onnx.resolve()
    if not onnx_path.is_file():
        print(f"错误: 找不到 ONNX: {onnx_path}", file=sys.stderr)
        return 1
    if onnx_path.suffix.lower() != ".onnx":
        print(f"错误: 输入须为 .onnx: {onnx_path}", file=sys.stderr)
        return 1

    if args.int8 and args.half:
        print("错误: 不要同时指定 --int8 与 --half", file=sys.stderr)
        return 1
    if args.int8:
        if args.data is None or not args.data.is_file():
            print("错误: --int8 需要 --data 指向存在的 YAML", file=sys.stderr)
            return 1
    if not (0.0 < args.fraction <= 1.0):
        print("错误: --fraction 应在 (0,1]", file=sys.stderr)
        return 1
    if args.max_calib_samples < 1:
        print("错误: --max-calib-samples 须 >= 1", file=sys.stderr)
        return 1

    _apply_cuda_visible(args.device)

    sh = _onnx_static_input_shape(onnx_path)
    shape_trt = (args.batch, sh[1], sh[2], sh[3])
    if args.imgsz is not None:
        if len(args.imgsz) == 1:
            e = (args.batch, 3, args.imgsz[0], args.imgsz[0])
        else:
            e = (args.batch, 3, args.imgsz[0], args.imgsz[1])
        if e != shape_trt:
            print(
                f"提示: --imgsz 与 ONNX 静态形状不一致 (ONNX {shape_trt} vs 指定 {e})，"
                f"TRT 使用 ONNX 形状 {shape_trt}",
                file=sys.stderr,
            )

    calib_ds = None
    if args.int8:
        dev_infer = args.device if "cuda" in args.device else f"cuda:{args.device}"
        calib_ds = _build_int8_calib_dataset(
            args.data.resolve(),
            (shape_trt[2], shape_trt[3]),
            args.batch,
            dev_infer,
            args.fraction,
            args.max_calib_samples,
        )

    engine_out = args.output
    if engine_out is None:
        engine_out = onnx_path.with_suffix(".engine")
    else:
        engine_out = engine_out.resolve()
        engine_out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    try:
        onnx2engine(
            str(onnx_path),
            output_file=engine_out,
            workspace=args.workspace,
            half=args.half,
            int8=args.int8,
            dynamic=args.dynamic,
            shape=shape_trt,
            dla=None,
            dataset=calib_ds,
            metadata=None,
            verbose=args.verbose,
            prefix="TensorRT: ",
        )
    except Exception as e:
        print(f"错误: TensorRT 构建失败: {e}", file=sys.stderr)
        return 1

    engine_path = Path(engine_out).resolve()
    wall = time.perf_counter() - t0
    mb = engine_path.stat().st_size / (1 << 20)
    print(
        f"TensorRT: export success {wall:.1f}s, saved as '{engine_path}' ({mb:.1f} MB)",
        flush=True,
    )
    print(f"OK: {engine_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
