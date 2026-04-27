# YOLO TensorRT turorial for V100 GPU

我仅仅在Tesla V100 GPU上测试过，由于最新的tensorrt(10.x.x)不支持SM 70及以下的设备（例如 V100），所以我放了一个备忘工程在这里，只要按照这个工程就可以跑通

---

## 中文

### 步骤

#### 第一步：按照要求组织环境


| 组件 | 版本 |
|------|------|
| python| 3.10 |
| `torch` | 2.3.0+cu121 |
| PyTorch 内置 CUDA (`torch.version.cuda`) | 12.1 |
| pip `nvidia-cudnn-cu12` | 8.9.2.26 |
| pip `tensorrt` | 8.6.1 |

通常只需要装torch，下面这些cuda什么的都会自动跟对torch把这些搞定之后再

```
pip install -r requirements
```

#### 第二步： 下载tensorrt包

直接pip安装是不行的， 最近一个支持SM70的是上一个Major版本：8.x.x， 可以在这里下载：

[Link](https://developer.nvidia.com/nvidia-tensorrt-8x-download)

下载完成之后安装里面的对应的pip包

```
cd ./TensorRT-8.6.1.6/python
pip install tensorrt-8.6.1-cp310-none-linux_x86_64.whl
```

#### 第四步：其他

先用交互式python import一下tensorrt，看看是否有问题

1. 若 `import tensorrt` 有问题，需手动设置 `LD_LIBRARY_PATH`（TensorRT / cuDNN 的 `lib`），请按本机安装路径配置。
这是一个例子: ()

```
export LD_LIBRARY_PATH="/home/junzhicai/tensorrt/TensorRT-8.6.1.6/lib:/home/junzhicai/anaconda3/envs/minimind/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
```

2. 可能需要手动把numpy降级回1.x.x大版本

### 脚本

| 文件 | 作用 |
|------|------|
| `yolo2tensorrt-vanilla.py` | ONNX → `.engine`（不依赖 ultralytics；INT8 用 `data.yaml` 的 `val`） |
| `yolo2tensorrt-ultralytics.py` | `.pt` → `.engine`（Ultralytics 官方导出链） |
| `detect_video.py` | `.pt` 视频检测 + 画框写视频 |
| `detect_video_tensorrt.py` | `.engine` 视频检测 + 画框写视频 |

### 示例（含全部可传参数；`--half` 与 `--int8` 勿同时使用）
### 哦对了，量化数据集需要组织成一个yolo训练数据集的同款格式，然后传入里面的data.yaml就可以了

**ONNX → TensorRT（FP16）：**

```bash
python yolo2tensorrt-vanilla.py model.onnx \
  --imgsz 640 \
  --batch 1 \
  --device 0 \
  --half \
  --workspace 4 \
  --dynamic \
  --verbose \
  -o out_fp16.engine
```

**ONNX → TensorRT（INT8）：**

```bash
python yolo2tensorrt-vanilla.py model.onnx \
  --imgsz 640 \
  --batch 1 \
  --device 0 \
  --int8 \
  --data /path/to/data.yaml \
  --workspace 4 \
  --dynamic \
  --fraction 1.0 \
  --max-calib-samples 1024 \
  --verbose \
  -o out_int8.engine
```

**Ultralytics：权重 → TensorRT：**

```bash
python yolo2tensorrt-ultralytics.py weights.pt \
  --imgsz 640 640 \
  --batch 1 \
  --device 0 \
  --half \
  --workspace 4 \
  --dynamic \
  --nms \
  --simplify \
  --opset 17 \
  --task detect \
  --verbose
```

**Ultralytics：INT8（全部参数，勿与 `--half` 同用）：**

```bash
python yolo2tensorrt-ultralytics.py weights.pt \
  --imgsz 640 640 \
  --batch 1 \
  --device 0 \
  --int8 \
  --data /path/to/data.yaml \
  --workspace 4 \
  --dynamic \
  --nms \
  --simplify \
  --opset 17 \
  --task detect \
  --verbose
```

（`--no-simplify` 可代替 `--simplify`。）

**视频检测（PyTorch）：**

```bash
python detect_video.py weights.pt video.mp4 \
  -o out.mp4 \
  --conf 0.25 \
  --iou 0.7 \
  --device 0 \
  --imgsz 640 \
  --line-width 2 \
  --crop-square \
  --save-cropped cropped.mp4
```

**视频检测（TensorRT）：**

```bash
python detect_video_tensorrt.py model.engine video.mp4 \
  -o out.mp4 \
  --conf 0.25 \
  --iou 0.7 \
  --device 0 \
  --imgsz 640 \
  --line-width 2 \
  --crop-square \
  --save-cropped cropped.mp4
```

依赖见 `requirements.txt`。

---

## English

This memo was validated on **Tesla V100**. Recent **TensorRT 10.x** drops support for **SM 70 and below** (e.g. V100); use a **TensorRT 8.x** line that still supports SM70 and follow this repo to get a working path.

### Steps

#### Step 1: Baseline environment

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| `torch` | 2.3.0+cu121 |
| PyTorch CUDA (`torch.version.cuda`) | 12.1 |
| pip `nvidia-cudnn-cu12` | 8.9.2.26 |
| pip `tensorrt` | 8.6.1 |
| pip `ultralytics` | 8.4.41 |

#### Step 2: Install the TensorRT package

Plain `pip install tensorrt` is often wrong for V100: the latest **TensorRT 10.x** does not support SM70. The last major line that still supports SM70 is **8.x**. Download from NVIDIA:

[TensorRT 8.x download](https://developer.nvidia.com/nvidia-tensorrt-8x-download)

Then install the matching wheel from the extracted package, for example:

```bash
cd ./TensorRT-8.6.1.6/python
pip install tensorrt-8.6.1-cp310-none-linux_x86_64.whl
```

#### Step 4: Misc

1. In an interactive Python shell, run `import tensorrt`. If it fails, add TensorRT and cuDNN `lib` dirs to **`LD_LIBRARY_PATH`** (paths depend on your install). Example:

```bash
export LD_LIBRARY_PATH="/home/junzhicai/tensorrt/TensorRT-8.6.1.6/lib:/home/junzhicai/anaconda3/envs/minimind/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
```

2. You may need to **pin NumPy to a 1.x** release if you hit binary/API issues.

### Scripts

| File | Role |
|------|------|
| `yolo2tensorrt-vanilla.py` | ONNX → `.engine` (no ultralytics; INT8 uses `val` in `data.yaml`) |
| `yolo2tensorrt-ultralytics.py` | `.pt` → `.engine` (Ultralytics export path) |
| `detect_video.py` | `.pt` video detect + draw boxes |
| `detect_video_tensorrt.py` | `.engine` video detect + draw boxes |

### Examples (all CLI flags shown; do **not** combine `--half` with `--int8`)

For calibration / INT8, lay out data in the **same layout as a YOLO training dataset** and pass its **`data.yaml`**.

**ONNX → TensorRT (FP16):**

```bash
python yolo2tensorrt-vanilla.py model.onnx \
  --imgsz 640 \
  --batch 1 \
  --device 0 \
  --half \
  --workspace 4 \
  --dynamic \
  --verbose \
  -o out_fp16.engine
```

**ONNX → TensorRT (INT8):**

```bash
python yolo2tensorrt-vanilla.py model.onnx \
  --imgsz 640 \
  --batch 1 \
  --device 0 \
  --int8 \
  --data /path/to/data.yaml \
  --workspace 4 \
  --dynamic \
  --fraction 1.0 \
  --max-calib-samples 1024 \
  --verbose \
  -o out_int8.engine
```

**Ultralytics: weights → TensorRT:**

```bash
python yolo2tensorrt-ultralytics.py weights.pt \
  --imgsz 640 640 \
  --batch 1 \
  --device 0 \
  --half \
  --workspace 4 \
  --dynamic \
  --nms \
  --simplify \
  --opset 17 \
  --task detect \
  --verbose
```

**Ultralytics: INT8 (all flags; never use with `--half`):**

```bash
python yolo2tensorrt-ultralytics.py weights.pt \
  --imgsz 640 640 \
  --batch 1 \
  --device 0 \
  --int8 \
  --data /path/to/data.yaml \
  --workspace 4 \
  --dynamic \
  --nms \
  --simplify \
  --opset 17 \
  --task detect \
  --verbose
```

Use `--no-simplify` instead of `--simplify` if needed.

**Video detect (PyTorch):**

```bash
python detect_video.py weights.pt video.mp4 \
  -o out.mp4 \
  --conf 0.25 \
  --iou 0.7 \
  --device 0 \
  --imgsz 640 \
  --line-width 2 \
  --crop-square \
  --save-cropped cropped.mp4
```

**Video detect (TensorRT):**

```bash
python detect_video_tensorrt.py model.engine video.mp4 \
  -o out.mp4 \
  --conf 0.25 \
  --iou 0.7 \
  --device 0 \
  --imgsz 640 \
  --line-width 2 \
  --crop-square \
  --save-cropped cropped.mp4
```

See `requirements.txt` for dependencies.
