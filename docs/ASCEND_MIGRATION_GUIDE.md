# 华为昇腾 Atlas 300V 迁移指南

## 概述

本文档描述如何将基于 NVIDIA GPU 的多路 RTSP 流人脸检测系统迁移到华为昇腾 Atlas 300V 平台。

## 平台对比

### 硬件规格对比

| 规格 | NVIDIA A10 | 华为 Atlas 300V |
|------|-----------|-----------------|
| **芯片** | Ampere GA102 | 昇腾 310P |
| **算力 (INT8)** | ~330 TOPS | 100 TOPS |
| **算力 (FP16)** | ~165 TFLOPS | 50 TFLOPS |
| **显存/内存** | 24GB GDDR6 | 24GB LPDDR4X |
| **视频解码** | ~40 路 1080p | **100 路 1080p** |
| **功耗** | 150W | 72W |
| **国产化** | 否 | **是** |

### 软件栈对比

| 功能 | NVIDIA | 华为昇腾 |
|------|--------|----------|
| **编程接口** | CUDA | AscendCL (ACL) |
| **推理引擎** | TensorRT | ATC + AscendCL |
| **视频解码** | NVDEC | DVPP |
| **模型格式** | .engine | .om |
| **Python API** | pycuda/tensorrt | pyACL |

## 迁移步骤

### 1. 环境准备

#### 安装 CANN Toolkit

```bash
# 1. 下载 CANN toolkit
# 访问 https://www.hiascend.com/software/cann

# 2. 安装
chmod +x Ascend-cann-toolkit_7.0.0_linux-x86_64.run
./Ascend-cann-toolkit_7.0.0_linux-x86_64.run --install

# 3. 配置环境变量
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 4. 验证安装
npu-smi info
python -c "import acl; print('ACL 可用')"
```

#### 安装 Python 依赖

```bash
pip install -r requirements_ascend.txt
```

### 2. 模型转换

#### 从 PyTorch 导出 ONNX

```python
import torch

# 加载原始 PyTorch 模型
model = torch.load("face_detection.pth")
model.eval()

# 导出 ONNX
dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    model,
    dummy_input,
    "face_detection.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)
```

#### 使用 ATC 转换为 .om

```bash
# 静态 shape
atc --framework=5 \
    --model=face_detection.onnx \
    --output=face_detection \
    --input_format=NCHW \
    --input_shape="input:1,3,640,640" \
    --soc_version=Ascend310P \
    --output_type=FP16 \
    --log=error

# 动态 batch
atc --framework=5 \
    --model=face_detection.onnx \
    --output=face_detection_dynamic \
    --input_format=NCHW \
    --input_shape="input:-1,3,640,640" \
    --dynamic_batch_size="1,2,4,8,16" \
    --soc_version=Ascend310P \
    --log=error
```

#### 使用转换工具

```bash
# 使用项目提供的转换工具
python ascend_model_converter.py convert \
    --model face_detection.onnx \
    --output models/face_detection \
    --framework onnx \
    --soc Ascend310P \
    --output-type FP16
```

### 3. 代码迁移

#### API 映射表

| NVIDIA API | 昇腾 API | 说明 |
|------------|----------|------|
| `torch.cuda.is_available()` | `acl.init()` | 检查设备可用性 |
| `torch.cuda.set_device(id)` | `acl.rt.set_device(id)` | 设置设备 |
| `cuda.memcpy_htod()` | `acl.rt.memcpy(..., MEMCPY_HOST_TO_DEVICE)` | 内存拷贝 |
| `cuda.memcpy_dtoh()` | `acl.rt.memcpy(..., MEMCPY_DEVICE_TO_HOST)` | 内存拷贝 |
| `trt.Runtime()` | `acl.mdl.load_from_file()` | 加载模型 |
| `context.execute_v2()` | `acl.mdl.execute()` | 执行推理 |

#### 推理代码对比

**NVIDIA TensorRT:**

```python
import tensorrt as trt
import pycuda.driver as cuda

# 加载引擎
runtime = trt.Runtime(trt.Logger())
engine = runtime.deserialize_cuda_engine(engine_data)
context = engine.create_execution_context()

# 执行推理
cuda.memcpy_htod(d_input, h_input)
context.execute_v2(bindings)
cuda.memcpy_dtoh(h_output, d_output)
```

**华为昇腾 ACL:**

```python
import acl

# 初始化
acl.init()
acl.rt.set_device(0)
context, _ = acl.rt.create_context(0)

# 加载模型
model_id, _ = acl.mdl.load_from_file("model.om")

# 执行推理
acl.rt.memcpy(d_input, size, h_input, size, ACL_MEMCPY_HOST_TO_DEVICE)
acl.mdl.execute(model_id, input_dataset, output_dataset)
acl.rt.memcpy(h_output, size, d_output, size, ACL_MEMCPY_DEVICE_TO_HOST)
```

### 4. 视频解码迁移

#### NVDEC → DVPP

**NVIDIA (OpenCV + CUDA):**

```python
import cv2

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
```

**华为 DVPP:**

```python
import acl

# 创建 DVPP 通道
dvpp_channel = acl.media.dvpp_create_channel_desc()
acl.media.dvpp_create_channel(dvpp_channel)

# 解码
acl.media.dvpp_vdec_send_frame(config, frame_data, size, pic_desc)
```

### 5. 性能监控迁移

#### nvidia-smi → npu-smi

```bash
# NVIDIA
nvidia-smi

# 华为
npu-smi info
npu-smi info -t common -i 0
```

#### Python 监控

**NVIDIA:**

```python
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
util = pynvml.nvmlDeviceGetUtilizationRates(handle)
```

**华为:**

```python
from ascend_performance_monitor import AscendPerformanceMonitor
monitor = AscendPerformanceMonitor(device_id=0)
metrics = monitor.get_npu_metrics()
```

## 文件对照表

| NVIDIA 版本 | 昇腾版本 | 说明 |
|-------------|----------|------|
| `config.py` | `config_ascend.py` | 配置文件 |
| `face_detector.py` | `ascend_face_detector.py` | 人脸检测器 |
| `tensorrt_face_detector.py` | `ascend_face_detector.py` | 加速推理 |
| `multi_stream_manager.py` | `ascend_stream_manager.py` | 多流管理 |
| `performance_monitor.py` | `ascend_performance_monitor.py` | 性能监控 |
| `tensorrt_optimizer.py` | `ascend_model_converter.py` | 模型转换 |
| `multi_rtsp_face_detection.py` | `multi_rtsp_face_detection_ascend.py` | 主程序 |
| `requirements.txt` | `requirements_ascend.txt` | 依赖 |

## 性能优化建议

### 1. 使用 DVPP 硬件加速

```python
# 启用 DVPP 解码 (默认)
manager = AscendMultiStreamManager(use_dvpp=True)
```

### 2. 批处理优化

```python
# 增大批处理大小
manager = AscendMultiStreamManager(batch_size=16)
```

### 3. 模型量化

```bash
# 使用 INT8 量化提升性能
atc ... --output_type=3  # INT8
```

### 4. 异步推理

```python
# 使用异步执行
acl.mdl.execute_async(model_id, input, output, stream)
acl.rt.synchronize_stream(stream)
```

## 常见问题

### Q1: ACL 初始化失败

```
解决方案:
1. 检查驱动是否安装: npu-smi info
2. 检查 CANN 是否安装: source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
3. 检查权限: 确保用户在 HwHiAiUser 组
```

### Q2: 模型转换失败

```
解决方案:
1. 检查 ONNX 算子是否支持
2. 使用 --log=debug 查看详细错误
3. 尝试简化模型或使用支持的算子替代
```

### Q3: 推理性能不如预期

```
解决方案:
1. 使用 FP16/INT8 量化
2. 增大批处理大小
3. 使用异步推理
4. 检查内存拷贝是否成为瓶颈
```

## 预期性能

### Atlas 300V 性能参考

| 模式 | 1080p 流数 | 检测 FPS | NPU 利用率 |
|------|-----------|----------|-----------|
| FP32 | 15-20 路 | 5 fps | 80% |
| **FP16** | **30-40 路** | **5 fps** | **75%** |
| **INT8** | **50-70 路** | **5 fps** | **85%** |

### 与 A10 对比

| 指标 | A10 (INT8) | Atlas 300V (INT8) |
|------|-----------|-------------------|
| 流数 | 80-120 路 | 50-70 路 |
| 视频解码 | 40 路限制 | **100 路** |
| 功耗 | 150W | **72W** |
| 国产化 | 否 | **是** |

## 参考资源

- [华为昇腾官网](https://www.hiascend.com/)
- [CANN 开发文档](https://www.hiascend.com/document)
- [昇腾社区](https://www.hiascend.com/forum)
- [GitHub 示例](https://github.com/Ascend/samples)
