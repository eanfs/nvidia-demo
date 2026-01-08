# 70 路 1080p 优化配置

## 系统配置
STREAM_COUNT = 70
INPUT_RESOLUTION = "1080p"  # 输入分辨率
DECODE_RESOLUTION = "960x540"  # 解码分辨率（降低以突破 NVDEC 瓶颈）
TARGET_FPS = 5  # 每路检测帧率

## GPU 配置
GPU_DEVICE = 0
USE_TENSORRT = True
TENSORRT_ENGINE = "yolov5s-face-int8.engine"
PRECISION = "int8"  # int8, fp16, fp32

## 批处理配置
BATCH_SIZE = 16  # 最优 batch size
MIN_BATCH_SIZE = 8
MAX_BATCH_SIZE = 32
BATCH_TIMEOUT_MS = 10  # 等待超时（毫秒）

## 检测器配置
DETECTOR_TYPE = "tensorrt"
INPUT_SIZE = (640, 640)  # 检测输入尺寸
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45

## 帧采样配置
SAMPLE_STRATEGY = "adaptive"  # fixed, adaptive
FIXED_SAMPLE_INTERVAL = 5  # 每 5 帧采样 1 帧（25fps → 5fps）
ADAPTIVE_HIGH_FPS = 8  # 检测到人脸时的采样间隔
ADAPTIVE_LOW_FPS = 10  # 无人脸时的采样间隔

## 解码配置
USE_HARDWARE_DECODE = True
DECODE_BACKEND = "ffmpeg"  # ffmpeg, opencv
# NVDEC 前 30 路，CPU 后 40 路
NVDEC_STREAM_LIMIT = 30
CPU_DECODE_THREADS = 16  # CPU 解码线程数

## 内存配置
FRAME_BUFFER_SIZE = 150  # 每路流的帧缓冲区大小
USE_ZERO_COPY = True  # 使用 Zero-Copy（CUDA Unified Memory）
ENABLE_MEMORY_POOL = True  # 启用内存池

## 性能优化
ENABLE_TF32 = True  # 启用 TF32（Tensor Core 自动优化）
ENABLE_CUDNN_BENCHMARK = True  # cuDNN 自动调优
NUM_WORKER_THREADS = 4  # 工作线程数

## 监控配置
ENABLE_MONITORING = True
MONITORING_INTERVAL = 10  # 性能报告间隔（秒）
LOG_GPU_METRICS = True
LOG_STREAM_METRICS = True

## RTSP 配置
RTSP_TRANSPORT = "tcp"  # tcp, udp
RTSP_TIMEOUT = 10  # 超时（秒）
MAX_RECONNECT_ATTEMPTS = 3
RECONNECT_DELAY = 2

## 流优先级配置
# 高优先级流使用更高的检测帧率
PRIORITY_HIGH_FPS = 10
PRIORITY_MEDIUM_FPS = 5
PRIORITY_LOW_FPS = 3

## 优化配置总结
"""
关键优化措施：
1. ✅ TensorRT INT8 量化 - 4-8x 性能提升
2. ✅ 降低解码分辨率到 960x540 - 突破 NVDEC 瓶颈
3. ✅ 动态批处理 batch=16 - GPU 利用率最大化
4. ✅ 智能帧采样 - 降低计算量
5. ✅ CPU 辅助解码 - 后 40 路使用 CPU

预期性能：
- 70 路 1080p @ 5fps 检测
- GPU 利用率: 80-85%
- 显存占用: < 22GB
- 单流延迟: < 200ms
"""
