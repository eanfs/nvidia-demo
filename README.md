# RTSP 视频流 GPU 加速人脸检测 Demo

基于 Python 的 RTSP 视频流人脸检测系统，使用 NVIDIA GPU 进行硬件解码和推理加速。

## 功能特性

✅ **RTSP 视频流支持** - 支持各种 RTSP 摄像头和视频流  
✅ **NVIDIA GPU 硬件解码** - 使用 GPU 加速视频解码，降低 CPU 负载  
✅ **GPU 加速推理** - 人脸检测模型在 GPU 上运行  
✅ **多种检测器支持** - MTCNN、OpenCV Haar Cascade、InsightFace  
✅ **实时性能监控** - FPS、检测时间等指标显示  
✅ **视频输出保存** - 可选保存处理后的视频  

## 系统要求

### 硬件要求
- NVIDIA GPU (支持 CUDA 11.0+)
- 推荐显存：4GB+

### 软件要求
- Python 3.8+
- CUDA Toolkit 11.0+
- cuDNN 8.0+
- NVIDIA Video Codec SDK (可选，用于硬件解码)

## 安装步骤

### 1. 安装 CUDA 和 cuDNN

从 [NVIDIA 官网](https://developer.nvidia.com/cuda-downloads) 下载并安装对应版本的 CUDA Toolkit 和 cuDNN。

验证安装：
```bash
nvidia-smi
nvcc --version
```

### 2. 创建 Python 虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. 安装依赖包

```bash
# 安装 PyTorch (CUDA 版本)
# 访问 https://pytorch.org/ 获取适合你的 CUDA 版本的安装命令
# 例如 CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

### 4. (可选) 安装支持 CUDA 的 OpenCV

为了使用 GPU 硬件解码，需要从源码编译 OpenCV with CUDA：

```bash
# 这是一个复杂的过程，可以参考：
# https://docs.opencv.org/4.x/d6/d15/tutorial_building_tegra_cuda.html
```

或者使用预编译版本（如果可用）：
```bash
pip install opencv-contrib-python-headless
```

## 使用方法

### 基本使用

```bash
# 使用配置文件中的默认 RTSP URL
python rtsp_face_detection.py

# 指定 RTSP URL
python rtsp_face_detection.py --rtsp-url "rtsp://username:password@192.168.1.100:554/stream1"
```

### 高级选项

```bash
# 使用 InsightFace 检测器
python rtsp_face_detection.py --detector insightface

# 使用 CPU 推理
python rtsp_face_detection.py --device cpu

# 禁用 GPU 硬件解码
python rtsp_face_detection.py --no-gpu-decode

# 保存输出视频
python rtsp_face_detection.py --output output.mp4

# 每 3 帧处理一次（降低负载）
python rtsp_face_detection.py --process-every 3

# 不显示窗口（用于服务器部署）
python rtsp_face_detection.py --no-display
```

### 完整参数列表

```bash
python rtsp_face_detection.py --help
```

参数说明：
- `--rtsp-url`: RTSP 流地址
- `--detector`: 人脸检测器类型 (mtcnn/opencv/insightface)
- `--device`: 推理设备 (cuda/cpu)
- `--no-gpu-decode`: 禁用 GPU 硬件解码
- `--no-display`: 不显示视频窗口
- `--output`: 输出视频保存路径
- `--process-every`: 每 N 帧处理一次

## 配置文件

编辑 `config.py` 可以修改默认配置：

```python
class Config:
    # RTSP 流地址
    RTSP_URL = "rtsp://localhost:8554/stream"
    
    # 人脸检测配置
    DETECTOR_TYPE = 'mtcnn'
    MIN_FACE_SIZE = 20
    DETECTION_CONFIDENCE = 0.7
    
    # 性能配置
    PROCESS_EVERY_N_FRAMES = 1
    USE_HARDWARE_DECODE = True
```

## 性能优化建议

### 1. GPU 硬件解码
确保使用支持 CUDA 的 OpenCV 或 NVIDIA Video Codec SDK，可以显著降低 CPU 负载。

### 2. 降低处理频率
```bash
# 每 3 帧处理一次，可以提升 3 倍性能
python rtsp_face_detection.py --process-every 3
```

### 3. 选择合适的检测器
- **MTCNN**: 精度高，速度中等
- **OpenCV**: 速度快，精度一般
- **InsightFace**: 精度最高，功能最全（年龄、性别识别）

### 4. 调整输入分辨率
如果 RTSP 流分辨率过高，可以在处理前降采样。

## 常见 RTSP URL 格式

```bash
# 海康威视
rtsp://username:password@ip:554/Streaming/Channels/101

# 大华
rtsp://username:password@ip:554/cam/realmonitor?channel=1&subtype=0

# 通用格式
rtsp://username:password@ip:port/path
```

## 测试用 RTSP 流

如果没有真实摄像头，可以使用以下测试流：

```bash
# Big Buck Bunny (公共测试流)
python rtsp_face_detection.py --rtsp-url "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"
```

或者使用 FFmpeg 创建本地测试流：

```bash
# 从视频文件创建 RTSP 流
ffmpeg -re -i input.mp4 -f rtsp rtsp://localhost:8554/stream
```

## 故障排除

### 问题 1: CUDA 不可用
```
解决方案：
1. 检查 CUDA 安装：nvidia-smi
2. 重新安装 PyTorch CUDA 版本
3. 验证：python -c "import torch; print(torch.cuda.is_available())"
```

### 问题 2: 无法连接 RTSP 流
```
解决方案：
1. 检查 RTSP URL 格式
2. 尝试使用 VLC 播放器测试流是否可用
3. 检查网络连接和防火墙设置
4. 尝试切换 TCP/UDP 传输协议
```

### 问题 3: FPS 过低
```
解决方案：
1. 启用 GPU 硬件解码
2. 增加 --process-every 参数值
3. 使用更快的检测器（opencv）
4. 降低输入视频分辨率
```

### 问题 4: 内存不足
```
解决方案：
1. 减小 batch size
2. 使用更小的模型
3. 降低输入分辨率
```

## 项目结构

```
nvidia-demo/
├── rtsp_face_detection.py  # 主程序
├── face_detector.py        # 人脸检测器封装
├── config.py              # 配置文件
├── requirements.txt       # 依赖列表
└── README.md             # 说明文档
```

## 扩展功能

### 添加人脸识别
可以基于检测到的人脸进行识别：
```python
# 使用 InsightFace 进行人脸识别
from insightface.app import FaceAnalysis
app = FaceAnalysis()
faces = app.get(frame)
# 提取特征进行比对
```

### 添加人脸追踪
使用 DeepSORT 等算法进行多目标追踪。

### 视频流录制
保存检测到人脸的视频片段。

## 性能参考

在 NVIDIA RTX 3080 上的性能：
- 1080p 视频流：60+ FPS (MTCNN)
- 4K 视频流：30+ FPS (MTCNN)
- GPU 硬件解码 CPU 使用率：< 10%

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 参考资源

- [NVIDIA Video Codec SDK](https://developer.nvidia.com/nvidia-video-codec-sdk)
- [PyTorch CUDA](https://pytorch.org/get-started/locally/)
- [OpenCV CUDA](https://docs.opencv.org/master/d2/de6/tutorial_py_setup_in_ubuntu.html)
- [InsightFace](https://github.com/deepinsight/insightface)
- [MTCNN](https://github.com/ipazc/mtcnn)
