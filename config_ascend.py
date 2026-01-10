"""
昇腾 Atlas 300V 配置文件
配置 NPU 连接和其他运行参数
"""


class AscendConfig:
    """昇腾平台配置类"""

    # RTSP 流地址
    RTSP_URL = "rtsp://localhost:8554/stream"

    # NPU 设备配置
    DEVICE_ID = 0  # NPU 设备 ID
    USE_NPU = True  # 是否使用 NPU

    # Atlas 300V 规格参数
    # - 算力: 22 TOPS INT8 / 11 TFLOPS FP16
    # - 显存: 16GB HBM2
    # - 视频解码: 32路 H.264/H.265 1080p30
    # - 功耗: 75W TDP

    # 人脸检测配置
    DETECTOR_TYPE = "acl"  # 'acl' (ACL推理), 'torch_npu' (PyTorch NPU)
    MIN_FACE_SIZE = 20  # 最小人脸尺寸（像素）
    DETECTION_CONFIDENCE = 0.7  # 检测置信度阈值
    MODEL_PATH = "models/face_detection.om"  # 离线模型路径

    # 模型精度配置
    # Atlas 300V 支持: FP32, FP16, INT8
    MODEL_PRECISION = "FP16"  # 推荐使用 FP16，性能/精度平衡最佳

    # 视频处理配置
    PROCESS_EVERY_N_FRAMES = 1  # 每 N 帧处理一次
    FRAME_BUFFER_SIZE = 3  # 帧缓冲区大小

    # DVPP 硬件解码配置
    USE_DVPP_DECODE = True  # 使用 DVPP 硬件解码
    DVPP_PIXEL_FORMAT = "YUV420SP_U8"  # DVPP 输出格式
    MAX_DECODE_STREAMS = 32  # Atlas 300V 最大解码路数

    # 显示配置
    DISPLAY_WINDOW = True  # 是否显示窗口
    SHOW_FPS = True  # 是否显示 FPS
    SHOW_CONFIDENCE = True  # 是否显示检测置信度

    # 输出配置
    SAVE_OUTPUT = False  # 是否保存输出视频
    OUTPUT_PATH = "output.mp4"  # 输出视频路径

    # RTSP 连接配置
    RTSP_TRANSPORT = "tcp"  # 'tcp' 或 'udp'
    MAX_RECONNECT_ATTEMPTS = 3  # 最大重连次数
    RECONNECT_DELAY = 2  # 重连延迟（秒）

    # 性能优化
    USE_HARDWARE_DECODE = True  # 使用硬件解码
    NUM_THREADS = 4  # 线程数

    # CANN/ACL 配置
    CANN_HOME = "/usr/local/Ascend/ascend-toolkit/latest"
    ACL_JSON_PATH = "acl.json"  # ACL 配置文件路径

    # 批处理配置
    BATCH_SIZE = 8  # 批处理大小
    MAX_BATCH_WAIT_MS = 50  # 最大批处理等待时间（毫秒）

    # 内存配置
    INPUT_BUFFER_SIZE = 1920 * 1080 * 3  # 输入缓冲区大小
    OUTPUT_BUFFER_SIZE = 1024 * 1024  # 输出缓冲区大小

    # 多流配置
    MAX_STREAMS = 70  # 最大流数量（基于 Atlas 300V 性能）
    STREAM_PRIORITY_LEVELS = 10  # 优先级级别数

    # 性能预估 (Atlas 300V)
    # FP32: 15-20 路 1080p @ 5fps
    # FP16: 30-40 路 1080p @ 5fps (推荐)
    # INT8: 50-70 路 1080p @ 5fps


class AscendOptimConfig:
    """昇腾优化配置"""

    # ATC 模型转换配置
    ATC_FRAMEWORK = "onnx"  # 源模型框架: onnx, tensorflow, pytorch
    ATC_SOC_VERSION = "Ascend310P"  # Atlas 300V 使用 Ascend 310P 芯片

    # 模型输入配置
    INPUT_FORMAT = "NCHW"  # 输入格式
    INPUT_SHAPE = "1,3,640,640"  # 输入尺寸 (batch, channels, height, width)

    # 量化配置
    QUANTIZE_MODE = "INT8"  # 量化模式: FP16, INT8
    CALIBRATION_DATA = "calibration_images/"  # INT8 校准数据路径

    # 图优化
    ENABLE_GRAPH_FUSION = True  # 启用图融合优化
    ENABLE_BUFFER_OPTIMIZE = True  # 启用缓冲区优化


class DVPPConfig:
    """DVPP 视频处理配置"""

    # 解码配置
    DECODE_MODE = "DVPP_DECODE_MODE_IPB"  # 解码模式
    PIXEL_FORMAT = "DVPP_PIXEL_FORMAT_YUV_SEMIPLANAR_420"

    # 视频编码支持
    SUPPORTED_CODECS = ["H264", "H265", "JPEG"]

    # 分辨率配置
    MAX_WIDTH = 4096  # 最大输入宽度
    MAX_HEIGHT = 4096  # 最大输入高度
    MIN_WIDTH = 128  # 最小输入宽度
    MIN_HEIGHT = 128  # 最小输入高度

    # 图像处理配置
    RESIZE_MODE = "DVPP_RESIZE_BILINEAR"  # 缩放模式
    VPC_BATCH_SIZE = 8  # VPC 批处理大小
