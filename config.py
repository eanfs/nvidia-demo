"""
配置文件 - 存储 RTSP 连接和其他配置参数
"""


class Config:
    """配置类"""

    # RTSP 流地址
    # 示例:
    # - 本地摄像头: "rtsp://username:password@192.168.1.100:554/stream1"
    # - 公共测试流: "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"
    RTSP_URL = "rtsp://localhost:8554/stream"

    # GPU 设备配置
    CUDA_DEVICE = 0  # CUDA 设备 ID
    USE_GPU = True  # 是否使用 GPU

    # 人脸检测配置
    DETECTOR_TYPE = "mtcnn"  # 'mtcnn', 'opencv', 'insightface'
    MIN_FACE_SIZE = 20  # 最小人脸尺寸（像素）
    DETECTION_CONFIDENCE = 0.7  # 检测置信度阈值

    # 视频处理配置
    PROCESS_EVERY_N_FRAMES = 1  # 每 N 帧处理一次（1 = 每帧都处理）
    FRAME_BUFFER_SIZE = 3  # 帧缓冲区大小

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
