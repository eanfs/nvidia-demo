"""
多路 RTSP 流人脸检测主程序
支持同时处理多路视频流
"""

import argparse
import time
import signal
import sys
import logging
from typing import List
from multi_stream_manager import MultiStreamManager, StreamConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MultiStreamApp:
    """多路流应用"""

    def __init__(self):
        self.manager = None
        self.running = False

        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"接收到信号 {signum}，准备退出...")
        self.stop()
        sys.exit(0)

    def _load_streams_from_file(self, filepath: str) -> List[StreamConfig]:
        """从文件加载流配置"""
        streams = []

        try:
            with open(filepath, "r") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    # 格式: stream_id,rtsp_url,priority,target_fps
                    parts = line.split(",")

                    if len(parts) >= 2:
                        stream_id = parts[0].strip() if parts[0] else f"stream_{i + 1}"
                        rtsp_url = parts[1].strip()
                        priority = int(parts[2].strip()) if len(parts) > 2 else 1
                        target_fps = int(parts[3].strip()) if len(parts) > 3 else 5

                        streams.append(
                            StreamConfig(
                                stream_id=stream_id,
                                rtsp_url=rtsp_url,
                                priority=priority,
                                target_fps=target_fps,
                            )
                        )

            logger.info(f"从文件 {filepath} 加载了 {len(streams)} 路流")
            return streams

        except Exception as e:
            logger.error(f"加载流配置文件失败: {e}")
            return []

    def start(self, config_file: str, detector_type: str = "mtcnn", device: str = "cuda", batch_size: int = 8, buffer_size: int = 100, report_interval: int = 10):
        """启动应用"""
        # 创建多路流管理器
        self.manager = MultiStreamManager(
            detector_type=detector_type,
            device=device,
            batch_size=batch_size,
            max_buffer_size=buffer_size,
        )

        # 从配置文件加载流配置
        streams = self._load_streams_from_file(config_file)

        if not streams:
            logger.error("没有可用的流配置")
            return

        # 添加流到管理器
        for stream in streams:
            self.manager.add_stream(stream)

        # 启动管理器
        self.manager.start()
        self.running = True

        logger.info(f"已启动 {len(streams)} 路流处理")

        # 性能监控循环
        try:
            last_report_time = time.time()

            while self.running:
                time.sleep(1)

                # 定期打印性能报告
                if time.time() - last_report_time >= report_interval:
                    print("\n" + self.manager.get_summary())
                    last_report_time = time.time()

        except KeyboardInterrupt:
            logger.info("用户中断")
        finally:
            self.stop()

    def stop(self):
        """停止应用"""
        if not self.running:
            return

        logger.info("正在停止应用...")
        self.running = False

        if self.manager:
            self.manager.stop()

        logger.info("应用已停止")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="多路 RTSP 流人脸检测 (NVIDIA GPU 加速)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 从配置文件加载流
  python multi_rtsp_face_detection.py streams.txt

配置文件格式 (streams.txt):
  # stream_id, rtsp_url, priority, target_fps
  cam1, rtsp://192.168.1.100:554/stream1, 1, 5
  cam2, rtsp://192.168.1.101:554/stream1, 2, 10
  cam3, rtsp://192.168.1.102:554/stream1, 1, 5
        """,
    )

    parser.add_argument("config_file", type=str, help="流配置文件路径")
    parser.add_argument(
        "--detector",
        type=str,
        default="mtcnn",
        choices=["mtcnn", "opencv", "insightface"],
        help="人脸检测器类型（默认: mtcnn）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="推理设备（默认: cuda）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="批处理大小（默认: 8）"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=100, help="帧缓冲区大小（默认: 100）"
    )
    parser.add_argument(
        "--report-interval", type=int, default=10, help="性能报告间隔（秒）（默认: 10）"
    )

    args = parser.parse_args()

    # 创建并启动应用
    app = MultiStreamApp()
    app.start(
        config_file=args.config_file,
        detector_type=args.detector,
        device=args.device,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        report_interval=args.report_interval,
    )


if __name__ == "__main__":
    main()
