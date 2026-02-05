"""
昇腾版多路 RTSP 流人脸检测主程序
基于华为 Atlas 300V NPU 进行硬件加速
"""

import argparse
import time
import signal
import sys
import logging
from typing import List
from ascend_stream_manager import AscendMultiStreamManager, StreamConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AscendMultiStreamApp:
    """昇腾多路流应用"""

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

    def start(self, config_file: str, model_path: str = None, device_id: int = 0, batch_size: int = 8, buffer_size: int = 100, use_dvpp: bool = True, report_interval: int = 10):
        """启动应用"""
        # 创建昇腾多路流管理器
        self.manager = AscendMultiStreamManager(
            model_path=model_path,
            device_id=device_id,
            batch_size=batch_size,
            max_buffer_size=buffer_size,
            use_dvpp=use_dvpp,
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

        logger.info(f"已启动 {len(streams)} 路流处理 (昇腾 Atlas 300V)")

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
        description="昇腾版多路 RTSP 流人脸检测 (Atlas 300V NPU 加速)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 从配置文件加载流
  python multi_rtsp_face_detection_ascend.py streams.txt --model models/face.om

配置文件格式 (streams.txt):
  # stream_id, rtsp_url, priority, target_fps
  cam1, rtsp://192.168.1.100:554/stream1, 1, 5
  cam2, rtsp://192.168.1.101:554/stream1, 2, 10
  cam3, rtsp://192.168.1.102:554/stream1, 1, 5

昇腾 Atlas 300V 性能参考:
  - FP16 模式: 30-40 路 1080p @ 5fps
  - INT8 模式: 50-70 路 1080p @ 5fps
  - DVPP 硬件解码: 最高 100 路 1080p @ 25fps
        """,
    )

    parser.add_argument("config_file", type=str, help="流配置文件路径")
    parser.add_argument("--model", type=str, default=None, help=".om 离线模型路径")
    parser.add_argument("--device", type=int, default=0, help="NPU 设备 ID（默认: 0）")
    parser.add_argument(
        "--batch-size", type=int, default=8, help="批处理大小（默认: 8）"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=100, help="帧缓冲区大小（默认: 100）"
    )
    parser.add_argument(
        "--report-interval", type=int, default=10, help="性能报告间隔（秒）（默认: 10）"
    )
    parser.add_argument(
        "--use-dvpp",
        action="store_true",
        default=True,
        help="使用 DVPP 硬件解码（默认: 启用）",
    )
    parser.add_argument(
        "--no-dvpp",
        action="store_false",
        dest="use_dvpp",
        help="禁用 DVPP 硬件解码",
    )

    args = parser.parse_args()

    # 打印配置信息
    print("=" * 60)
    print("昇腾 Atlas 300V 多路 RTSP 流人脸检测系统")
    print("=" * 60)
    print(f"NPU 设备:     {args.device}")
    print(f"模型路径:     {args.model or '(使用 OpenCV 后端)'}")
    print(f"批处理大小:   {args.batch_size}")
    print(f"DVPP 解码:    {'启用' if args.use_dvpp else '禁用'}")
    print("=" * 60)

    # 创建并启动应用
    app = AscendMultiStreamApp()
    app.start(
        config_file=args.config_file,
        model_path=args.model,
        device_id=args.device,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        use_dvpp=args.use_dvpp,
        report_interval=args.report_interval,
    )


if __name__ == "__main__":
    main()
