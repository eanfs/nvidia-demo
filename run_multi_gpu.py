#!/usr/bin/env python
"""
多 GPU 并行处理 RTSP 流启动脚本
自动将流分配到多个 GPU 上并行处理
"""

import argparse
import signal
import sys
import time
import logging
from multi_gpu_manager import MultiGPUManager
from multi_stream_manager import StreamConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_streams_from_file(filepath: str):
    """从文件加载流配置"""
    streams = []

    try:
        with open(filepath, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

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


def main():
    parser = argparse.ArgumentParser(
        description="多 GPU 并行处理 RTSP 流",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 使用 8 个 GPU 处理配置文件中的流（轮询分配）
  python run_multi_gpu.py --config-file streams.txt --num-gpus 8

  # 使用 4 个 GPU，按优先级分配
  python run_multi_gpu.py --config-file streams.txt --num-gpus 4 --load-balance priority

  # 使用 8 个 GPU，自动负载均衡
  python run_multi_gpu.py --config-file streams.txt --num-gpus 8 --load-balance auto

  # 自定义批大小和检测器
  python run_multi_gpu.py --config-file streams.txt --num-gpus 8 \\
      --detector mtcnn --batch-size 16
        """,
    )

    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="流配置文件路径",
    )

    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="GPU 数量（默认: 8）",
    )

    parser.add_argument(
        "--detector",
        type=str,
        default="mtcnn",
        choices=["mtcnn", "opencv", "insightface"],
        help="人脸检测器类型（默认: mtcnn）",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="每个 GPU 的批处理大小（默认: 8）",
    )

    parser.add_argument(
        "--buffer-size",
        type=int,
        default=100,
        help="帧缓冲区大小（默认: 100）",
    )

    parser.add_argument(
        "--load-balance",
        type=str,
        default="round_robin",
        choices=["round_robin", "priority", "auto"],
        help="负载均衡策略（默认: round_robin）",
    )

    args = parser.parse_args()

    streams = load_streams_from_file(args.config_file)

    if not streams:
        logger.error("没有可用的流配置")
        return 1

    logger.info(f"=" * 80)
    logger.info(f"多 GPU 并行处理启动")
    logger.info(f"=" * 80)
    logger.info(f"GPU 数量: {args.num_gpus}")
    logger.info(f"总流数: {len(streams)}")
    logger.info(f"检测器: {args.detector}")
    logger.info(f"批大小: {args.batch_size}")
    logger.info(f"负载均衡: {args.load_balance}")
    logger.info(f"=" * 80)

    manager = MultiGPUManager(
        num_gpus=args.num_gpus,
        detector_type=args.detector,
        batch_size=args.batch_size,
        max_buffer_size=args.buffer_size,
        load_balance=args.load_balance,
    )

    def signal_handler(signum, frame):
        logger.info(f"接收到信号 {signum}，正在停止...")
        manager.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        manager.start(streams)

        logger.info("所有 GPU 工作进程已启动，按 Ctrl+C 停止")

        while True:
            time.sleep(5)
            status = manager.get_status()
            logger.info(
                f"运行状态: {status['active_workers']}/{status['num_gpus']} GPU 活跃"
            )

    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"运行错误: {e}", exc_info=True)
    finally:
        manager.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
