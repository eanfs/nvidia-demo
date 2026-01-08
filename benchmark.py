"""
性能基准测试脚本 - 测试不同配置下的多路流性能
"""

import time
import argparse
import logging
from typing import List
from multi_stream_manager import MultiStreamManager, StreamConfig
from performance_monitor import PerformanceMonitor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Benchmark:
    """基准测试类"""

    def __init__(self, args):
        self.args = args
        self.gpu_monitor = PerformanceMonitor(gpu_id=0)
        self.results = []

    def create_test_streams(self, num_streams: int) -> List[StreamConfig]:
        """创建测试流配置"""
        streams = []

        # 使用公共测试流
        base_url = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"

        for i in range(num_streams):
            streams.append(
                StreamConfig(
                    stream_id=f"bench_stream_{i + 1:02d}",
                    rtsp_url=base_url,
                    priority=1,
                    target_fps=self.args.target_fps,
                )
            )

        return streams

    def run_test(self, num_streams: int, batch_size: int, duration: int = 60):
        """运行单个测试"""
        logger.info(f"\n{'=' * 80}")
        logger.info(f"开始测试: {num_streams} 路流, Batch={batch_size}")
        logger.info(f"{'=' * 80}\n")

        # 创建管理器
        manager = MultiStreamManager(
            detector_type=self.args.detector,
            device=self.args.device,
            batch_size=batch_size,
            max_buffer_size=100,
        )

        # 添加流
        streams = self.create_test_streams(num_streams)
        for stream in streams:
            manager.add_stream(stream)

        # 启动
        manager.start()

        # 运行测试时长
        start_time = time.time()
        gpu_metrics_list = []

        try:
            while time.time() - start_time < duration:
                # 采集 GPU 指标
                gpu_metrics = self.gpu_monitor.get_gpu_metrics()
                if gpu_metrics:
                    gpu_metrics_list.append(gpu_metrics)

                time.sleep(1)

                # 每10秒打印一次状态
                if int(time.time() - start_time) % 10 == 0:
                    print(manager.get_summary())

        except KeyboardInterrupt:
            logger.info("测试被中断")

        finally:
            # 停止管理器
            manager.stop()

        # 收集结果
        elapsed = time.time() - start_time
        stream_metrics = manager.get_metrics()

        total_frames = sum(m.frames_processed for m in stream_metrics.values())
        total_faces = sum(m.faces_detected for m in stream_metrics.values())
        avg_fps = total_frames / max(elapsed, 1)

        # 计算平均 GPU 指标
        if gpu_metrics_list:
            avg_gpu_util = sum(m.utilization for m in gpu_metrics_list) / len(
                gpu_metrics_list
            )
            avg_mem_util = sum(m.memory_utilization for m in gpu_metrics_list) / len(
                gpu_metrics_list
            )
            avg_temp = sum(m.temperature for m in gpu_metrics_list) / len(
                gpu_metrics_list
            )
            avg_power = sum(m.power_draw_w for m in gpu_metrics_list) / len(
                gpu_metrics_list
            )
        else:
            avg_gpu_util = avg_mem_util = avg_temp = avg_power = 0.0

        result = {
            "num_streams": num_streams,
            "batch_size": batch_size,
            "duration": elapsed,
            "total_frames": total_frames,
            "total_faces": total_faces,
            "avg_fps": avg_fps,
            "fps_per_stream": avg_fps / num_streams if num_streams > 0 else 0,
            "avg_gpu_util": avg_gpu_util,
            "avg_mem_util": avg_mem_util,
            "avg_temp": avg_temp,
            "avg_power": avg_power,
        }

        self.results.append(result)

        # 打印结果
        self.print_result(result)

        return result

    def print_result(self, result):
        """打印测试结果"""
        print(f"\n{'=' * 80}")
        print("测试结果")
        print(f"{'=' * 80}")
        print(f"流数量:           {result['num_streams']} 路")
        print(f"批处理大小:       {result['batch_size']}")
        print(f"测试时长:         {result['duration']:.1f} 秒")
        print(f"总处理帧数:       {result['total_frames']}")
        print(f"总检测人脸:       {result['total_faces']}")
        print(f"平均总FPS:        {result['avg_fps']:.2f}")
        print(f"平均单流FPS:      {result['fps_per_stream']:.2f}")
        print(f"-" * 80)
        print(f"平均GPU使用率:    {result['avg_gpu_util']:.1f}%")
        print(f"平均显存使用率:   {result['avg_mem_util']:.1f}%")
        print(f"平均温度:         {result['avg_temp']:.1f}°C")
        print(f"平均功耗:         {result['avg_power']:.1f}W")
        print(f"{'=' * 80}\n")

    def print_summary(self):
        """打印所有测试的汇总"""
        if not self.results:
            logger.warning("没有测试结果")
            return

        print(f"\n{'=' * 80}")
        print("基准测试汇总")
        print(f"{'=' * 80}\n")
        print(
            f"{'流数':>6} | {'Batch':>6} | {'FPS/流':>8} | {'GPU%':>6} | {'显存%':>6} | {'温度':>6} | {'功耗':>6}"
        )
        print(
            f"{'-' * 6}-+-{'-' * 6}-+-{'-' * 8}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 6}"
        )

        for r in self.results:
            print(
                f"{r['num_streams']:>6} | "
                f"{r['batch_size']:>6} | "
                f"{r['fps_per_stream']:>8.2f} | "
                f"{r['avg_gpu_util']:>6.1f} | "
                f"{r['avg_mem_util']:>6.1f} | "
                f"{r['avg_temp']:>6.1f} | "
                f"{r['avg_power']:>6.1f}"
            )

        print(f"{'=' * 80}\n")

    def run_progressive_test(self):
        """渐进式测试 - 逐步增加流数量"""
        logger.info("开始渐进式基准测试")

        start_streams = self.args.start_streams
        end_streams = self.args.end_streams
        step_streams = self.args.step_streams
        batch_size = self.args.batch_size
        duration = self.args.duration

        current_streams = start_streams

        while current_streams <= end_streams:
            self.run_test(
                num_streams=current_streams, batch_size=batch_size, duration=duration
            )

            current_streams += step_streams

            # 等待冷却
            logger.info("等待 GPU 冷却...")
            time.sleep(5)

        # 打印汇总
        self.print_summary()

    def run_batch_size_test(self):
        """批大小测试 - 测试不同 batch size 的影响"""
        logger.info("开始批大小基准测试")

        num_streams = self.args.num_streams
        batch_sizes = self.args.batch_sizes or [1, 2, 4, 8, 16, 32]
        duration = self.args.duration

        for batch_size in batch_sizes:
            self.run_test(
                num_streams=num_streams, batch_size=batch_size, duration=duration
            )

            # 等待冷却
            logger.info("等待 GPU 冷却...")
            time.sleep(5)

        # 打印汇总
        self.print_summary()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="多路流性能基准测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
测试模式:

  1. 渐进式测试 (默认)
     逐步增加流数量，测试最大承载能力
     
     python benchmark.py --mode progressive \\
         --start-streams 5 --end-streams 30 --step-streams 5

  2. 批大小测试
     固定流数量，测试不同 batch size 的性能
     
     python benchmark.py --mode batch-size \\
         --num-streams 20 --batch-sizes 1 2 4 8 16 32
        """,
    )

    # 测试模式
    parser.add_argument(
        "--mode",
        type=str,
        default="progressive",
        choices=["progressive", "batch-size"],
        help="测试模式",
    )

    # 渐进式测试参数
    parser.add_argument(
        "--start-streams", type=int, default=5, help="起始流数量 (默认: 5)"
    )
    parser.add_argument(
        "--end-streams", type=int, default=30, help="结束流数量 (默认: 30)"
    )
    parser.add_argument(
        "--step-streams", type=int, default=5, help="每次增加流数量 (默认: 5)"
    )

    # 批大小测试参数
    parser.add_argument(
        "--num-streams", type=int, default=20, help="流数量 (批大小测试用) (默认: 20)"
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", help="批大小列表 (默认: 1 2 4 8 16 32)"
    )

    # 通用参数
    parser.add_argument(
        "--detector",
        type=str,
        default="mtcnn",
        choices=["mtcnn", "opencv", "insightface"],
        help="检测器类型 (默认: mtcnn)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="推理设备 (默认: cuda)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="批处理大小 (默认: 8)"
    )
    parser.add_argument(
        "--target-fps", type=int, default=5, help="目标检测帧率 (默认: 5)"
    )
    parser.add_argument(
        "--duration", type=int, default=60, help="每个测试持续时间（秒） (默认: 60)"
    )

    args = parser.parse_args()

    # 创建基准测试
    benchmark = Benchmark(args)

    # 运行测试
    if args.mode == "progressive":
        benchmark.run_progressive_test()
    elif args.mode == "batch-size":
        benchmark.run_batch_size_test()


if __name__ == "__main__":
    main()
