"""
性能监控工具 - 监控 GPU 使用率、显存、温度等
"""

import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GPUMetrics:
    """GPU 性能指标"""

    gpu_id: int
    name: str
    utilization: float  # GPU 使用率 (%)
    memory_used_mb: float  # 已用显存 (MB)
    memory_total_mb: float  # 总显存 (MB)
    memory_utilization: float  # 显存使用率 (%)
    temperature: float  # 温度 (°C)
    power_draw_w: float  # 功耗 (W)
    timestamp: datetime


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, gpu_id: int = 0):
        """
        初始化性能监控器

        Args:
            gpu_id: GPU 设备 ID
        """
        self.gpu_id = gpu_id
        self.has_torch = False
        self.has_pynvml = False

        # 尝试导入 PyTorch
        try:
            import torch

            if torch.cuda.is_available():
                self.torch = torch
                self.has_torch = True
                logger.info("已启用 PyTorch GPU 监控")
        except ImportError:
            logger.warning("未安装 PyTorch，部分监控功能不可用")

        # 尝试导入 pynvml (NVIDIA Management Library)
        try:
            import pynvml

            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            self.has_pynvml = True
            logger.info("已启用 NVML 监控")
        except Exception as e:
            logger.warning(f"未能初始化 NVML: {e}")

    def get_gpu_metrics(self) -> Optional[GPUMetrics]:
        """获取 GPU 性能指标"""

        if self.has_pynvml:
            return self._get_metrics_nvml()
        elif self.has_torch:
            return self._get_metrics_torch()
        else:
            logger.error("无可用的 GPU 监控方法")
            return None

    def _get_metrics_nvml(self) -> GPUMetrics:
        """使用 NVML 获取指标（最准确）"""
        try:
            # GPU 名称
            name = self.pynvml.nvmlDeviceGetName(self.nvml_handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            # GPU 使用率
            utilization = self.pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
            gpu_util = utilization.gpu

            # 显存信息
            mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
            memory_used_mb = mem_info.used / 1024 / 1024
            memory_total_mb = mem_info.total / 1024 / 1024
            memory_util = (mem_info.used / mem_info.total) * 100

            # 温度
            temperature = self.pynvml.nvmlDeviceGetTemperature(
                self.nvml_handle, self.pynvml.NVML_TEMPERATURE_GPU
            )

            # 功耗
            try:
                power_draw_w = (
                    self.pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle) / 1000
                )
            except:
                power_draw_w = 0.0

            return GPUMetrics(
                gpu_id=self.gpu_id,
                name=name,
                utilization=gpu_util,
                memory_used_mb=memory_used_mb,
                memory_total_mb=memory_total_mb,
                memory_utilization=memory_util,
                temperature=temperature,
                power_draw_w=power_draw_w,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"NVML 获取指标失败: {e}")
            return None

    def _get_metrics_torch(self) -> GPUMetrics:
        """使用 PyTorch 获取指标（基本信息）"""
        try:
            # GPU 名称
            name = self.torch.cuda.get_device_name(self.gpu_id)

            # 显存信息
            memory_used_mb = self.torch.cuda.memory_allocated(self.gpu_id) / 1024 / 1024
            memory_total_mb = (
                self.torch.cuda.get_device_properties(self.gpu_id).total_memory
                / 1024
                / 1024
            )
            memory_util = (memory_used_mb / memory_total_mb) * 100

            return GPUMetrics(
                gpu_id=self.gpu_id,
                name=name,
                utilization=0.0,  # PyTorch 不提供
                memory_used_mb=memory_used_mb,
                memory_total_mb=memory_total_mb,
                memory_utilization=memory_util,
                temperature=0.0,  # PyTorch 不提供
                power_draw_w=0.0,  # PyTorch 不提供
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"PyTorch 获取指标失败: {e}")
            return None

    def print_metrics(self, metrics: GPUMetrics):
        """打印性能指标"""
        print("\n" + "=" * 80)
        print(f"GPU 性能监控 - {metrics.name} (GPU {metrics.gpu_id})")
        print("=" * 80)
        print(f"时间:         {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"GPU 使用率:   {metrics.utilization:>6.1f}%")
        print(
            f"显存使用:     {metrics.memory_used_mb:>7.0f} MB / {metrics.memory_total_mb:>7.0f} MB ({metrics.memory_utilization:>5.1f}%)"
        )
        print(f"温度:         {metrics.temperature:>6.1f}°C")
        print(f"功耗:         {metrics.power_draw_w:>6.1f}W")
        print("=" * 80 + "\n")

    def monitor_loop(self, interval: float = 1.0):
        """监控循环"""
        logger.info(f"开始 GPU 性能监控，间隔 {interval} 秒")

        try:
            while True:
                metrics = self.get_gpu_metrics()
                if metrics:
                    self.print_metrics(metrics)
                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("监控已停止")

    def __del__(self):
        """清理资源"""
        if self.has_pynvml:
            try:
                self.pynvml.nvmlShutdown()
            except:
                pass


class StreamPerformanceTracker:
    """流性能追踪器"""

    def __init__(self):
        self.stream_stats: Dict[str, Dict] = {}
        self.start_time = time.time()

    def update(
        self,
        stream_id: str,
        frames_processed: int,
        faces_detected: int,
        latency_ms: float,
    ):
        """更新流统计"""
        if stream_id not in self.stream_stats:
            self.stream_stats[stream_id] = {
                "frames": 0,
                "faces": 0,
                "total_latency": 0.0,
                "count": 0,
                "start_time": time.time(),
            }

        stats = self.stream_stats[stream_id]
        stats["frames"] += frames_processed
        stats["faces"] += faces_detected
        stats["total_latency"] += latency_ms
        stats["count"] += 1

    def get_summary(self) -> str:
        """获取性能摘要"""
        elapsed = time.time() - self.start_time

        summary = ["\n" + "=" * 80, f"流性能统计 (运行时间: {elapsed:.1f}s)", "=" * 80]

        for stream_id, stats in self.stream_stats.items():
            fps = stats["frames"] / max(elapsed, 1)
            avg_latency = stats["total_latency"] / max(stats["count"], 1)

            summary.append(
                f"[{stream_id}] "
                f"帧数: {stats['frames']} | "
                f"FPS: {fps:.1f} | "
                f"人脸: {stats['faces']} | "
                f"平均延迟: {avg_latency:.1f}ms"
            )

        summary.append("=" * 80 + "\n")
        return "\n".join(summary)


def main():
    """主函数 - 独立运行性能监控"""
    import argparse

    parser = argparse.ArgumentParser(description="GPU 性能监控工具")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU 设备 ID (默认: 0)")
    parser.add_argument(
        "--interval", type=float, default=1.0, help="监控间隔（秒）(默认: 1.0)"
    )

    args = parser.parse_args()

    # 创建监控器
    monitor = PerformanceMonitor(gpu_id=args.gpu_id)

    # 开始监控
    monitor.monitor_loop(interval=args.interval)


if __name__ == "__main__":
    main()
