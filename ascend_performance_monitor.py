"""
昇腾 NPU 性能监控工具
监控昇腾芯片的使用率、内存、功耗等指标
"""

import time
import logging
import subprocess
import re
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NPUMetrics:
    """NPU 性能指标"""

    device_id: int
    name: str
    chip_name: str
    aicore_utilization: float  # AI Core 使用率 (%)
    memory_used_mb: float  # 已用内存 (MB)
    memory_total_mb: float  # 总内存 (MB)
    memory_utilization: float  # 内存使用率 (%)
    temperature: float  # 温度 (°C)
    power_draw_w: float  # 功耗 (W)
    health_status: str  # 健康状态
    timestamp: datetime


class AscendPerformanceMonitor:
    """昇腾性能监控器"""

    def __init__(self, device_id: int = 0):
        """
        初始化性能监控器

        Args:
            device_id: NPU 设备 ID
        """
        self.device_id = device_id
        self.has_npu_smi = False
        self.has_acl = False

        # 检测可用的监控方法
        self._check_monitoring_methods()

    def _check_monitoring_methods(self):
        """检测可用的监控方法"""
        # 检查 npu-smi 命令
        try:
            result = subprocess.run(
                ["npu-smi", "info"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self.has_npu_smi = True
                logger.info("已启用 npu-smi 监控")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("npu-smi 不可用")

        # 检查 ACL Python 库
        try:
            import acl

            self.acl = acl
            self.has_acl = True
            logger.info("已启用 ACL 监控")
        except ImportError:
            logger.warning("ACL Python 库不可用")

    def get_npu_metrics(self) -> Optional[NPUMetrics]:
        """获取 NPU 性能指标"""
        if self.has_npu_smi:
            return self._get_metrics_npu_smi()
        elif self.has_acl:
            return self._get_metrics_acl()
        else:
            logger.error("无可用的 NPU 监控方法")
            return None

    def _get_metrics_npu_smi(self) -> Optional[NPUMetrics]:
        """使用 npu-smi 获取指标"""
        try:
            # 获取设备信息
            result = subprocess.run(
                ["npu-smi", "info", "-t", "common", "-i", str(self.device_id)],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                logger.error(f"npu-smi 执行失败: {result.stderr}")
                return None

            output = result.stdout

            # 解析输出
            metrics = self._parse_npu_smi_output(output)

            return metrics

        except subprocess.TimeoutExpired:
            logger.error("npu-smi 执行超时")
            return None
        except Exception as e:
            logger.error(f"npu-smi 获取指标失败: {e}")
            return None

    def _parse_npu_smi_output(self, output: str) -> NPUMetrics:
        """解析 npu-smi 输出"""
        # 默认值
        name = "Ascend 310P"
        chip_name = "Ascend 310P"
        aicore_util = 0.0
        memory_used = 0.0
        memory_total = 16384.0  # Atlas 300V 默认 16GB
        temperature = 0.0
        power_draw = 0.0
        health_status = "Unknown"

        # 解析各项指标
        lines = output.strip().split("\n")

        for line in lines:
            line = line.strip()

            # 芯片名称
            if "Chip Name" in line:
                match = re.search(r":\s*(.+)", line)
                if match:
                    chip_name = match.group(1).strip()

            # AI Core 使用率
            if "Aicore Usage Rate" in line or "AI Core" in line:
                match = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
                if match:
                    aicore_util = float(match.group(1))

            # 内存使用
            if "HBM Usage" in line or "Memory" in line:
                match = re.search(r"(\d+)\s*/\s*(\d+)", line)
                if match:
                    memory_used = float(match.group(1))
                    memory_total = float(match.group(2))

            # 温度
            if "Temperature" in line or "Temp" in line:
                match = re.search(r"(\d+(?:\.\d+)?)\s*[°C]?", line)
                if match:
                    temperature = float(match.group(1))

            # 功耗
            if "Power" in line:
                match = re.search(r"(\d+(?:\.\d+)?)\s*W", line)
                if match:
                    power_draw = float(match.group(1))

            # 健康状态
            if "Health Status" in line or "Status" in line:
                if "OK" in line or "Normal" in line:
                    health_status = "OK"
                elif "Warning" in line:
                    health_status = "Warning"
                elif "Critical" in line or "Error" in line:
                    health_status = "Critical"

        memory_util = (memory_used / memory_total * 100) if memory_total > 0 else 0

        return NPUMetrics(
            device_id=self.device_id,
            name=name,
            chip_name=chip_name,
            aicore_utilization=aicore_util,
            memory_used_mb=memory_used,
            memory_total_mb=memory_total,
            memory_utilization=memory_util,
            temperature=temperature,
            power_draw_w=power_draw,
            health_status=health_status,
            timestamp=datetime.now(),
        )

    def _get_metrics_acl(self) -> Optional[NPUMetrics]:
        """使用 ACL 获取指标（基础信息）"""
        try:
            # 初始化 ACL
            ret = self.acl.init()
            if ret != 0:
                logger.error(f"ACL 初始化失败: {ret}")
                return None

            # 设置设备
            ret = self.acl.rt.set_device(self.device_id)
            if ret != 0:
                logger.error(f"设置设备失败: {ret}")
                return None

            # 获取内存信息
            try:
                free_mem, ret = self.acl.rt.get_mem_info(0)  # ACL_HBM_MEM
                if ret == 0:
                    total_mem = 16 * 1024 * 1024 * 1024  # 假设 16GB
                    used_mem = total_mem - free_mem
                    memory_used_mb = used_mem / 1024 / 1024
                    memory_total_mb = total_mem / 1024 / 1024
                else:
                    memory_used_mb = 0
                    memory_total_mb = 16384  # 16GB
            except Exception:
                memory_used_mb = 0
                memory_total_mb = 16384

            memory_util = (
                (memory_used_mb / memory_total_mb * 100) if memory_total_mb > 0 else 0
            )

            # 重置设备
            self.acl.rt.reset_device(self.device_id)
            self.acl.finalize()

            return NPUMetrics(
                device_id=self.device_id,
                name="Ascend NPU",
                chip_name="Ascend 310P",
                aicore_utilization=0.0,  # ACL 不直接提供
                memory_used_mb=memory_used_mb,
                memory_total_mb=memory_total_mb,
                memory_utilization=memory_util,
                temperature=0.0,  # ACL 不直接提供
                power_draw_w=0.0,  # ACL 不直接提供
                health_status="OK",
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"ACL 获取指标失败: {e}")
            return None

    def get_all_devices(self) -> List[int]:
        """获取所有可用的 NPU 设备"""
        try:
            result = subprocess.run(
                ["npu-smi", "info", "-l"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                # 解析设备列表
                devices = []
                for line in result.stdout.split("\n"):
                    match = re.search(r"NPU\s+(\d+)", line)
                    if match:
                        devices.append(int(match.group(1)))
                return devices

        except Exception:
            pass

        # 默认返回单个设备
        return [0]

    def print_metrics(self, metrics: NPUMetrics):
        """打印性能指标"""
        print("\n" + "=" * 80)
        print(f"昇腾 NPU 性能监控 - {metrics.chip_name} (NPU {metrics.device_id})")
        print("=" * 80)
        print(f"时间:           {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"AI Core 使用率: {metrics.aicore_utilization:>6.1f}%")
        print(
            f"内存使用:       {metrics.memory_used_mb:>7.0f} MB / "
            f"{metrics.memory_total_mb:>7.0f} MB ({metrics.memory_utilization:>5.1f}%)"
        )
        print(f"温度:           {metrics.temperature:>6.1f}°C")
        print(f"功耗:           {metrics.power_draw_w:>6.1f}W")
        print(f"健康状态:       {metrics.health_status}")
        print("=" * 80 + "\n")

    def monitor_loop(self, interval: float = 1.0):
        """监控循环"""
        logger.info(f"开始 NPU 性能监控，间隔 {interval} 秒")

        try:
            while True:
                metrics = self.get_npu_metrics()
                if metrics:
                    self.print_metrics(metrics)
                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("监控已停止")


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

        summary = [
            "\n" + "=" * 80,
            f"昇腾流性能统计 (运行时间: {elapsed:.1f}s)",
            "=" * 80,
        ]

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

    parser = argparse.ArgumentParser(description="昇腾 NPU 性能监控工具")
    parser.add_argument("--device", type=int, default=0, help="NPU 设备 ID (默认: 0)")
    parser.add_argument(
        "--interval", type=float, default=1.0, help="监控间隔（秒）(默认: 1.0)"
    )
    parser.add_argument("--list", action="store_true", help="列出所有 NPU 设备")

    args = parser.parse_args()

    # 创建监控器
    monitor = AscendPerformanceMonitor(device_id=args.device)

    if args.list:
        devices = monitor.get_all_devices()
        print(f"发现 {len(devices)} 个 NPU 设备: {devices}")
        return

    # 开始监控
    monitor.monitor_loop(interval=args.interval)


if __name__ == "__main__":
    main()
