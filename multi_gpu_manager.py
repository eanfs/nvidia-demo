"""
多 GPU 管理器
支持在多个 GPU 上并行处理 RTSP 流
"""

import os
import multiprocessing as mp
import logging
import time
from typing import List, Dict, Any
from dataclasses import dataclass
from multi_stream_manager import MultiStreamManager, StreamConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class GPUWorkerConfig:
    gpu_id: int
    streams: List[StreamConfig]
    detector_type: str = "mtcnn"
    batch_size: int = 8
    max_buffer_size: int = 100


def gpu_worker_process(config: GPUWorkerConfig, stop_event):
    """
    GPU 工作进程
    在指定 GPU 上运行流处理
    """
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

        logger.info(f"GPU {config.gpu_id} 进程启动，处理 {len(config.streams)} 路流")

        manager = MultiStreamManager(
            detector_type=config.detector_type,
            device="cuda",
            batch_size=config.batch_size,
            max_buffer_size=config.max_buffer_size,
        )

        for stream in config.streams:
            manager.add_stream(stream)

        manager.start()

        last_report = time.time()
        while not stop_event.is_set():
            time.sleep(1)

            if time.time() - last_report >= 10:
                summary = manager.get_summary()
                logger.info(f"\n[GPU {config.gpu_id}]\n{summary}")
                last_report = time.time()

        manager.stop()
        logger.info(f"GPU {config.gpu_id} 进程已停止")

    except Exception as e:
        logger.error(f"GPU {config.gpu_id} 进程出错: {e}", exc_info=True)


class MultiGPUManager:
    """多 GPU 管理器"""

    def __init__(
        self,
        num_gpus: int = 8,
        detector_type: str = "mtcnn",
        batch_size: int = 8,
        max_buffer_size: int = 100,
        load_balance: str = "round_robin",
    ):
        """
        初始化多 GPU 管理器

        Args:
            num_gpus: GPU 数量
            detector_type: 检测器类型
            batch_size: 每个 GPU 的批处理大小
            max_buffer_size: 缓冲区大小
            load_balance: 负载均衡策略 (round_robin, priority, auto)
        """
        self.num_gpus = num_gpus
        self.detector_type = detector_type
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size
        self.load_balance = load_balance

        self.workers = []
        self.worker_configs = []
        self.stop_events = []

        logger.info(f"多 GPU 管理器已初始化，GPU 数量: {num_gpus}")

    def distribute_streams(self, streams: List[StreamConfig]) -> List:
        """
        将流分配到各个 GPU

        Args:
            streams: 所有流配置

        Returns:
            每个 GPU 的流列表
        """
        if self.load_balance == "round_robin":
            return self._round_robin_distribute(streams)
        elif self.load_balance == "priority":
            return self._priority_distribute(streams)
        elif self.load_balance == "auto":
            return self._auto_distribute(streams)
        else:
            return self._round_robin_distribute(streams)

    def _round_robin_distribute(
        self, streams: List[StreamConfig]
    ) -> List[List[StreamConfig]]:
        """轮询分配"""
        gpu_streams = [[] for _ in range(self.num_gpus)]

        for i, stream in enumerate(streams):
            gpu_id = i % self.num_gpus
            gpu_streams[gpu_id].append(stream)

        return gpu_streams

    def _priority_distribute(
        self, streams: List[StreamConfig]
    ) -> List[List[StreamConfig]]:
        """按优先级分配（高优先级流优先分配到空闲 GPU）"""
        sorted_streams = sorted(streams, key=lambda x: x.priority, reverse=True)
        gpu_streams = [[] for _ in range(self.num_gpus)]
        gpu_loads = [0] * self.num_gpus

        for stream in sorted_streams:
            min_load_gpu = gpu_loads.index(min(gpu_loads))
            gpu_streams[min_load_gpu].append(stream)
            gpu_loads[min_load_gpu] += stream.priority

        return gpu_streams

    def _auto_distribute(self, streams: List[StreamConfig]) -> List[List[StreamConfig]]:
        """自动分配（考虑流的目标 FPS 和优先级）"""
        sorted_streams = sorted(
            streams, key=lambda x: (x.priority, x.target_fps), reverse=True
        )
        gpu_streams = [[] for _ in range(self.num_gpus)]
        gpu_loads = [0.0] * self.num_gpus

        for stream in sorted_streams:
            load_score = stream.target_fps * stream.priority
            min_load_gpu = gpu_loads.index(min(gpu_loads))
            gpu_streams[min_load_gpu].append(stream)
            gpu_loads[min_load_gpu] += load_score

        return gpu_streams

    def start(self, streams: List[StreamConfig]):
        """启动所有 GPU 工作进程"""
        gpu_streams = self.distribute_streams(streams)

        logger.info(f"开始在 {self.num_gpus} 个 GPU 上分配 {len(streams)} 路流")
        for gpu_id, gpu_stream_list in enumerate(gpu_streams):
            logger.info(f"  GPU {gpu_id}: {len(gpu_stream_list)} 路流")

        for gpu_id, gpu_stream_list in enumerate(gpu_streams):
            if not gpu_stream_list:
                logger.info(f"GPU {gpu_id} 没有分配流，跳过")
                continue

            config = GPUWorkerConfig(
                gpu_id=gpu_id,
                streams=gpu_stream_list,
                detector_type=self.detector_type,
                batch_size=self.batch_size,
                max_buffer_size=self.max_buffer_size,
            )

            stop_event = mp.Event()
            process = mp.Process(
                target=gpu_worker_process,
                args=(config, stop_event),
                name=f"GPU-{gpu_id}",
            )

            self.worker_configs.append(config)
            self.stop_events.append(stop_event)
            self.workers.append(process)

            process.start()
            logger.info(f"GPU {gpu_id} 工作进程已启动 (PID: {process.pid})")

        logger.info(f"所有 GPU 工作进程已启动，共 {len(self.workers)} 个进程")

    def stop(self):
        """停止所有 GPU 工作进程"""
        logger.info("正在停止所有 GPU 工作进程...")

        for event in self.stop_events:
            event.set()

        for worker in self.workers:
            worker.join(timeout=10)
            if worker.is_alive():
                logger.warning(f"进程 {worker.name} 未正常退出，强制终止")
                worker.terminate()
                worker.join(timeout=2)

        self.workers.clear()
        self.worker_configs.clear()
        self.stop_events.clear()

        logger.info("所有 GPU 工作进程已停止")

    def get_status(self) -> Dict:
        """获取所有 GPU 的状态"""
        status = {
            "num_gpus": self.num_gpus,
            "active_workers": sum(1 for w in self.workers if w.is_alive()),
            "total_streams": sum(len(cfg.streams) for cfg in self.worker_configs),
            "workers": [],
        }

        for i, (worker, config) in enumerate(zip(self.workers, self.worker_configs)):
            status["workers"].append(
                {
                    "gpu_id": config.gpu_id,
                    "pid": worker.pid,
                    "alive": worker.is_alive(),
                    "num_streams": len(config.streams),
                    "stream_ids": [s.stream_id for s in config.streams],
                }
            )

        return status
