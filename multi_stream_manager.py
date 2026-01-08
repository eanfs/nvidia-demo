"""
多路 RTSP 流管理器
支持同时处理多路视频流，使用批处理优化 GPU 性能
"""

import cv2
import numpy as np
import torch
import threading
import queue
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from face_detector import FaceDetector, InsightFaceDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """单路流配置"""

    stream_id: str
    rtsp_url: str
    priority: int = 1  # 优先级 1-10，越高越重要
    target_fps: int = 5  # 目标检测帧率
    enabled: bool = True


@dataclass
class StreamMetrics:
    """流性能指标"""

    stream_id: str
    frames_received: int = 0
    frames_processed: int = 0
    faces_detected: int = 0
    avg_latency_ms: float = 0.0
    current_fps: float = 0.0
    decode_fps: float = 0.0
    error_count: int = 0
    last_update: datetime = None


class FrameBuffer:
    """帧缓冲区"""

    def __init__(self, maxsize: int = 100):
        self.queue = queue.Queue(maxsize=maxsize)

    def put(self, item, timeout: float = 0.1):
        """添加帧到缓冲区"""
        try:
            self.queue.put(item, timeout=timeout)
            return True
        except queue.Full:
            # 如果队列满了，丢弃旧帧
            try:
                self.queue.get_nowait()
                self.queue.put(item, timeout=timeout)
                return True
            except:
                return False

    def get(self, timeout: float = 1.0):
        """从缓冲区获取帧"""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def qsize(self):
        """获取队列大小"""
        return self.queue.qsize()

    def clear(self):
        """清空队列"""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                break


class StreamDecoder:
    """视频流解码器（单独线程）"""

    def __init__(self, config: StreamConfig, frame_buffer: FrameBuffer):
        self.config = config
        self.frame_buffer = frame_buffer
        self.cap = None
        self.running = False
        self.thread = None
        self.metrics = StreamMetrics(stream_id=config.stream_id)
        self.last_frame_time = time.time()

    def start(self):
        """启动解码线程"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._decode_loop, daemon=True)
        self.thread.start()
        logger.info(f"流 {self.config.stream_id} 解码器已启动")

    def stop(self):
        """停止解码线程"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logger.info(f"流 {self.config.stream_id} 解码器已停止")

    def _open_stream(self) -> bool:
        """打开视频流"""
        try:
            self.cap = cv2.VideoCapture(self.config.rtsp_url, cv2.CAP_FFMPEG)

            if not self.cap.isOpened():
                logger.error(
                    f"无法打开流 {self.config.stream_id}: {self.config.rtsp_url}"
                )
                return False

            # 设置缓冲区
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            logger.info(f"流 {self.config.stream_id} 已连接")
            return True

        except Exception as e:
            logger.error(f"打开流 {self.config.stream_id} 失败: {e}")
            return False

    def _decode_loop(self):
        """解码循环"""
        reconnect_attempts = 0
        max_reconnect = 5

        while self.running:
            if not self.config.enabled:
                time.sleep(1)
                continue

            # 打开或重连流
            if self.cap is None or not self.cap.isOpened():
                if reconnect_attempts >= max_reconnect:
                    logger.error(
                        f"流 {self.config.stream_id} 重连失败次数过多，停止尝试"
                    )
                    break

                if self._open_stream():
                    reconnect_attempts = 0
                else:
                    reconnect_attempts += 1
                    time.sleep(2)
                    continue

            # 读取帧
            ret, frame = self.cap.read()

            if not ret:
                logger.warning(f"流 {self.config.stream_id} 读取失败，尝试重连...")
                self.cap.release()
                self.cap = None
                self.metrics.error_count += 1
                time.sleep(1)
                continue

            # 计算 FPS
            current_time = time.time()
            self.metrics.decode_fps = 1.0 / (current_time - self.last_frame_time + 1e-6)
            self.last_frame_time = current_time

            # 帧采样（根据目标 FPS）
            self.metrics.frames_received += 1

            # 计算采样间隔
            stream_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if stream_fps > 0:
                sample_interval = max(1, int(stream_fps / self.config.target_fps))
            else:
                sample_interval = 6  # 默认 30fps -> 5fps

            if self.metrics.frames_received % sample_interval != 0:
                continue

            # 添加到缓冲区
            frame_data = {
                "stream_id": self.config.stream_id,
                "frame": frame,
                "timestamp": current_time,
                "frame_number": self.metrics.frames_received,
                "priority": self.config.priority,
            }

            if not self.frame_buffer.put(frame_data):
                logger.warning(f"流 {self.config.stream_id} 缓冲区已满，丢帧")


class MultiStreamManager:
    """多路流管理器"""

    def __init__(
        self,
        detector_type: str = "mtcnn",
        device: str = "cuda",
        batch_size: int = 8,
        max_buffer_size: int = 100,
    ):
        """
        初始化多路流管理器

        Args:
            detector_type: 检测器类型
            device: 推理设备
            batch_size: 批处理大小
            max_buffer_size: 最大缓冲区大小
        """
        self.detector_type = detector_type
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        # 初始化检测器
        self._init_detector()

        # 流管理
        self.streams: Dict[str, StreamDecoder] = {}
        self.frame_buffer = FrameBuffer(maxsize=max_buffer_size)
        self.running = False
        self.inference_thread = None

        # 性能统计
        self.global_metrics = {
            "total_frames_processed": 0,
            "total_faces_detected": 0,
            "avg_batch_time_ms": 0.0,
            "gpu_utilization": 0.0,
        }

        logger.info(f"多路流管理器已初始化，设备: {self.device}, 批大小: {batch_size}")

    def _init_detector(self):
        """初始化人脸检测器"""
        if self.detector_type == "insightface":
            self.detector = InsightFaceDetector(device=self.device)
        else:
            use_mtcnn = self.detector_type == "mtcnn"
            self.detector = FaceDetector(device=self.device, use_mtcnn=use_mtcnn)

    def add_stream(self, config: StreamConfig):
        """添加视频流"""
        if config.stream_id in self.streams:
            logger.warning(f"流 {config.stream_id} 已存在")
            return

        decoder = StreamDecoder(config, self.frame_buffer)
        self.streams[config.stream_id] = decoder

        if self.running:
            decoder.start()

        logger.info(f"已添加流 {config.stream_id}")

    def remove_stream(self, stream_id: str):
        """移除视频流"""
        if stream_id not in self.streams:
            logger.warning(f"流 {stream_id} 不存在")
            return

        decoder = self.streams[stream_id]
        decoder.stop()
        del self.streams[stream_id]

        logger.info(f"已移除流 {stream_id}")

    def start(self):
        """启动所有流和推理线程"""
        if self.running:
            return

        self.running = True

        # 启动所有解码器
        for decoder in self.streams.values():
            decoder.start()

        # 启动推理线程
        self.inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self.inference_thread.start()

        logger.info(f"多路流管理器已启动，共 {len(self.streams)} 路流")

    def stop(self):
        """停止所有流和推理"""
        self.running = False

        # 停止所有解码器
        for decoder in self.streams.values():
            decoder.stop()

        # 等待推理线程结束
        if self.inference_thread:
            self.inference_thread.join(timeout=5.0)

        logger.info("多路流管理器已停止")

    def _inference_loop(self):
        """推理循环（批处理）"""
        batch_frames = []
        batch_metadata = []

        while self.running:
            # 收集批处理帧
            while len(batch_frames) < self.batch_size:
                frame_data = self.frame_buffer.get(timeout=0.5)

                if frame_data is None:
                    break

                batch_frames.append(frame_data["frame"])
                batch_metadata.append(frame_data)

            if len(batch_frames) == 0:
                continue

            # 批量推理
            batch_start = time.time()

            try:
                # 执行检测
                if self.detector_type == "insightface":
                    results = [
                        self.detector.detect_faces(frame) for frame in batch_frames
                    ]
                else:
                    results = [
                        self.detector.detect_faces(frame) for frame in batch_frames
                    ]

                batch_time = (time.time() - batch_start) * 1000

                # 更新统计
                for i, (metadata, result) in enumerate(zip(batch_metadata, results)):
                    stream_id = metadata["stream_id"]

                    if stream_id in self.streams:
                        decoder = self.streams[stream_id]
                        decoder.metrics.frames_processed += 1

                        if self.detector_type == "insightface":
                            num_faces = len(result)
                        else:
                            boxes, _ = result
                            num_faces = len(boxes)

                        decoder.metrics.faces_detected += num_faces

                        # 更新延迟
                        latency = (time.time() - metadata["timestamp"]) * 1000
                        decoder.metrics.avg_latency_ms = (
                            decoder.metrics.avg_latency_ms * 0.9 + latency * 0.1
                        )
                        decoder.metrics.last_update = datetime.now()

                # 更新全局统计
                self.global_metrics["total_frames_processed"] += len(batch_frames)
                self.global_metrics["avg_batch_time_ms"] = (
                    self.global_metrics["avg_batch_time_ms"] * 0.9 + batch_time * 0.1
                )

            except Exception as e:
                logger.error(f"批处理推理失败: {e}")

            finally:
                # 清空批次
                batch_frames.clear()
                batch_metadata.clear()

    def get_metrics(self) -> Dict[str, StreamMetrics]:
        """获取所有流的性能指标"""
        return {sid: decoder.metrics for sid, decoder in self.streams.items()}

    def get_summary(self) -> str:
        """获取性能摘要"""
        metrics = self.get_metrics()

        summary = [
            "=" * 80,
            f"多路流性能摘要 (共 {len(self.streams)} 路)",
            "=" * 80,
        ]

        total_fps = 0
        total_faces = 0

        for stream_id, metric in metrics.items():
            if metric.last_update:
                time_diff = (datetime.now() - metric.last_update).total_seconds()
                current_fps = metric.frames_processed / max(time_diff, 1)
            else:
                current_fps = 0

            total_fps += current_fps
            total_faces += metric.faces_detected

            summary.append(
                f"[{stream_id}] "
                f"处理: {metric.frames_processed} 帧 | "
                f"FPS: {current_fps:.1f} | "
                f"人脸: {metric.faces_detected} | "
                f"延迟: {metric.avg_latency_ms:.1f}ms | "
                f"错误: {metric.error_count}"
            )

        summary.extend(
            [
                "-" * 80,
                f"总处理FPS: {total_fps:.1f}",
                f"总检测人脸: {total_faces}",
                f"平均批处理时间: {self.global_metrics['avg_batch_time_ms']:.1f}ms",
                f"缓冲区大小: {self.frame_buffer.qsize()}",
                "=" * 80,
            ]
        )

        return "\n".join(summary)
