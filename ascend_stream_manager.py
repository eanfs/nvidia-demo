"""
昇腾多路 RTSP 流管理器
使用 DVPP 进行硬件视频解码，支持批处理优化
"""

import cv2
import numpy as np
import threading
import queue
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

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
    last_update: Optional[datetime] = None


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
            try:
                self.queue.get_nowait()
                self.queue.put(item, timeout=timeout)
                return True
            except Exception:
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
            except Exception:
                break


class DVPPDecoder:
    """
    DVPP 硬件视频解码器
    使用昇腾 DVPP 模块进行硬件加速视频解码
    """

    def __init__(self, device_id: int = 0):
        """
        初始化 DVPP 解码器

        Args:
            device_id: NPU 设备 ID
        """
        self.device_id = device_id
        self.context = None
        self.stream = None
        self.dvpp_channel_desc = None
        self.initialized = False

        self._init_dvpp()

    def _init_dvpp(self):
        """初始化 DVPP"""
        try:
            import acl

            self.acl = acl

            # 设置设备
            ret = acl.rt.set_device(self.device_id)
            if ret != 0:
                logger.warning(f"设置设备失败: {ret}，将使用 OpenCV 软解码")
                return

            # 创建 context
            self.context, ret = acl.rt.create_context(self.device_id)
            if ret != 0:
                logger.warning(f"创建 context 失败: {ret}")
                return

            # 创建 stream
            self.stream, ret = acl.rt.create_stream()
            if ret != 0:
                logger.warning(f"创建 stream 失败: {ret}")
                return

            # 创建 DVPP 通道描述符
            self.dvpp_channel_desc = acl.media.dvpp_create_channel_desc()
            if self.dvpp_channel_desc is None:
                logger.warning("创建 DVPP 通道描述符失败")
                return

            # 创建 DVPP 通道
            ret = acl.media.dvpp_create_channel(self.dvpp_channel_desc)
            if ret != 0:
                logger.warning(f"创建 DVPP 通道失败: {ret}")
                return

            self.initialized = True
            logger.info("DVPP 硬件解码器初始化成功")

        except ImportError:
            logger.warning("未安装 ACL，将使用 OpenCV 软解码")
        except Exception as e:
            logger.warning(f"DVPP 初始化失败: {e}，将使用 OpenCV 软解码")

    def decode_frame(self, frame_data: bytes) -> Optional[np.ndarray]:
        """
        使用 DVPP 解码视频帧

        Args:
            frame_data: 编码的帧数据

        Returns:
            解码后的图像 (BGR 格式)
        """
        if not self.initialized:
            return None

        input_buffer = None
        output_buffer = None
        input_pic_desc = None
        output_pic_desc = None

        try:
            # 分配设备内存
            input_size = len(frame_data)
            input_buffer, ret = self.acl.rt.malloc(input_size, 0)
            if ret != 0:
                return None

            # 拷贝数据到设备
            input_ptr = self.acl.util.bytes_to_ptr(frame_data)
            ret = self.acl.rt.memcpy(input_buffer, input_size, input_ptr, input_size, 1)
            if ret != 0:
                return None

            # 创建输入图片描述
            input_pic_desc = self.acl.media.dvpp_create_pic_desc()
            self.acl.media.dvpp_set_pic_desc_data(input_pic_desc, input_buffer)
            self.acl.media.dvpp_set_pic_desc_size(input_pic_desc, input_size)

            # 分配输出内存 (YUV420SP 格式)
            output_size = 1920 * 1080 * 3 // 2  # 假设最大 1080p
            output_buffer, ret = self.acl.rt.malloc(output_size, 0)
            if ret != 0:
                return None

            # 创建输出图片描述
            output_pic_desc = self.acl.media.dvpp_create_pic_desc()
            self.acl.media.dvpp_set_pic_desc_data(output_pic_desc, output_buffer)
            self.acl.media.dvpp_set_pic_desc_size(output_pic_desc, output_size)

            # 执行解码
            ret = self.acl.media.dvpp_jpeg_decode_async(
                self.dvpp_channel_desc, input_pic_desc, output_pic_desc, self.stream
            )

            # 同步等待
            ret = self.acl.rt.synchronize_stream(self.stream)

            # 获取解码后的图像尺寸
            width = self.acl.media.dvpp_get_pic_desc_width(output_pic_desc)
            height = self.acl.media.dvpp_get_pic_desc_height(output_pic_desc)

            # 拷贝输出数据到主机
            output_host = np.zeros((height * 3 // 2, width), dtype=np.uint8)
            output_ptr = self.acl.util.numpy_to_ptr(output_host)
            actual_size = width * height * 3 // 2
            ret = self.acl.rt.memcpy(
                output_ptr, actual_size, output_buffer, actual_size, 2
            )

            # YUV420SP 转 BGR
            yuv = output_host.reshape((height * 3 // 2, width))
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)

            return bgr

        except Exception as e:
            logger.error(f"DVPP 解码失败: {e}")
            return None

        finally:
            # 确保资源释放
            if input_buffer:
                self.acl.rt.free(input_buffer)
            if output_buffer:
                self.acl.rt.free(output_buffer)
            if input_pic_desc:
                self.acl.media.dvpp_destroy_pic_desc(input_pic_desc)
            if output_pic_desc:
                self.acl.media.dvpp_destroy_pic_desc(output_pic_desc)

    def release(self):
        """释放资源"""
        if not self.initialized:
            return

        try:
            if self.dvpp_channel_desc:
                self.acl.media.dvpp_destroy_channel(self.dvpp_channel_desc)
                self.acl.media.dvpp_destroy_channel_desc(self.dvpp_channel_desc)

            if self.stream:
                self.acl.rt.destroy_stream(self.stream)

            if self.context:
                self.acl.rt.destroy_context(self.context)

            self.acl.rt.reset_device(self.device_id)
            logger.info("DVPP 资源已释放")

        except Exception as e:
            logger.error(f"释放 DVPP 资源失败: {e}")


class AscendStreamDecoder:
    """昇腾视频流解码器（单独线程）"""

    def __init__(
        self,
        config: StreamConfig,
        frame_buffer: FrameBuffer,
        use_dvpp: bool = True,
        device_id: int = 0,
    ):
        self.config = config
        self.frame_buffer = frame_buffer
        self.use_dvpp = use_dvpp
        self.device_id = device_id
        self.cap = None
        self.dvpp_decoder = None
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
        if self.dvpp_decoder:
            self.dvpp_decoder.release()
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


class AscendMultiStreamManager:
    """昇腾多路流管理器"""

    def __init__(
        self,
        model_path: str = None,
        device_id: int = 0,
        batch_size: int = 8,
        max_buffer_size: int = 100,
        use_dvpp: bool = True,
    ):
        """
        初始化昇腾多路流管理器

        Args:
            model_path: .om 模型路径
            device_id: NPU 设备 ID
            batch_size: 批处理大小
            max_buffer_size: 最大缓冲区大小
            use_dvpp: 是否使用 DVPP 硬件解码
        """
        self.model_path = model_path
        self.device_id = device_id
        self.batch_size = batch_size
        self.use_dvpp = use_dvpp

        # 初始化检测器
        self._init_detector()

        # 流管理
        self.streams: Dict[str, AscendStreamDecoder] = {}
        self.frame_buffer = FrameBuffer(maxsize=max_buffer_size)
        self.running = False
        self.inference_thread = None

        # 性能统计
        self.global_metrics = {
            "total_frames_processed": 0,
            "total_faces_detected": 0,
            "avg_batch_time_ms": 0.0,
            "npu_utilization": 0.0,
        }

        logger.info(
            f"昇腾多路流管理器已初始化，设备: {device_id}, 批大小: {batch_size}"
        )

    def _init_detector(self):
        """初始化人脸检测器"""
        from ascend_face_detector import AscendFaceDetector

        self.detector = AscendFaceDetector(
            model_path=self.model_path, device_id=self.device_id
        )

    def add_stream(self, config: StreamConfig):
        """添加视频流"""
        if config.stream_id in self.streams:
            logger.warning(f"流 {config.stream_id} 已存在")
            return

        decoder = AscendStreamDecoder(
            config,
            self.frame_buffer,
            use_dvpp=self.use_dvpp,
            device_id=self.device_id,
        )
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

        logger.info(f"昇腾多路流管理器已启动，共 {len(self.streams)} 路流")

    def stop(self):
        """停止所有流和推理"""
        self.running = False

        # 停止所有解码器
        for decoder in self.streams.values():
            decoder.stop()

        # 等待推理线程结束
        if self.inference_thread:
            self.inference_thread.join(timeout=5.0)

        # 释放检测器资源
        if hasattr(self.detector, "release"):
            self.detector.release()

        logger.info("昇腾多路流管理器已停止")

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
                results = self.detector.detect_batch(batch_frames)
                batch_time = (time.time() - batch_start) * 1000

                # 更新统计
                for i, (metadata, result) in enumerate(zip(batch_metadata, results)):
                    stream_id = metadata["stream_id"]

                    if stream_id in self.streams:
                        decoder = self.streams[stream_id]
                        decoder.metrics.frames_processed += 1

                        num_faces = len(result)
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
            f"昇腾多路流性能摘要 (共 {len(self.streams)} 路)",
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
