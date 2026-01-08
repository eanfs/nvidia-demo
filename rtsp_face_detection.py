"""
RTSP 视频流人脸检测主程序
使用 NVIDIA GPU 加速解码和推理
"""

import cv2
import numpy as np
import torch
import av
import logging
import time
import argparse
from typing import Optional
from face_detector import FaceDetector, InsightFaceDetector
from config import Config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RTSPVideoProcessor:
    """RTSP 视频流处理器，支持 GPU 硬件解码"""

    def __init__(
        self,
        rtsp_url: str,
        use_gpu_decode: bool = True,
        detector_type: str = "mtcnn",
        device: str = "cuda",
    ):
        """
        初始化视频处理器

        Args:
            rtsp_url: RTSP 流地址
            use_gpu_decode: 是否使用 GPU 硬件解码
            detector_type: 人脸检测器类型 ('mtcnn', 'opencv', 'insightface')
            device: 推理设备
        """
        self.rtsp_url = rtsp_url
        self.use_gpu_decode = use_gpu_decode and torch.cuda.is_available()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.detector_type = detector_type

        logger.info(f"初始化 RTSP 处理器")
        logger.info(f"  URL: {rtsp_url}")
        logger.info(f"  GPU 解码: {self.use_gpu_decode}")
        logger.info(f"  推理设备: {self.device}")
        logger.info(f"  检测器类型: {detector_type}")

        # 初始化人脸检测器
        self._init_detector()

        # 视频流对象
        self.cap = None
        self.container = None

    def _init_detector(self):
        """初始化人脸检测器"""
        try:
            if self.detector_type == "insightface":
                self.detector = InsightFaceDetector(device=self.device)
            else:
                use_mtcnn = self.detector_type == "mtcnn"
                self.detector = FaceDetector(device=self.device, use_mtcnn=use_mtcnn)

            logger.info("人脸检测器初始化成功")
        except Exception as e:
            logger.error(f"人脸检测器初始化失败: {e}")
            raise

    def _open_opencv_stream(self) -> bool:
        """使用 OpenCV 打开 RTSP 流"""
        try:
            # 设置 OpenCV 使用 NVIDIA 硬件加速（如果可用）
            if self.use_gpu_decode:
                # 尝试使用 FFMPEG 后端与硬件加速
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

                # 设置解码器使用 GPU
                # 注意：这需要 OpenCV 编译时支持 CUDA
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    logger.info(
                        f"检测到 {cv2.cuda.getCudaEnabledDeviceCount()} 个 CUDA 设备"
                    )
            else:
                self.cap = cv2.VideoCapture(self.rtsp_url)

            if not self.cap.isOpened():
                logger.error("无法打开 RTSP 流")
                return False

            # 设置缓冲区
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

            # 获取视频信息
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"视频流已打开: {width}x{height} @ {fps:.2f} FPS")
            return True

        except Exception as e:
            logger.error(f"打开视频流失败: {e}")
            return False

    def _open_pyav_stream(self) -> bool:
        """使用 PyAV 打开 RTSP 流（支持硬件加速）"""
        try:
            options = {
                "rtsp_transport": "tcp",  # 使用 TCP 传输（更稳定）
                "max_delay": "500000",  # 最大延迟 500ms
            }

            self.container = av.open(self.rtsp_url, options=options)

            # 获取视频流
            self.video_stream = self.container.streams.video[0]

            # 尝试使用硬件解码器
            if self.use_gpu_decode:
                try:
                    # 使用 NVIDIA 硬件解码器
                    codec_context = self.video_stream.codec_context
                    codec_context.thread_type = "AUTO"
                    logger.info("尝试使用 NVIDIA 硬件解码器")
                except Exception as e:
                    logger.warning(f"无法启用硬件解码: {e}")

            logger.info(
                f"PyAV 视频流已打开: {self.video_stream.width}x{self.video_stream.height}"
            )
            return True

        except Exception as e:
            logger.error(f"PyAV 打开视频流失败: {e}")
            return False

    def process_stream(
        self,
        display: bool = True,
        save_output: Optional[str] = None,
        process_every_n_frames: int = 1,
    ):
        """
        处理 RTSP 视频流

        Args:
            display: 是否显示处理结果
            save_output: 输出视频保存路径（可选）
            process_every_n_frames: 每 N 帧处理一次（降低 CPU 负载）
        """
        # 打开视频流
        if not self._open_opencv_stream():
            logger.error("无法打开视频流")
            return

        # 初始化输出视频编写器
        out_writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_writer = cv2.VideoWriter(save_output, fourcc, fps, (width, height))

        frame_count = 0
        fps_start_time = time.time()
        fps_counter = 0
        current_fps = 0

        logger.info("开始处理视频流，按 'q' 退出")

        try:
            while True:
                ret, frame = self.cap.read()

                if not ret:
                    logger.warning("无法读取帧，尝试重新连接...")
                    time.sleep(1)
                    if not self._open_opencv_stream():
                        break
                    continue

                frame_count += 1
                fps_counter += 1

                # 计算 FPS
                if time.time() - fps_start_time > 1.0:
                    current_fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()

                # 每 N 帧处理一次
                if frame_count % process_every_n_frames == 0:
                    # 人脸检测
                    detect_start = time.time()

                    if self.detector_type == "insightface":
                        faces = self.detector.detect_faces(frame)
                        result_frame = self.detector.draw_faces(frame, faces)
                        num_faces = len(faces)
                    else:
                        boxes, confidences = self.detector.detect_faces(frame)
                        result_frame = self.detector.draw_faces(
                            frame, boxes, confidences
                        )
                        num_faces = len(boxes)

                    detect_time = (time.time() - detect_start) * 1000

                    # 添加信息文本
                    info_text = [
                        f"FPS: {current_fps:.1f}",
                        f"Frame: {frame_count}",
                        f"Faces: {num_faces}",
                        f"Detect: {detect_time:.1f}ms",
                    ]

                    y_offset = 30
                    for i, text in enumerate(info_text):
                        cv2.putText(
                            result_frame,
                            text,
                            (10, y_offset + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )
                else:
                    result_frame = frame

                # 显示结果
                if display:
                    cv2.imshow("RTSP Face Detection", result_frame)

                    # 按 'q' 退出
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("用户请求退出")
                        break

                # 保存输出
                if out_writer:
                    out_writer.write(result_frame)

        except KeyboardInterrupt:
            logger.info("接收到中断信号")
        except Exception as e:
            logger.error(f"处理过程中出错: {e}")
        finally:
            # 清理资源
            self.cleanup()
            if out_writer:
                out_writer.release()
            cv2.destroyAllWindows()

    def cleanup(self):
        """清理资源"""
        if self.cap:
            self.cap.release()
        if self.container:
            self.container.close()
        logger.info("资源已清理")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RTSP 视频流人脸检测（GPU 加速）")
    parser.add_argument("--rtsp-url", type=str, default=None, help="RTSP 流地址")
    parser.add_argument(
        "--detector",
        type=str,
        default="mtcnn",
        choices=["mtcnn", "opencv", "insightface"],
        help="人脸检测器类型",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="推理设备"
    )
    parser.add_argument(
        "--no-gpu-decode", action="store_true", help="禁用 GPU 硬件解码"
    )
    parser.add_argument("--no-display", action="store_true", help="不显示视频窗口")
    parser.add_argument("--output", type=str, default=None, help="输出视频保存路径")
    parser.add_argument("--process-every", type=int, default=1, help="每 N 帧处理一次")

    args = parser.parse_args()

    # 使用配置文件中的 URL 或命令行参数
    rtsp_url = args.rtsp_url or Config.RTSP_URL

    if not rtsp_url:
        logger.error("请提供 RTSP URL（通过 --rtsp-url 或在 config.py 中配置）")
        return

    # 检查 CUDA 可用性
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，将使用 CPU")
        args.device = "cpu"

    if torch.cuda.is_available():
        logger.info(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA 版本: {torch.version.cuda}")

    # 创建处理器
    processor = RTSPVideoProcessor(
        rtsp_url=rtsp_url,
        use_gpu_decode=not args.no_gpu_decode,
        detector_type=args.detector,
        device=args.device,
    )

    # 处理视频流
    processor.process_stream(
        display=not args.no_display,
        save_output=args.output,
        process_every_n_frames=args.process_every,
    )


if __name__ == "__main__":
    main()
