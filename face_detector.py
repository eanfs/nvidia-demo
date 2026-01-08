"""
人脸检测器模块 - 使用 MTCNN 和 InsightFace
支持 GPU 加速
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional
from facenet_pytorch import MTCNN
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetector:
    """人脸检测器类，支持 GPU 加速"""

    def __init__(self, device: str = "cuda", use_mtcnn: bool = True):
        """
        初始化人脸检测器

        Args:
            device: 使用的设备 ('cuda' 或 'cpu')
            use_mtcnn: 是否使用 MTCNN (否则使用 OpenCV Haar Cascade)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_mtcnn = use_mtcnn

        logger.info(f"初始化人脸检测器，使用设备: {self.device}")

        if self.use_mtcnn:
            try:
                # 初始化 MTCNN
                self.detector = MTCNN(
                    image_size=160,
                    margin=0,
                    min_face_size=20,
                    thresholds=[0.6, 0.7, 0.7],
                    factor=0.709,
                    post_process=True,
                    device=self.device,
                    keep_all=True,  # 检测所有人脸
                )
                logger.info("MTCNN 初始化成功")
            except Exception as e:
                logger.warning(f"MTCNN 初始化失败: {e}, 降级使用 OpenCV")
                self.use_mtcnn = False
                self._init_opencv_detector()
        else:
            self._init_opencv_detector()

    def _init_opencv_detector(self):
        """初始化 OpenCV 人脸检测器"""
        try:
            # 使用 Haar Cascade
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(cascade_path)
            logger.info("OpenCV Haar Cascade 初始化成功")
        except Exception as e:
            logger.error(f"OpenCV 检测器初始化失败: {e}")
            raise

    def detect_faces(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """
        检测图像中的人脸

        Args:
            frame: BGR 格式的图像 (OpenCV 格式)

        Returns:
            boxes: 人脸边界框列表 [(x1, y1, x2, y2), ...]
            confidences: 置信度列表
        """
        if self.use_mtcnn:
            return self._detect_mtcnn(frame)
        else:
            return self._detect_opencv(frame)

    def _detect_mtcnn(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """使用 MTCNN 检测人脸"""
        try:
            # 转换 BGR 到 RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 检测人脸
            boxes, probs = self.detector.detect(rgb_frame)

            if boxes is None:
                return [], []

            # 转换为整数坐标
            boxes = boxes.astype(int)
            face_boxes = [(box[0], box[1], box[2], box[3]) for box in boxes]
            confidences = probs.tolist() if probs is not None else [0.0] * len(boxes)

            return face_boxes, confidences

        except Exception as e:
            logger.error(f"MTCNN 检测失败: {e}")
            return [], []

    def _detect_opencv(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """使用 OpenCV Haar Cascade 检测人脸"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测人脸
            faces = self.detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            # 转换格式 (x, y, w, h) -> (x1, y1, x2, y2)
            face_boxes = [(x, y, x + w, y + h) for (x, y, w, h) in faces]
            confidences = [1.0] * len(faces)  # OpenCV 不提供置信度

            return face_boxes, confidences

        except Exception as e:
            logger.error(f"OpenCV 检测失败: {e}")
            return [], []

    def draw_faces(
        self,
        frame: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        confidences: List[float],
        show_conf: bool = True,
    ) -> np.ndarray:
        """
        在图像上绘制人脸框

        Args:
            frame: 原始图像
            boxes: 人脸边界框列表
            confidences: 置信度列表
            show_conf: 是否显示置信度

        Returns:
            绘制了人脸框的图像
        """
        result = frame.copy()

        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = box

            # 绘制矩形框
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # 添加标签
            if show_conf:
                label = f"Face {i + 1}: {conf:.2f}"
            else:
                label = f"Face {i + 1}"

            # 绘制背景
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                result, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1
            )

            # 绘制文字
            cv2.putText(
                result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
            )

        return result


class InsightFaceDetector:
    """使用 InsightFace 进行人脸检测和识别"""

    def __init__(self, model_name: str = "buffalo_l", device: str = "cuda"):
        """
        初始化 InsightFace 检测器

        Args:
            model_name: 模型名称
            device: 设备类型
        """
        try:
            from insightface.app import FaceAnalysis

            self.device = device if torch.cuda.is_available() else "cpu"
            ctx_id = 0 if self.device == "cuda" else -1

            self.app = FaceAnalysis(
                name=model_name,
                providers=[
                    "CUDAExecutionProvider"
                    if self.device == "cuda"
                    else "CPUExecutionProvider"
                ],
            )
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

            logger.info(f"InsightFace 初始化成功，使用设备: {self.device}")

        except Exception as e:
            logger.error(f"InsightFace 初始化失败: {e}")
            raise

    def detect_faces(self, frame: np.ndarray):
        """检测人脸并返回特征"""
        try:
            faces = self.app.get(frame)
            return faces
        except Exception as e:
            logger.error(f"InsightFace 检测失败: {e}")
            return []

    def draw_faces(self, frame: np.ndarray, faces) -> np.ndarray:
        """绘制检测到的人脸"""
        result = frame.copy()

        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # 绘制矩形
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 添加年龄和性别信息（如果可用）
            if hasattr(face, "age") and hasattr(face, "gender"):
                gender = "M" if face.gender == 1 else "F"
                label = f"Age: {int(face.age)}, {gender}"
                cv2.putText(
                    result,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        return result
