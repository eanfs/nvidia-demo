"""
昇腾人脸检测器模块 - 使用 ACL (Ascend Computing Language)
支持 Atlas 300V NPU 加速推理
"""

import cv2
import numpy as np
import logging
import threading
from typing import List, Tuple, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ACLFaceDetector:
    """
    基于 ACL 的人脸检测器
    使用 .om 离线模型在昇腾 NPU 上进行推理
    """

    def __init__(
        self,
        model_path: str,
        device_id: int = 0,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
    ):
        """
        初始化 ACL 人脸检测器

        Args:
            model_path: .om 离线模型路径
            device_id: NPU 设备 ID
            input_size: 输入尺寸 (width, height)
            conf_threshold: 置信度阈值
            nms_threshold: NMS 阈值
        """
        self.model_path = model_path
        self.device_id = device_id
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        # ACL 资源
        self.context = None
        self.stream = None
        self.model_id = None
        self.model_desc = None
        self.input_dataset = None
        self.output_dataset = None

        # 线程锁 (保护 ACL 推理操作)
        self._inference_lock = threading.Lock()

        # 初始化 ACL
        self._init_acl()

        logger.info(f"ACL 人脸检测器已初始化，设备: {device_id}, 模型: {model_path}")

    def _init_acl(self):
        """初始化 ACL 运行时环境"""
        acl_initialized = False
        device_set = False

        try:
            import acl

            self.acl = acl

            # 初始化 ACL
            ret = acl.init()
            if ret != 0:
                raise RuntimeError(f"ACL 初始化失败，错误码: {ret}")
            acl_initialized = True

            # 设置设备
            ret = acl.rt.set_device(self.device_id)
            if ret != 0:
                raise RuntimeError(f"设置设备失败，错误码: {ret}")
            device_set = True

            try:
                # 创建 context
                self.context, ret = acl.rt.create_context(self.device_id)
                if ret != 0:
                    raise RuntimeError(f"创建 context 失败，错误码: {ret}")

                try:
                    # 创建 stream
                    self.stream, ret = acl.rt.create_stream()
                    if ret != 0:
                        raise RuntimeError(f"创建 stream 失败，错误码: {ret}")

                    # 加载模型
                    self._load_model()

                    logger.info("ACL 运行时环境初始化成功")

                except Exception:
                    # 清理 context
                    if self.context:
                        acl.rt.destroy_context(self.context)
                        self.context = None
                    raise

            except Exception:
                # 清理设备
                if device_set:
                    acl.rt.reset_device(self.device_id)
                raise

        except ImportError:
            logger.error("未安装 ACL Python 包，请确保已安装 CANN toolkit")
            raise
        except Exception as e:
            # 清理 ACL
            if acl_initialized:
                acl.finalize()
            logger.error(f"ACL 初始化失败: {e}")
            raise

    def _load_model(self):
        """加载离线模型"""
        # 加载模型
        self.model_id, ret = self.acl.mdl.load_from_file(self.model_path)
        if ret != 0:
            raise RuntimeError(f"加载模型失败，错误码: {ret}")

        # 获取模型描述
        self.model_desc = self.acl.mdl.create_desc()
        ret = self.acl.mdl.get_desc(self.model_desc, self.model_id)
        if ret != 0:
            raise RuntimeError(f"获取模型描述失败，错误码: {ret}")

        # 创建输入数据集
        self._create_input_dataset()

        # 创建输出数据集
        self._create_output_dataset()

        logger.info(f"模型 {self.model_path} 加载成功")

    def _create_input_dataset(self):
        """创建输入数据集"""
        input_num = self.acl.mdl.get_num_inputs(self.model_desc)
        self.input_dataset = self.acl.mdl.create_dataset()
        self.input_buffers = []

        try:
            for i in range(input_num):
                input_size = self.acl.mdl.get_input_size_by_index(self.model_desc, i)

                # 分配设备内存
                buffer_dev, ret = self.acl.rt.malloc(input_size, 0)
                if ret != 0:
                    raise RuntimeError(f"分配输入内存失败，错误码: {ret}")

                try:
                    # 创建数据缓冲区
                    data_buffer = self.acl.create_data_buffer(buffer_dev, input_size)

                    # 添加到数据集
                    ret = self.acl.mdl.add_dataset_buffer(
                        self.input_dataset, data_buffer
                    )
                    if ret != 0:
                        self.acl.destroy_data_buffer(data_buffer)
                        raise RuntimeError(f"添加输入缓冲区失败，错误码: {ret}")

                    self.input_buffers.append(
                        {
                            "buffer": buffer_dev,
                            "size": input_size,
                            "data_buffer": data_buffer,
                        }
                    )
                except Exception:
                    # 释放刚分配的设备内存
                    self.acl.rt.free(buffer_dev)
                    raise

        except Exception:
            # 清理已创建的缓冲区
            for buf in self.input_buffers:
                self.acl.rt.free(buf["buffer"])
                self.acl.destroy_data_buffer(buf["data_buffer"])
            self.input_buffers = []
            if self.input_dataset:
                self.acl.mdl.destroy_dataset(self.input_dataset)
                self.input_dataset = None
            raise

    def _create_output_dataset(self):
        """创建输出数据集"""
        output_num = self.acl.mdl.get_num_outputs(self.model_desc)
        self.output_dataset = self.acl.mdl.create_dataset()
        self.output_buffers = []

        try:
            for i in range(output_num):
                output_size = self.acl.mdl.get_output_size_by_index(self.model_desc, i)

                # 分配设备内存
                buffer_dev, ret = self.acl.rt.malloc(output_size, 0)
                if ret != 0:
                    raise RuntimeError(f"分配输出内存失败，错误码: {ret}")

                try:
                    # 创建数据缓冲区
                    data_buffer = self.acl.create_data_buffer(buffer_dev, output_size)

                    # 添加到数据集
                    ret = self.acl.mdl.add_dataset_buffer(
                        self.output_dataset, data_buffer
                    )
                    if ret != 0:
                        self.acl.destroy_data_buffer(data_buffer)
                        raise RuntimeError(f"添加输出缓冲区失败，错误码: {ret}")

                    self.output_buffers.append(
                        {
                            "buffer": buffer_dev,
                            "size": output_size,
                            "data_buffer": data_buffer,
                        }
                    )
                except Exception:
                    # 释放刚分配的设备内存
                    self.acl.rt.free(buffer_dev)
                    raise

        except Exception:
            # 清理已创建的缓冲区
            for buf in self.output_buffers:
                self.acl.rt.free(buf["buffer"])
                self.acl.destroy_data_buffer(buf["data_buffer"])
            self.output_buffers = []
            if self.output_dataset:
                self.acl.mdl.destroy_dataset(self.output_dataset)
                self.output_dataset = None
            raise

    def preprocess(self, images: List[np.ndarray]) -> np.ndarray:
        """
        预处理图片

        Args:
            images: 图片列表 (BGR格式)

        Returns:
            预处理后的 batch (NCHW格式)
        """
        batch = []

        for img in images:
            # 调整大小
            img = cv2.resize(img, self.input_size)

            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 归一化
            img = img.astype(np.float32) / 255.0

            # HWC to CHW
            img = np.transpose(img, (2, 0, 1))

            batch.append(img)

        return np.array(batch, dtype=np.float32)

    def infer(self, batch: np.ndarray) -> List[np.ndarray]:
        """
        执行推理

        Args:
            batch: 预处理后的图片 batch

        Returns:
            检测结果列表
        """
        # 使用线程锁保护 ACL 推理操作（ACL Context 非线程安全）
        with self._inference_lock:
            # 拷贝输入数据到设备
            input_data = np.ascontiguousarray(batch)
            input_ptr = self.acl.util.numpy_to_ptr(input_data)
            input_size = input_data.size * input_data.itemsize

            ret = self.acl.rt.memcpy(
                self.input_buffers[0]["buffer"],
                self.input_buffers[0]["size"],
                input_ptr,
                input_size,
                1,  # ACL_MEMCPY_HOST_TO_DEVICE
            )
            if ret != 0:
                raise RuntimeError(f"拷贝输入数据失败，错误码: {ret}")

            # 执行推理
            ret = self.acl.mdl.execute(
                self.model_id, self.input_dataset, self.output_dataset
            )
            if ret != 0:
                raise RuntimeError(f"推理执行失败，错误码: {ret}")

            # 拷贝输出数据到主机
            outputs = []
            for i, out_buf in enumerate(self.output_buffers):
                # 获取输出形状信息
                output_dims = self.acl.mdl.get_output_dims(self.model_desc, i)
                output_shape = [dim for dim in output_dims[0]["dims"] if dim != 0]

                # 分配主机内存
                output_data = np.zeros(output_shape, dtype=np.float32)
                output_ptr = self.acl.util.numpy_to_ptr(output_data)

                ret = self.acl.rt.memcpy(
                    output_ptr,
                    output_data.nbytes,
                    out_buf["buffer"],
                    out_buf["size"],
                    2,  # ACL_MEMCPY_DEVICE_TO_HOST
                )
                if ret != 0:
                    raise RuntimeError(f"拷贝输出数据失败，错误码: {ret}")

                outputs.append(output_data)

            return outputs

    def postprocess(
        self, outputs: List[np.ndarray], orig_shapes: List[Tuple[int, int]]
    ) -> List[List[Tuple[int, int, int, int, float]]]:
        """
        后处理检测结果

        Args:
            outputs: 模型输出
            orig_shapes: 原始图片尺寸列表 [(h, w), ...]

        Returns:
            检测框列表 [[(x1, y1, x2, y2, conf), ...], ...]
        """
        # 解析输出 (假设 YOLO 格式输出)
        if not outputs or len(outputs) == 0:
            return [[] for _ in orig_shapes]

        detections = outputs[0]

        results = []
        batch_size = len(orig_shapes)

        for i, (orig_h, orig_w) in enumerate(orig_shapes):
            if len(detections.shape) == 3:
                # [batch, num_dets, 6]
                if i >= detections.shape[0]:
                    results.append([])
                    continue
                batch_dets = detections[i]
            else:
                # [num_dets, 6] - 使用 array_split 安全分割
                batch_dets_list = np.array_split(detections, batch_size)
                if i < len(batch_dets_list):
                    batch_dets = batch_dets_list[i]
                else:
                    batch_dets = np.array([])

            # 过滤低置信度检测 (增加维度检查)
            if len(batch_dets) > 0 and batch_dets.ndim == 2:
                if batch_dets.shape[-1] >= 5:
                    conf_idx = 4
                    mask = batch_dets[:, conf_idx] > self.conf_threshold
                    batch_dets = batch_dets[mask]
                else:
                    logger.warning(f"检测输出维度异常: {batch_dets.shape}，跳过此batch")
                    batch_dets = np.array([])

            if len(batch_dets) == 0:
                results.append([])
                continue

            # 缩放到原始尺寸
            scale_x = orig_w / self.input_size[0]
            scale_y = orig_h / self.input_size[1]

            batch_dets[:, 0] *= scale_x  # x1
            batch_dets[:, 1] *= scale_y  # y1
            batch_dets[:, 2] *= scale_x  # x2
            batch_dets[:, 3] *= scale_y  # y2

            # NMS
            boxes = batch_dets[:, :4]
            scores = batch_dets[:, 4]
            indices = self._nms(boxes, scores, self.nms_threshold)

            # 提取结果
            final_dets = [
                (int(det[0]), int(det[1]), int(det[2]), int(det[3]), float(det[4]))
                for det in batch_dets[indices]
            ]

            results.append(final_dets)

        return results

    def _nms(
        self, boxes: np.ndarray, scores: np.ndarray, threshold: float
    ) -> List[int]:
        """Non-Maximum Suppression using OpenCV (优化性能)"""
        # 转换为 OpenCV 格式
        boxes_list = boxes.tolist()
        scores_list = scores.tolist()

        # 使用 OpenCV 的 NMS 实现 (比手写实现快很多)
        indices = cv2.dnn.NMSBoxes(boxes_list, scores_list, 0.0, threshold)

        # 返回索引列表
        if len(indices) > 0:
            return indices.flatten().tolist()
        else:
            return []

    def detect_faces(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """
        检测单张图像中的人脸

        Args:
            frame: BGR 格式的图像

        Returns:
            boxes: 人脸边界框列表 [(x1, y1, x2, y2), ...]
            confidences: 置信度列表
        """
        results = self.detect_batch([frame])
        if results and results[0]:
            boxes = [(r[0], r[1], r[2], r[3]) for r in results[0]]
            confs = [r[4] for r in results[0]]
            return boxes, confs
        return [], []

    def detect_batch(
        self, images: List[np.ndarray]
    ) -> List[List[Tuple[int, int, int, int, float]]]:
        """
        批量检测人脸

        Args:
            images: 图片列表 (BGR 格式)

        Returns:
            检测结果列表
        """
        # 预处理
        orig_shapes = [(img.shape[0], img.shape[1]) for img in images]
        batch = self.preprocess(images)

        # 推理
        outputs = self.infer(batch)

        # 后处理
        results = self.postprocess(outputs, orig_shapes)

        return results

    def draw_detections(
        self, image: np.ndarray, detections: List[Tuple[int, int, int, int, float]]
    ) -> np.ndarray:
        """在图片上绘制检测框"""
        result = image.copy()

        for i, (x1, y1, x2, y2, conf) in enumerate(detections):
            # 绘制矩形
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制标签
            label = f"Face {i + 1}: {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                result, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1
            )
            cv2.putText(
                result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
            )

        return result

    def release(self):
        """释放资源"""
        try:
            # 释放输入缓冲区
            if hasattr(self, "input_buffers"):
                for buf in self.input_buffers:
                    self.acl.rt.free(buf["buffer"])
                    self.acl.destroy_data_buffer(buf["data_buffer"])

            # 释放输出缓冲区
            if hasattr(self, "output_buffers"):
                for buf in self.output_buffers:
                    self.acl.rt.free(buf["buffer"])
                    self.acl.destroy_data_buffer(buf["data_buffer"])

            # 销毁数据集
            if self.input_dataset:
                self.acl.mdl.destroy_dataset(self.input_dataset)
            if self.output_dataset:
                self.acl.mdl.destroy_dataset(self.output_dataset)

            # 卸载模型
            if self.model_id:
                self.acl.mdl.unload(self.model_id)
            if self.model_desc:
                self.acl.mdl.destroy_desc(self.model_desc)

            # 销毁 stream 和 context
            if self.stream:
                self.acl.rt.destroy_stream(self.stream)
            if self.context:
                self.acl.rt.destroy_context(self.context)

            # 重置设备
            self.acl.rt.reset_device(self.device_id)

            # 终止 ACL
            self.acl.finalize()

            logger.info("ACL 资源已释放")

        except Exception as e:
            logger.error(f"释放 ACL 资源失败: {e}")

    def __del__(self):
        """析构函数"""
        self.release()


class AscendFaceDetector:
    """
    昇腾人脸检测器（自动降级）
    优先使用 ACL，失败时降级到 OpenCV
    """

    def __init__(
        self,
        model_path: str = None,
        device_id: int = 0,
        conf_threshold: float = 0.5,
    ):
        """
        初始化检测器

        Args:
            model_path: .om 模型路径
            device_id: NPU 设备 ID
            conf_threshold: 置信度阈值
        """
        self.device_id = device_id
        self.conf_threshold = conf_threshold
        self.backend = None

        if model_path and Path(model_path).exists():
            try:
                self.detector = ACLFaceDetector(
                    model_path=model_path,
                    device_id=device_id,
                    conf_threshold=conf_threshold,
                )
                self.backend = "acl"
                logger.info("使用 ACL 后端")
            except Exception as e:
                logger.warning(f"ACL 初始化失败: {e}")
                self._init_opencv_backend()
        else:
            logger.info(f"模型文件不存在: {model_path}")
            self._init_opencv_backend()

    def _init_opencv_backend(self):
        """初始化 OpenCV 后端"""
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(cascade_path)
            self.backend = "opencv"
            logger.info("降级到 OpenCV 后端")
        except Exception as e:
            logger.error(f"OpenCV 初始化失败: {e}")
            raise

    def detect_faces(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """检测人脸"""
        if self.backend == "acl":
            return self.detector.detect_faces(frame)
        else:
            # OpenCV 后端
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            boxes = [(x, y, x + w, y + h) for (x, y, w, h) in faces]
            confs = [1.0] * len(faces)
            return boxes, confs

    def detect_batch(
        self, images: List[np.ndarray]
    ) -> List[List[Tuple[int, int, int, int, float]]]:
        """批量检测"""
        if self.backend == "acl":
            return self.detector.detect_batch(images)
        else:
            results = []
            for img in images:
                boxes, confs = self.detect_faces(img)
                dets = [
                    (box[0], box[1], box[2], box[3], conf)
                    for box, conf in zip(boxes, confs)
                ]
                results.append(dets)
            return results

    def draw_faces(
        self,
        frame: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        confidences: List[float],
    ) -> np.ndarray:
        """绘制检测框"""
        result = frame.copy()

        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"Face {i + 1}: {conf:.2f}"
            cv2.putText(
                result,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return result

    def release(self):
        """释放资源"""
        if self.backend == "acl" and hasattr(self.detector, "release"):
            self.detector.release()


def main():
    """测试函数"""
    import argparse

    parser = argparse.ArgumentParser(description="昇腾人脸检测器测试")
    parser.add_argument("--model", required=True, help=".om 模型路径")
    parser.add_argument("--image", required=True, help="测试图片路径")
    parser.add_argument("--device", type=int, default=0, help="NPU 设备 ID")

    args = parser.parse_args()

    # 加载检测器
    detector = AscendFaceDetector(
        model_path=args.model,
        device_id=args.device,
    )

    # 加载图片
    img = cv2.imread(args.image)
    if img is None:
        logger.error(f"无法读取图片: {args.image}")
        return

    # 检测
    import time

    start = time.time()
    boxes, confs = detector.detect_faces(img)
    elapsed = (time.time() - start) * 1000

    # 打印结果
    print(f"检测到 {len(boxes)} 张人脸")
    for i, (box, conf) in enumerate(zip(boxes, confs)):
        print(f"  人脸 {i + 1}: {box}, 置信度={conf:.3f}")

    print(f"\n推理时间: {elapsed:.2f}ms")

    # 可视化
    result = detector.draw_faces(img, boxes, confs)
    cv2.imwrite("ascend_detection_result.jpg", result)
    print("结果保存到 ascend_detection_result.jpg")

    # 释放资源
    detector.release()


if __name__ == "__main__":
    main()
