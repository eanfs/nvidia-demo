"""
优化版人脸检测器 - 使用 TensorRT INT8 引擎
专门针对 70 路 1080p 场景优化
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TensorRTFaceDetector:
    """TensorRT 加速的人脸检测器"""

    def __init__(
        self,
        engine_path: str,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
    ):
        """
        初始化 TensorRT 人脸检测器

        Args:
            engine_path: TensorRT 引擎文件路径
            input_size: 输入尺寸 (width, height)
            conf_threshold: 置信度阈值
            nms_threshold: NMS 阈值
        """
        self.engine_path = engine_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        # 加载 TensorRT 引擎
        self._load_engine()

        logger.info(f"TensorRT 人脸检测器已初始化，引擎: {engine_path}")

    def _load_engine(self):
        """加载 TensorRT 引擎"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # 自动初始化 CUDA

            self.trt = trt
            self.cuda = cuda

        except ImportError as e:
            logger.error(f"导入失败: {e}")
            logger.info("请安装: pip install tensorrt pycuda")
            raise

        # 加载引擎
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        with open(self.engine_path, "rb") as f:
            engine_data = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # 分配 GPU 内存
        self.stream = cuda.Stream()
        self._allocate_buffers()

        logger.info("TensorRT 引擎加载成功")

    def _allocate_buffers(self):
        """分配输入输出缓冲区"""
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for binding in self.engine:
            size = self.trt.volume(self.engine.get_binding_shape(binding))
            dtype = self.trt.nptype(self.engine.get_binding_dtype(binding))

            # 分配 GPU 内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})

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
        # 拷贝输入数据到 GPU
        np.copyto(self.inputs[0]["host"], batch.ravel())
        self.cuda.memcpy_htod_async(
            self.inputs[0]["device"], self.inputs[0]["host"], self.stream
        )

        # 执行推理
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle
        )

        # 拷贝输出数据到 CPU
        for output in self.outputs:
            self.cuda.memcpy_dtoh_async(output["host"], output["device"], self.stream)

        # 同步
        self.stream.synchronize()

        # 返回输出
        return [output["host"] for output in self.outputs]

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
        # 解析输出
        # 假设输出格式: [batch_size, num_detections, 6]  (x1, y1, x2, y2, conf, class)
        detections = outputs[0].reshape(-1, 6)

        results = []
        batch_size = len(orig_shapes)
        dets_per_image = len(detections) // batch_size

        for i, (orig_h, orig_w) in enumerate(orig_shapes):
            batch_dets = detections[i * dets_per_image : (i + 1) * dets_per_image]

            # 过滤低置信度检测
            mask = batch_dets[:, 4] > self.conf_threshold
            batch_dets = batch_dets[mask]

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
        """
        Non-Maximum Suppression

        Args:
            boxes: 边界框 (N, 4) [x1, y1, x2, y2]
            scores: 置信度 (N,)
            threshold: IOU 阈值

        Returns:
            保留的索引列表
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return keep

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
        """
        在图片上绘制检测框

        Args:
            image: 原始图片
            detections: 检测结果 [(x1, y1, x2, y2, conf), ...]

        Returns:
            绘制后的图片
        """
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


class OptimizedFaceDetector:
    """
    优化版人脸检测器（自动降级）
    优先使用 TensorRT，失败时降级到 PyTorch
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        初始化检测器

        Args:
            model_path: 模型路径 (.engine 或 .pt)
            device: 设备类型
        """
        self.model_path = model_path
        self.device = device

        if model_path.endswith(".engine"):
            try:
                self.detector = TensorRTFaceDetector(model_path)
                self.backend = "tensorrt"
                logger.info("使用 TensorRT 后端")
            except Exception as e:
                logger.warning(f"TensorRT 初始化失败: {e}")
                logger.info("降级到 PyTorch 后端")
                self._init_pytorch_backend()
        else:
            self._init_pytorch_backend()

    def _init_pytorch_backend(self):
        """初始化 PyTorch 后端"""
        from face_detector import FaceDetector

        self.detector = FaceDetector(device=self.device, use_mtcnn=True)
        self.backend = "pytorch"
        logger.info("使用 PyTorch 后端")

    def detect_batch(self, images: List[np.ndarray]):
        """批量检测"""
        if self.backend == "tensorrt":
            return self.detector.detect_batch(images)
        else:
            # PyTorch 后端逐张检测
            results = []
            for img in images:
                boxes, confs = self.detector.detect_faces(img)
                # 转换格式
                dets = [
                    (box[0], box[1], box[2], box[3], conf)
                    for box, conf in zip(boxes, confs)
                ]
                results.append(dets)
            return results


def main():
    """测试函数"""
    import argparse

    parser = argparse.ArgumentParser(description="TensorRT 人脸检测器测试")
    parser.add_argument("--engine", required=True, help="TensorRT 引擎路径")
    parser.add_argument("--image", required=True, help="测试图片路径")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")

    args = parser.parse_args()

    # 加载检测器
    detector = TensorRTFaceDetector(args.engine)

    # 加载图片
    images = []
    for _ in range(args.batch):
        img = cv2.imread(args.image)
        if img is None:
            logger.error(f"无法读取图片: {args.image}")
            return
        images.append(img)

    # 检测
    import time

    start = time.time()
    results = detector.detect_batch(images)
    elapsed = (time.time() - start) * 1000

    # 打印结果
    for i, dets in enumerate(results):
        print(f"Image {i}: {len(dets)} faces detected")
        for j, (x1, y1, x2, y2, conf) in enumerate(dets):
            print(f"  Face {j}: ({x1}, {y1}, {x2}, {y2}), conf={conf:.3f}")

    print(f"\nInference time: {elapsed:.2f}ms")
    print(f"Per-image: {elapsed / len(images):.2f}ms")

    # 可视化第一张
    if len(results) > 0 and len(results[0]) > 0:
        vis = detector.draw_detections(images[0], results[0])
        cv2.imwrite("detection_result.jpg", vis)
        print("Result saved to detection_result.jpg")


if __name__ == "__main__":
    main()
