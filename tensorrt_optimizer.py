"""
TensorRT INT8 量化优化脚本
用于将 PyTorch/ONNX 模型转换为 TensorRT INT8 引擎
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Int8Calibrator:
    """INT8 校准器 - 用于 TensorRT 量化"""

    def __init__(
        self,
        calibration_images: List[str],
        cache_file: str,
        batch_size: int = 8,
        input_shape: Tuple[int, int, int, int] = (8, 3, 640, 640),
    ):
        """
        初始化校准器

        Args:
            calibration_images: 校准图片路径列表
            cache_file: 校准缓存文件路径
            batch_size: 批大小
            input_shape: 输入形状 (N, C, H, W)
        """
        self.calibration_images = calibration_images
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.current_index = 0

        logger.info(f"初始化 INT8 校准器，校准图片数: {len(calibration_images)}")

    def prepare_batch(self) -> np.ndarray:
        """准备一个 batch 的数据"""
        batch = []

        for i in range(self.batch_size):
            if self.current_index >= len(self.calibration_images):
                break

            img_path = self.calibration_images[self.current_index]
            self.current_index += 1

            # 读取和预处理图片
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"无法读取图片: {img_path}")
                continue

            # 调整大小
            h, w = self.input_shape[2], self.input_shape[3]
            img = cv2.resize(img, (w, h))

            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 归一化到 [0, 1]
            img = img.astype(np.float32) / 255.0

            # HWC to CHW
            img = np.transpose(img, (2, 0, 1))

            batch.append(img)

        if len(batch) < self.batch_size:
            # 填充到 batch_size
            while len(batch) < self.batch_size:
                batch.append(
                    np.zeros(
                        (3, self.input_shape[2], self.input_shape[3]), dtype=np.float32
                    )
                )

        return np.array(batch, dtype=np.float32)

    def get_batch(self) -> np.ndarray:
        """获取下一个 batch"""
        if self.current_index >= len(self.calibration_images):
            return None

        batch = self.prepare_batch()
        return batch


def collect_calibration_images(rtsp_url: str, output_dir: str, num_samples: int = 1000):
    """
    从 RTSP 流收集校准图片

    Args:
        rtsp_url: RTSP 流地址
        output_dir: 输出目录
        num_samples: 采样数量
    """
    logger.info(f"从 {rtsp_url} 收集 {num_samples} 张校准图片")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        logger.error(f"无法打开 RTSP 流: {rtsp_url}")
        return

    collected = 0
    frame_count = 0

    # 计算采样间隔
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(fps * 60)  # 1 分钟视频
    sample_interval = max(1, total_frames // num_samples)

    while collected < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 间隔采样
        if frame_count % sample_interval != 0:
            continue

        # 保存图片
        img_path = os.path.join(output_dir, f"calib_{collected:04d}.jpg")
        cv2.imwrite(img_path, frame)

        collected += 1

        if collected % 100 == 0:
            logger.info(f"已收集 {collected}/{num_samples} 张图片")

    cap.release()
    logger.info(f"校准图片收集完成: {collected} 张，保存到 {output_dir}")


def export_onnx_model(
    pytorch_model_path: str,
    onnx_output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
):
    """
    将 PyTorch 模型导出为 ONNX 格式

    Args:
        pytorch_model_path: PyTorch 模型路径 (.pt)
        onnx_output_path: ONNX 输出路径 (.onnx)
        input_shape: 输入形状
    """
    try:
        import torch

        logger.info(f"加载 PyTorch 模型: {pytorch_model_path}")

        # 加载模型
        model = torch.load(pytorch_model_path, map_location="cpu")
        if hasattr(model, "model"):
            model = model.model

        model.eval()

        # 创建示例输入
        dummy_input = torch.randn(*input_shape)

        # 导出 ONNX
        logger.info(f"导出 ONNX 模型到: {onnx_output_path}")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_output_path,
            opset_version=11,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        logger.info("ONNX 导出成功")

    except Exception as e:
        logger.error(f"ONNX 导出失败: {e}")
        raise


def build_tensorrt_engine(
    onnx_model_path: str,
    engine_output_path: str,
    calibration_dir: str = None,
    precision: str = "int8",
    min_batch: int = 1,
    opt_batch: int = 16,
    max_batch: int = 32,
):
    """
    使用 TensorRT 构建优化引擎

    Args:
        onnx_model_path: ONNX 模型路径
        engine_output_path: TensorRT 引擎输出路径
        calibration_dir: 校准数据目录（INT8 需要）
        precision: 精度类型 ('fp32', 'fp16', 'int8')
        min_batch: 最小 batch size
        opt_batch: 最优 batch size
        max_batch: 最大 batch size
    """
    logger.info(f"构建 TensorRT {precision.upper()} 引擎")

    # 使用 trtexec 命令行工具（更稳定）
    cmd = [
        "trtexec",
        f"--onnx={onnx_model_path}",
        f"--saveEngine={engine_output_path}",
        f"--minShapes=input:{min_batch}x3x640x640",
        f"--optShapes=input:{opt_batch}x3x640x640",
        f"--maxShapes=input:{max_batch}x3x640x640",
        "--workspace=4096",  # 4GB
    ]

    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.append("--int8")
        if calibration_dir:
            # 准备校准缓存
            calib_images = [
                os.path.join(calibration_dir, f)
                for f in os.listdir(calibration_dir)
                if f.endswith((".jpg", ".png"))
            ]
            logger.info(f"使用 {len(calib_images)} 张图片进行 INT8 校准")

            # 注意: trtexec 需要自定义校准器，这里提供 Python API 示例
            logger.warning("使用 Python API 进行 INT8 校准（trtexec 需要 C++ 校准器）")
            return build_tensorrt_engine_python_api(
                onnx_model_path,
                engine_output_path,
                calib_images,
                min_batch,
                opt_batch,
                max_batch,
            )

    # 执行命令
    import subprocess

    logger.info(f"执行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(result.stdout)
        logger.info(f"TensorRT 引擎构建成功: {engine_output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"TensorRT 构建失败: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("trtexec 未找到，请安装 TensorRT")
        logger.info("尝试使用 Python API...")
        return build_tensorrt_engine_python_api(
            onnx_model_path,
            engine_output_path,
            None,
            min_batch,
            opt_batch,
            max_batch,
            precision,
        )


def build_tensorrt_engine_python_api(
    onnx_path: str,
    engine_path: str,
    calib_images: List[str] = None,
    min_batch: int = 1,
    opt_batch: int = 16,
    max_batch: int = 32,
    precision: str = "int8",
):
    """
    使用 TensorRT Python API 构建引擎（备用方案）
    """
    try:
        import tensorrt as trt

        logger.info("使用 TensorRT Python API")

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # 解析 ONNX
        logger.info(f"解析 ONNX 模型: {onnx_path}")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("ONNX 解析失败")

        # 配置
        config = builder.create_builder_config()
        config.max_workspace_size = 4 << 30  # 4GB

        # 设置精度
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)

            if calib_images:
                # 设置 INT8 校准器
                cache_file = engine_path.replace(".engine", "_calib.cache")
                calibrator = Int8EntropyCalibrator(
                    calib_images, cache_file, batch_size=8
                )
                config.int8_calibrator = calibrator

        # 优化配置
        profile = builder.create_optimization_profile()
        profile.set_shape(
            "input",
            (min_batch, 3, 640, 640),
            (opt_batch, 3, 640, 640),
            (max_batch, 3, 640, 640),
        )
        config.add_optimization_profile(profile)

        # 构建引擎
        logger.info("开始构建 TensorRT 引擎（可能需要几分钟）...")
        engine = builder.build_engine(network, config)

        if engine is None:
            raise RuntimeError("引擎构建失败")

        # 保存引擎
        logger.info(f"保存引擎到: {engine_path}")
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())

        logger.info("TensorRT 引擎构建成功！")

    except ImportError:
        logger.error("TensorRT Python 包未安装")
        logger.info("安装命令: pip install tensorrt")
        raise
    except Exception as e:
        logger.error(f"引擎构建失败: {e}")
        raise


class Int8EntropyCalibrator:
    """TensorRT INT8 熵校准器"""

    def __init__(self, calib_images: List[str], cache_file: str, batch_size: int = 8):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit

            self.trt = trt
            self.cuda = cuda

        except ImportError as e:
            logger.error(f"导入失败: {e}")
            raise

        self.calibrator_base = self.trt.IInt8EntropyCalibrator2
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.calib_images = calib_images
        self.current_index = 0

        # 分配 GPU 内存
        self.device_input = self.cuda.mem_alloc(batch_size * 3 * 640 * 640 * 4)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.calib_images):
            return None

        calibrator = Int8Calibrator(
            self.calib_images[self.current_index :], self.cache_file, self.batch_size
        )
        batch = calibrator.get_batch()

        if batch is None:
            return None

        self.current_index += self.batch_size

        # 拷贝到 GPU
        self.cuda.memcpy_htod(self.device_input, batch.ravel())
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TensorRT INT8 量化工具")

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # 收集校准数据
    collect_parser = subparsers.add_parser("collect", help="收集校准数据")
    collect_parser.add_argument("--rtsp-url", required=True, help="RTSP 流地址")
    collect_parser.add_argument(
        "--output-dir", default="./calibration_data", help="输出目录"
    )
    collect_parser.add_argument(
        "--num-samples", type=int, default=1000, help="采样数量"
    )

    # 导出 ONNX
    export_parser = subparsers.add_parser("export", help="导出 ONNX 模型")
    export_parser.add_argument("--model", required=True, help="PyTorch 模型路径")
    export_parser.add_argument("--output", default="model.onnx", help="ONNX 输出路径")

    # 构建引擎
    build_parser = subparsers.add_parser("build", help="构建 TensorRT 引擎")
    build_parser.add_argument("--onnx", required=True, help="ONNX 模型路径")
    build_parser.add_argument("--output", required=True, help="引擎输出路径")
    build_parser.add_argument(
        "--precision", choices=["fp32", "fp16", "int8"], default="int8", help="精度类型"
    )
    build_parser.add_argument("--calib-dir", help="校准数据目录（INT8 需要）")
    build_parser.add_argument("--min-batch", type=int, default=1, help="最小 batch")
    build_parser.add_argument("--opt-batch", type=int, default=16, help="最优 batch")
    build_parser.add_argument("--max-batch", type=int, default=32, help="最大 batch")

    args = parser.parse_args()

    if args.command == "collect":
        collect_calibration_images(args.rtsp_url, args.output_dir, args.num_samples)

    elif args.command == "export":
        export_onnx_model(args.model, args.output)

    elif args.command == "build":
        if args.precision == "int8" and not args.calib_dir:
            logger.error("INT8 量化需要提供校准数据目录 (--calib-dir)")
            sys.exit(1)

        build_tensorrt_engine(
            args.onnx,
            args.output,
            args.calib_dir,
            args.precision,
            args.min_batch,
            args.opt_batch,
            args.max_batch,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
