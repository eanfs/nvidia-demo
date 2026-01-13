"""
昇腾模型转换工具
将 ONNX 模型转换为昇腾 .om 离线模型格式
"""

import subprocess
import logging
import os
from pathlib import Path
from typing import Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AscendModelConverter:
    """
    昇腾模型转换器
    使用 ATC (Ascend Tensor Compiler) 将模型转换为 .om 格式
    """

    # 支持的 SOC 版本
    SOC_VERSIONS = {
        "atlas300v": "Ascend310P",
        "atlas300i": "Ascend310",
        "atlas500": "Ascend310",
        "atlas800": "Ascend910",
        "atlas900": "Ascend910",
    }

    # 框架类型
    FRAMEWORKS = {
        "caffe": 0,
        "tensorflow": 1,
        "onnx": 5,
    }

    def __init__(self, cann_home: str = None):
        """
        初始化模型转换器

        Args:
            cann_home: CANN 安装目录
        """
        self.cann_home = cann_home or os.environ.get(
            "CANN_HOME", "/usr/local/Ascend/ascend-toolkit/latest"
        )

        # 检查 ATC 工具是否可用
        self._check_atc()

    def _check_atc(self):
        """检查 ATC 工具是否可用"""
        try:
            result = subprocess.run(
                ["atc", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                logger.info("ATC 工具可用")
            else:
                logger.warning(f"ATC 返回非零状态: {result.returncode}")
        except FileNotFoundError:
            logger.error("找不到 atc 命令，请确保 CANN 已正确安装并配置环境变量")
            logger.info(f"尝试设置: source {self.cann_home}/set_env.sh")
            raise
        except subprocess.TimeoutExpired:
            logger.error("ATC 命令执行超时")
            raise

    def convert_onnx(
        self,
        model_path: str,
        output_path: str,
        soc_version: str = "Ascend310P",
        input_shape: str = None,
        input_format: str = "NCHW",
        output_type: str = "FP16",
        dynamic_batch_size: str = None,
        dynamic_image_size: str = None,
        log_level: str = "error",
    ) -> str:
        """
        转换 ONNX 模型为 .om 格式

        Args:
            model_path: ONNX 模型文件路径
            output_path: 输出 .om 模型路径 (不含扩展名)
            soc_version: 目标芯片型号 (Ascend310P, Ascend310, Ascend910)
            input_shape: 输入形状，如 "input:1,3,640,640"
            input_format: 输入格式 (NCHW, NHWC, ND)
            output_type: 输出类型 (FP32, FP16, INT8)
            dynamic_batch_size: 动态 batch size，如 "1,2,4,8"
            dynamic_image_size: 动态图像尺寸，如 "640,640;320,320"
            log_level: 日志级别 (error, info, debug)

        Returns:
            转换后的 .om 模型路径
        """
        # 检查输入文件
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 构建 ATC 命令
        cmd = [
            "atc",
            "--framework=5",  # ONNX
            f"--model={model_path}",
            f"--output={output_path}",
            f"--soc_version={soc_version}",
            f"--input_format={input_format}",
            f"--log={log_level}",
        ]

        # 添加可选参数
        if input_shape:
            cmd.append(f"--input_shape={input_shape}")

        if output_type:
            output_type_map = {"FP32": 0, "FP16": 1, "INT8": 3, "UINT8": 4}
            if output_type in output_type_map:
                cmd.append(f"--output_type={output_type_map[output_type]}")

        if dynamic_batch_size:
            cmd.append(f"--dynamic_batch_size={dynamic_batch_size}")

        if dynamic_image_size:
            cmd.append(f"--dynamic_image_size={dynamic_image_size}")

        logger.info(f"执行 ATC 命令: {' '.join(cmd)}")

        # 执行转换
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10分钟超时
            )

            if result.returncode == 0:
                om_path = f"{output_path}.om"
                logger.info(f"模型转换成功: {om_path}")
                return om_path
            else:
                logger.error(f"模型转换失败:\n{result.stderr}")
                raise RuntimeError(f"ATC 转换失败: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.error("模型转换超时")
            raise

    def convert_pytorch_to_om(
        self,
        model_path: str,
        output_path: str,
        input_shape: tuple = (1, 3, 640, 640),
        soc_version: str = "Ascend310P",
        **kwargs,
    ) -> str:
        """
        将 PyTorch 模型转换为 .om 格式

        Args:
            model_path: PyTorch 模型文件路径 (.pt/.pth)
            output_path: 输出路径
            input_shape: 输入形状
            soc_version: 目标芯片型号
            **kwargs: 其他 ATC 参数

        Returns:
            转换后的 .om 模型路径
        """
        import torch

        # 中间 ONNX 文件路径
        onnx_path = output_path + ".onnx"

        logger.info(f"正在加载 PyTorch 模型: {model_path}")

        # 加载模型
        model = torch.load(model_path, map_location="cpu")
        if hasattr(model, "eval"):
            model.eval()

        # 创建示例输入
        dummy_input = torch.randn(*input_shape)

        # 导出 ONNX
        logger.info(f"导出 ONNX 模型: {onnx_path}")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=11,
            dynamic_axes=None,
        )

        # 转换为 OM
        input_shape_str = f"input:{','.join(map(str, input_shape))}"
        om_path = self.convert_onnx(
            model_path=onnx_path,
            output_path=output_path,
            soc_version=soc_version,
            input_shape=input_shape_str,
            **kwargs,
        )

        # 删除中间 ONNX 文件 (可选)
        # os.remove(onnx_path)

        return om_path

    def get_model_info(self, model_path: str) -> dict:
        """
        获取 .om 模型信息

        Args:
            model_path: .om 模型路径

        Returns:
            模型信息字典
        """
        try:
            import acl

            # 初始化
            ret = acl.init()
            if ret != 0:
                raise RuntimeError(f"ACL 初始化失败: {ret}")

            ret = acl.rt.set_device(0)
            if ret != 0:
                raise RuntimeError(f"设置设备失败: {ret}")

            # 加载模型
            model_id, ret = acl.mdl.load_from_file(model_path)
            if ret != 0:
                raise RuntimeError(f"加载模型失败: {ret}")

            # 获取模型描述
            model_desc = acl.mdl.get_desc(model_id)

            # 获取输入输出信息
            input_num = acl.mdl.get_num_inputs(model_desc)
            output_num = acl.mdl.get_num_outputs(model_desc)

            info = {
                "model_path": model_path,
                "input_num": input_num,
                "output_num": output_num,
                "inputs": [],
                "outputs": [],
            }

            for i in range(input_num):
                input_info = {
                    "index": i,
                    "name": acl.mdl.get_input_name_by_index(model_desc, i),
                    "size": acl.mdl.get_input_size_by_index(model_desc, i, 0),
                    "dims": acl.mdl.get_input_dims(model_desc, i),
                }
                info["inputs"].append(input_info)

            for i in range(output_num):
                output_info = {
                    "index": i,
                    "name": acl.mdl.get_output_name_by_index(model_desc, i),
                    "size": acl.mdl.get_output_size_by_index(model_desc, i, 0),
                    "dims": acl.mdl.get_output_dims(model_desc, i),
                }
                info["outputs"].append(output_info)

            # 卸载模型
            acl.mdl.unload(model_id)
            acl.rt.reset_device(0)
            acl.finalize()

            return info

        except ImportError:
            logger.warning("ACL 库未安装，无法获取模型信息")
            return {"model_path": model_path, "error": "ACL not available"}

    def validate_model(self, om_path: str, test_input: "np.ndarray" = None) -> bool:
        """
        验证 .om 模型是否可用

        Args:
            om_path: .om 模型路径
            test_input: 测试输入 (可选)

        Returns:
            是否验证通过
        """
        try:
            import acl
            import numpy as np

            # 初始化
            ret = acl.init()
            if ret != 0:
                return False

            ret = acl.rt.set_device(0)
            if ret != 0:
                return False

            # 加载模型
            model_id, ret = acl.mdl.load_from_file(om_path)
            if ret != 0:
                logger.error(f"模型加载失败: {ret}")
                return False

            logger.info(f"模型验证通过: {om_path}")

            # 卸载模型
            acl.mdl.unload(model_id)
            acl.rt.reset_device(0)
            acl.finalize()

            return True

        except Exception as e:
            logger.error(f"模型验证失败: {e}")
            return False


def main():
    """命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description="昇腾模型转换工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # convert 子命令
    convert_parser = subparsers.add_parser("convert", help="转换模型")
    convert_parser.add_argument("--model", required=True, help="输入模型路径")
    convert_parser.add_argument("--output", required=True, help="输出路径 (不含扩展名)")
    convert_parser.add_argument(
        "--framework",
        choices=["onnx", "pytorch"],
        default="onnx",
        help="输入模型框架",
    )
    convert_parser.add_argument(
        "--soc", default="Ascend310P", help="目标芯片 (默认: Ascend310P)"
    )
    convert_parser.add_argument("--input-shape", help="输入形状，如 input:1,3,640,640")
    convert_parser.add_argument(
        "--output-type", default="FP16", help="输出类型 (FP32/FP16/INT8)"
    )
    convert_parser.add_argument("--dynamic-batch", help="动态 batch size")

    # info 子命令
    info_parser = subparsers.add_parser("info", help="获取模型信息")
    info_parser.add_argument("--model", required=True, help=".om 模型路径")

    # validate 子命令
    validate_parser = subparsers.add_parser("validate", help="验证模型")
    validate_parser.add_argument("--model", required=True, help=".om 模型路径")

    args = parser.parse_args()

    converter = AscendModelConverter()

    if args.command == "convert":
        if args.framework == "onnx":
            om_path = converter.convert_onnx(
                model_path=args.model,
                output_path=args.output,
                soc_version=args.soc,
                input_shape=args.input_shape,
                output_type=args.output_type,
                dynamic_batch_size=args.dynamic_batch,
            )
        else:
            # 解析 input_shape
            if args.input_shape:
                shape_str = args.input_shape.split(":")[1]
                shape = tuple(map(int, shape_str.split(",")))
            else:
                shape = (1, 3, 640, 640)

            om_path = converter.convert_pytorch_to_om(
                model_path=args.model,
                output_path=args.output,
                input_shape=shape,
                soc_version=args.soc,
                output_type=args.output_type,
            )

        print(f"转换完成: {om_path}")

    elif args.command == "info":
        info = converter.get_model_info(args.model)
        print("\n模型信息:")
        print(f"  路径: {info['model_path']}")
        if "error" not in info:
            print(f"  输入数量: {info['input_num']}")
            print(f"  输出数量: {info['output_num']}")
            print("  输入:")
            for inp in info["inputs"]:
                print(f"    [{inp['index']}] {inp['name']}: {inp['dims']}")
            print("  输出:")
            for out in info["outputs"]:
                print(f"    [{out['index']}] {out['name']}: {out['dims']}")
        else:
            print(f"  错误: {info['error']}")

    elif args.command == "validate":
        if converter.validate_model(args.model):
            print("模型验证通过!")
        else:
            print("模型验证失败!")
            exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
