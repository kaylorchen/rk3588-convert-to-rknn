#!/bin/python
import argparse
from ultralytics import YOLO
import onnx
from onnx import shape_inference

# 设置命令行参数解析器
parser = argparse.ArgumentParser(description="Convert a YOLO model to ONNX format.")
parser.add_argument(
    "model_name",
    type=str,
    help='Name of the YOLO model, e.g., "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x".',
)
parser.add_argument(
    "--optimize_cpu",
    default=True,
    action="store_true",
    help="Whether to optimize the model for CPU.",
)
parser.add_argument(
    "--input_width", type=int, default=640, help="Width of the input image."
)
parser.add_argument(
    "--input_height", type=int, default=640, help="Height of the input image."
)

# 解析命令行参数
args = parser.parse_args()

# 打印输入的参数
print(f"Model name: {args.model_name}")
print(f"Optimize for CPU: {'Yes' if args.optimize_cpu else 'No'}")
print(f"Input width: {args.input_width}")
print(f"Input height: {args.input_height}")
onnx_path = args.model_name.replace(".pt", ".onnx")
print(f"onnx_path is {onnx_path}")

# 使用解析的参数
model = YOLO(f"{args.model_name}")
model.export(
    format="onnx",
    imgsz=[args.input_height, args.input_width],
    optimize=args.optimize_cpu,
)
onnx.save(
    onnx.shape_inference.infer_shapes(onnx.load(onnx_path)),
    onnx_path.replace(".onnx", "_new.onnx"),
)
