import rk_head
from io import BytesIO
import onnx
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import head
import argparse

try:
    import onnxsim
except ImportError:
    onnxsim = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--weights", type=str, required=True, help="PyTorch yolov8 weights"
    )
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--sim", action="store_true", help="simplify onnx model")
    parser.add_argument(
        "--input-shape",
        nargs="+",
        type=int,
        default=[1, 3, 640, 640],
        help="Model input shape only for api builder",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Export ONNX device")
    args = parser.parse_args()
    assert len(args.input_shape) == 4
    return args


def main(args):
    setattr(head.Detect, "forward", rk_head.detect_forward)
    YOLOv8 = YOLO(args.weights)
    model = YOLOv8.model.fuse().eval()
    model.to(args.device)
    fake_input = torch.randn(args.input_shape).to(args.device)
    for _ in range(2):
        model(fake_input)
    save_path = args.weights.replace(".pt", ".onnx")
    output_names = [
        "yolo11_detect_output0_box",
        "yolo11_detect_output0_class",
        "yolo11_detect_output0_class_sum",
        "yolo11_detect_output1_box",
        "yolo11_detect_output1_class",
        "yolo11_detect_output1_class_sum",
        "yolo11_detect_output2_box",
        "yolo11_detect_output2_class",
        "yolo11_detect_output2_class_sum",
    ]
    with BytesIO() as f:
        torch.onnx.export(
            model,
            fake_input,
            f,
            opset_version=args.opset,
            input_names=["images"],
            output_names=output_names,
        )
        f.seek(0)
        onnx_model = onnx.load(f)
    onnx.checker.check_model(onnx_model)
    if args.sim:
        try:
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, "assert check failed"
        except Exception as e:
            print(f"Simplifier failure: {e}")
    onnx.save(onnx_model, save_path)
    print(f"ONNX export success, saved as {save_path}")


if __name__ == "__main__":
    main(parse_args())
