import argparse
from io import BytesIO

import onnx
import torch
from torch import nn
from ultralytics import YOLO

try:
    import onnxsim
except ImportError:
    onnxsim = None


class PostSeg(nn.Module):
    export = True
    shape = None
    dynamic = False

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size
        mc = [self.cv4[i](x[i]) for i in range(self.nl)]
        x = self.forward_det(x)
        bo = len(x) // 3
        relocated = []
        for i in range(len(mc)):
            relocated.extend(x[i * bo : (i + 1) * bo])
            relocated.extend([mc[i]])
        relocated.extend([p])
        return relocated

    def forward_det(self, x):
        shape = x[0].shape
        y = []
        for i in range(self.nl):
            y.append(self.cv2[i](x[i]))
            y.append(self.cv3[i](x[i]))
        return y

        def optim(module: nn.Module):
            s = str(type(module))[6:-2].split(".")[-1]
            print(str(type(module))[6:-2])
            if s == "Segment":
                setattr(module, "__class__", PostSeg)


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
    YOLOv8 = YOLO(args.weights)
    model = YOLOv8.model.fuse().eval()
    for m in model.modules():
        optim(m)
        m.to(args.device)
    model.to(args.device)
    fake_input = torch.randn(args.input_shape).to(args.device)
    for _ in range(2):
        model(fake_input)
    save_path = args.weights.replace(".pt", ".onnx")
    output_names = [
        "output0",
        "output1",
        "output2",
        "output3",
        "output4",
        "output5",
        "output6",
        "output7",
        "output8",
        "proto",
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
