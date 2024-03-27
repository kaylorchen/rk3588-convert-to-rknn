import os
import sys
import numpy as np
from rknn.api import RKNN

DATASET_PATH = "./COCO/coco_subset_20.txt"
DEFAULT_RKNN_PATH = "./output.rknn"
DEFAULT_QUANT = True

import subprocess


def get_first_adb_device():
    # 执行 adb devices 命令并捕获输出
    result = subprocess.run(
        ["adb", "devices"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    output = result.stdout

    # 按照换行符拆分输出结果
    lines = output.strip().split("\n")

    # 除去第一行（标题行），遍历后续行以查找第一个设备
    for line in lines[1:]:
        if line.strip():  # 确保不是空行
            parts = line.split()
            # 如果存在有效的行，并且至少有两部分（确保状态也存在）
            # 第二个元素通常是'device'（如果设备连接正常的话）
            if len(parts) >= 2 and parts[1] == "device":
                return parts[0]  # 返回设备序列号

    # 如果没有找到设备，返回一个空字符串或者None
    return None


def parse_arg():
    if len(sys.argv) < 3:
        print(
            "Usage: python3 {} onnx_model_path [platform] [dtype(optional)] [output_rknn_path(optional)]".format(
                sys.argv[0]
            )
        )
        print("       platform choose from [rk3562,rk3566,rk3568,rk3588]")
        print("       dtype choose from    [i8, fp]")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ["i8", "fp"]:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)
        elif model_type == "i8":
            do_quant = True
        else:
            do_quant = False

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = DEFAULT_RKNN_PATH

    return model_path, platform, do_quant, output_path


if __name__ == "__main__":
    model_path, platform, do_quant, output_path = parse_arg()

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print("--> Config model")
    rknn.config(
        mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=platform
    )
    print("done")

    # Load model
    print("--> Loading model")
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print("Load model failed!")
        exit(ret)
    print("done")

    # Build model
    print("--> Building model")
    ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)
    if ret != 0:
        print("Build model failed!")
        exit(ret)
    print("done")

    # Export rknn model
    print("--> Export rknn model")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print("Export rknn model failed!")
        exit(ret)
    print("done")

    device_serial = get_first_adb_device()
    if device_serial:
        print(f"The first device's serial number is: {device_serial}")
        print("--> Accuracy analysis")
        ret = rknn.accuracy_analysis(
            inputs=["./COCO/subset/000000005001.jpg"],
            target=platform,
            device_id="99ce62f3843fc38c",
        )
    else:
        print("No connected ADB devices found")
        print("--> Accuracy analysis")
        ret = rknn.accuracy_analysis(
            inputs=["./COCO/subset/000000005001.jpg"],
            target=platform,
        )

    # Release
    rknn.release()
