import argparse
from pathlib import Path
import time
import os
import numpy as np
import openvino as ov
from sklearn.metrics import accuracy_score
from torchvision.datasets import ImageFolder
from models import get_transform
from data_loader import target_transform_imagenette_to_imagenet
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="cpu",
                        help="cpu or gpu or npu")
    parser.add_argument("model_dir", help="model directories")
    parser.add_argument("--times", default=10, type=int,
                        help="how many times to infer")
    parser.add_argument("--verify-imagenette", action="store_true",
                        help="calculate imagenette accuracy")
    parser.add_argument("--comparison-model-dir", default=None,
                        help="valid only verify-imagenette=True.")
    args = parser.parse_args()
    return args
# list of (data, cls)
g_imagenette = []
def verify_imagenette(model_name: str, compiled_model, comparison_model,
                      imagenette_dir="imagenette2-320/val",):
    """
    Calculate imagenette accuracy.
    """
    global g_imagenette
    if not g_imagenette:
        print("load imagenette")
        dataset = ImageFolder(
            imagenette_dir,
            transform=None,  # None to support model specific transform
            target_transform=target_transform_imagenette_to_imagenet)
        for data, cls in dataset:
            g_imagenette.append((data, cls))
        print("load imagenette done")
    dataset = g_imagenette
    gt = []
    pred = []
    transform = get_transform(model_name)
    compare_true = 0
    compare_false = 0
    for i, (data, cls) in enumerate(dataset):
        if i % 1000 == 0:
            print(i)
        x = transform(data).numpy()[np.newaxis, ]
        out = compiled_model([x])
        gt.append(cls)
        output_label = out[0].argmax()
        pred.append(output_label)
        if comparison_model:
            comparison_out = comparison_model([x])
            compare_output_label = comparison_out[0].argmax()
            if output_label == compare_output_label:
                compare_true += 1
            else:
                compare_false += 1
    ap = accuracy_score(gt, pred)
    print(f"{model_name}: {ap=}")
    if comparison_model:
        rate = compare_true / (compare_true + compare_false)
        print(f"The agreement rate of {model_name}'s \
            output with respect to compare model's output: {rate}")

def compile_model(onnx_model_path, device):
    core = ov.Core()
    ir_model = ov.convert_model(onnx_model_path)
    compiled_model = core.compile_model(ir_model, device)
    return compiled_model
def benchmark(args, model_path):
    """
    model_path [Path] onnx model path
    """
    print(f"benchmark {str(model_path)}")
    model_name = model_path.stem
    file_size_bytes = os.path.getsize(model_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"Model size: {model_path.stem} {file_size_mb:.2f} MB")
    device = args.mode.upper()
    try:
        compiled_model = compile_model(model_path, device)
    except:
        print(f"failed to compile {model_name} for device '{device}'")
        return
    input_shape = tuple(compiled_model.inputs[0].shape)
    print(f"{input_shape=}")
    dummy_input = np.random.random(input_shape).astype(np.float32)
    total_ms = 0.0
    for i in range(args.times + 10):
        st = time.time()
        _ = compiled_model([dummy_input])
        et = time.time()
        elasped_ms = (et - st) * 1000
        print(f'{elasped_ms} [ms]')
        if i >= 10:
            total_ms += elasped_ms
    avg_ms = total_ms / args.times
    print(f"{model_name:30}: {avg_ms:>10.3f} [ms] in average")
    if args.verify_imagenette:
        comparison_model = None
        if args.comparison_model_dir:
            comparison_model_file = os.path.join(args.comparison_model_dir, f"{model_name}.onnx")
            if os.path.isfile(comparison_model_file):
                print(f"{comparison_model_file} is exist")
                comparison_model = compile_model(comparison_model_file, device)
            else:
                print(f"{comparison_model_file} is not exist")
        verify_imagenette(model_name,
                          compiled_model,
                          comparison_model)
def main():
    args = parse_args()
    device = args.mode
    core = ov.Core()
    available_device = core.available_devices
    if not device.upper() in available_device:
        print(f"{device} not supported.")
        print(f"supported device is {available_device}.")
        return

    print(f"{device} mode")
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"{model_dir} not exists.")
        return
    for m in sorted(model_dir.glob("*.onnx")):
        benchmark(args, m)
if __name__ == '__main__':
    main()
