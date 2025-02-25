#!/usr/bin/python3

import argparse
from pathlib import Path
import time

import numpy as np
import onnxruntime as rt
from sklearn.metrics import accuracy_score
from torchvision.datasets import ImageFolder

from models import get_transform
from data_loader import target_transform_imagenette_to_imagenet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="cpu",
                        help="cpu, gpu, openvino-{cpu,gpu,npu}-{fp32,fp16}-<n_thread>")
    parser.add_argument("model_dir", help="model directories")
    parser.add_argument("--times", default=10, type=int,
                        help="how many times to infer")
    parser.add_argument("--show-providers", action="store_true")
    parser.add_argument("--verify-imagenette", action="store_true",
                        help="calculate imagenette accuracy")
    args = parser.parse_args()
    return args


# list of (data, cls)
g_imagenette = []


def verify_imagenette(model_name: str, sess, imagenette_dir="imagenette2-320/val"):
    """
    Calculate imagenette accuracy.
    """
    input_names = [x.name for x in sess.get_inputs()]
    output_names = [x.name for x in sess.get_outputs()]
    _, h, w, _ = sess.get_inputs()[0].shape

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

    for i, (data, cls) in enumerate(dataset):
        if i % 1000 == 0:
            print(i)
        x = transform(data).numpy()[np.newaxis, ]
        out = sess.run(output_names,
                       {input_names[0]: x})[0]
        gt.append(cls)
        pred.append(out[0].argmax())

    ap = accuracy_score(gt, pred)
    print(f"{model_name}: {ap=}")


def benchmark(args, model_path):
    """
    model_path [Path] onnx model path
    """
    print(f"benchmark {str(model_path)}")

    providers = []
    options = rt.SessionOptions()
    provider_options = []

    if args.mode == "tensorrt":
        print("append TensorRT Provider")
        providers.append("TensorrtExecutionProvider")
    if args.mode in ("gpu", "tensorrt"):
        print("append CUDA Provider")
        providers.append("CUDAExecutionProvider")
    if "openvino" in args.mode:
        openvino_mode = args.mode.split("-")
        assert len(openvino_mode) == 4
        providers.append("OpenVINOExecutionProvider")
        options.graph_optimization_level = \
          rt.GraphOptimizationLevel.ORT_DISABLE_ALL
        provider_options.append({
            "device_type": openvino_mode[1].upper(),
            "precision": openvino_mode[2].upper(),
            "num_of_threads": int(openvino_mode[3]),
        })

    provider_options.append({})
    providers.append("CPUExecutionProvider")

    print(f"{providers=}")
    print(f"{provider_options=}")
    onnx_model = rt.InferenceSession(
        str(model_path),
        options,
        providers=providers,
        provider_options=provider_options)

    onnx_input = [x.name for x in onnx_model.get_inputs()]
    onnx_output = [x.name for x in onnx_model.get_outputs()]

    input_shape = onnx_model.get_inputs()[0].shape
    print(f"{input_shape=}")
    dummy_input = np.random.random(input_shape).astype(np.float32)

    # dry run
    _ = onnx_model.run(onnx_output,
                       {onnx_input[0]: dummy_input})

    total_ms = 0.0
    for _ in range(args.times):
        st = time.time()
        _ = onnx_model.run(onnx_output,
                           {onnx_input[0]: dummy_input})
        et = time.time()
        elasped_ms = (et - st) * 1000
        print(f'{elasped_ms} [ms]')
        total_ms += elasped_ms

    avg_ms = total_ms / args.times
    model_name = model_path.stem
    print(f"{model_name:30}: {avg_ms:>10.3f} [ms] in average")

    if args.verify_imagenette:
        verify_imagenette(model_path.stem,
                          onnx_model)


def main():
    args = parse_args()

    if args.show_providers:
        print(rt.get_available_providers())
        return

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"{model_dir} not exists")
        return

    for m in sorted(model_dir.glob("*.onnx")):
        benchmark(args, m)


if __name__ == '__main__':
    main()
