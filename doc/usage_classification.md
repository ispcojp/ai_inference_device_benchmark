# Usage: classification

## 概要

- imagenet 学習済みモデルを実行します
- 画像サイズは 224x224
- 下記では PyTorch での計測および ONNX モデルをエクスポートします

## コマンド

```bash
# torch benchmark
python3 classification/benchmark_torch.py | grep average

# export onnx, we get onnx-224/*.onnx
python3 classification/export_onnx.py onnx

# ORT benchmark

## ORT CPU
python3 classification/benchmark_ort.py onnx-224 | grep average

## ORT NVIDIA GPU
python3 classification/benchmark_ort.py --mode gpu onnx-224 | grep average
python3 classification/benchmark_ort.py --mode tensorrt onnx-224 | grep average

## ORT OpenVINO
python3 classification/benchmark_ort.py --mode openvino-cpu-fp32-12 onnx-224 | grep average
python3 classification/benchmark_ort.py --mode openvino-gpu-fp16-12 onnx-224 | grep average
python3 classification/benchmark_ort.py --mode openvino-npu-fp16-12 onnx-224 | grep average

# OpenVINO benchmark
python3 classification/benchmark_openvino.py onnx-224 --mode npu | grep average
```

CPU/GPU の指定方法

- benchmark_torch.py
  - GPU 利用: `--mode gpu` を指定
- benchmark_ort.py
  - CUDA 利用: `--mode gpu` を指定
  - TensorRT 利用: `--mode をtensorrt` を指定
  - OpenVINO 利用: `--mode openvino-npu-fp16-12` などを指定。書式は `openvino-<device>-<bit>-<num_threads>`.
    詳しくは `--help` 参照
- benchmark_openvino.py
  - NPU 利用: `--mode npu` を指定
  - GPU (iGPU, dGPU) 利用: `--mode gpu` を指定
