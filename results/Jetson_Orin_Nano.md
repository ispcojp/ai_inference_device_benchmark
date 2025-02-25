# Jetson Orin Nano

## 構成

<https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/>

- CPU: 6-core Arm® Cortex®-A78AE v8.2 64-bit CPU 1.5MB L2 + 4MB L3
- GPU: 	NVIDIA Ampere architecture with 1024 CUDA cores and 32 tensor cores
- OS: Linux

## 結果早見表

(224, 224) の推論時間。数字は [ms]

| model                 | torch CPU | ORT CPU |   | torch GPU | ORT GPU |   | ORT TensorRT |
| --------------------- | --------: | ------: | - | --------: | ------: | - | -----------: |
| efficientnet_b0       |   130.594 |  47.993 |   |    29.078 |  12.455 |   |        6.346 |
| ese_vovnet19b_dw      |   126.762 |  43.191 |   |    11.253 |  12.400 |   |        4.429 |
| mixnet_s              |   125.913 |  51.086 |   |    31.207 |  14.882 |   |        7.463 |
| mnasnet_a1            |    91.927 |  31.279 |   |    19.686 |   8.671 |   |        3.864 |
| mobilenetv2_050       |    68.557 |  12.926 |   |    16.208 |   6.084 |   |        2.323 |
| mobilenetv2_100       |    87.977 |  26.283 |   |    16.234 |   7.216 |   |        3.098 |
| mobilenetv2_110d      |   112.539 |  35.295 |   |    20.462 |   8.714 |   |        4.140 |
| mobilenetv3_large_100 |    79.734 |  26.290 |   |    18.311 |   8.332 |   |        3.888 |
| mobilenetv3_small_050 |    30.481 |   7.092 |   |    15.041 |   6.096 |   |        2.481 |
| repvgg_b0             |   202.678 |  83.378 |   |    23.354 |  14.744 |   |        7.968 |
| resnet18              |    95.595 |  40.043 |   |     6.815 |   9.806 |   |        3.580 |
| shufflenet_v2_x1_0    |    52.481 |  10.497 |   |    17.478 |   7.940 |   |        3.099 |
| vgg13                 |   519.255 | 262.275 |   |     3.540 |  69.167 |   |       18.442 |

## install

torch.
<https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html>

```bash
$ apt-cache show nvidia-jetpack
Package: nvidia-jetpack
Version: 5.1.1-b56
(snip)

$ python --version  # 3.8.10

wget https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
pip install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl

pip install torchvision==0.15.0
```

ORT
<https://elinux.org/Jetson_Zoo#ONNX_Runtime>

```bash
wget https://nvidia.box.com/shared/static/mvdcltm9ewdy2d5nurkiqorofz1s53ww.whl -O onnxruntime_gpu-1.15.1-cp38-cp38-linux_aarch64.whl
pip install onnxruntime_gpu-1.15.1-cp38-cp38-linux_aarch64.whl
```

## torch

### CPU

```text
$ python3 classification/benchmark_torch.py | grep average

efficientnet_b0               :    130.594 [ms] in average
ese_vovnet19b_dw              :    126.762 [ms] in average
mixnet_s                      :    125.913 [ms] in average
mnasnet_a1                    :     91.927 [ms] in average
mobilenetv2_050               :     68.557 [ms] in average
mobilenetv2_100               :     87.977 [ms] in average
mobilenetv2_110d              :    112.539 [ms] in average
mobilenetv3_large_100         :     79.734 [ms] in average
mobilenetv3_small_050         :     30.481 [ms] in average
repvgg_b0                     :    202.678 [ms] in average
resnet18                      :     95.595 [ms] in average
shufflenet_v2_x1_0            :     52.481 [ms] in average
vgg13                         :    519.255 [ms] in average
```

### GPU

```text
$ python3 classification/benchmark_torch.py --mode gpu | grep average
efficientnet_b0               :     29.078 [ms] in average
ese_vovnet19b_dw              :     11.253 [ms] in average
mixnet_s                      :     31.207 [ms] in average
mnasnet_a1                    :     19.686 [ms] in average
mobilenetv2_050               :     16.208 [ms] in average
mobilenetv2_100               :     16.234 [ms] in average
mobilenetv2_110d              :     20.462 [ms] in average
mobilenetv3_large_100         :     18.311 [ms] in average
mobilenetv3_small_050         :     15.041 [ms] in average
repvgg_b0                     :     23.354 [ms] in average
resnet18                      :      6.815 [ms] in average
shufflenet_v2_x1_0            :     17.478 [ms] in average
vgg13                         :      3.540 [ms] in average
```

## ORT

### CPU (ORT)

```text
$ python3 classification/benchmark_ort.py onnx-224 | grep average
efficientnet_b0               :     47.993 [ms] in average
ese_vovnet19b_dw              :     43.191 [ms] in average
mixnet_s                      :     51.086 [ms] in average
mnasnet_a1                    :     31.279 [ms] in average
mobilenetv2_050               :     12.926 [ms] in average
mobilenetv2_100               :     26.283 [ms] in average
mobilenetv2_110d              :     35.295 [ms] in average
mobilenetv3_large_100         :     26.290 [ms] in average
mobilenetv3_small_050         :      7.092 [ms] in average
repvgg_b0                     :     83.378 [ms] in average
resnet18                      :     40.043 [ms] in average
shufflenet_v2_x1_0            :     10.497 [ms] in average
vgg13                         :    262.275 [ms] in average
```

### GPU (ORT)

```text
$ python3 classification/benchmark_ort.py onnx-224 --mode gpu | grep average
efficientnet_b0               :     12.455 [ms] in average
ese_vovnet19b_dw              :     12.400 [ms] in average
mixnet_s                      :     14.882 [ms] in average
mnasnet_a1                    :      8.671 [ms] in average
mobilenetv2_050               :      6.084 [ms] in average
mobilenetv2_100               :      7.216 [ms] in average
mobilenetv2_110d              :      8.714 [ms] in average
mobilenetv3_large_100         :      8.332 [ms] in average
mobilenetv3_small_050         :      6.096 [ms] in average
repvgg_b0                     :     14.744 [ms] in average
resnet18                      :      9.806 [ms] in average
shufflenet_v2_x1_0            :      7.940 [ms] in average
vgg13                         :     69.167 [ms] in average
```

### TensorRT (ORT)

```text
$ python3 classification/benchmark_ort.py onnx-224 --mode tensorrt | grep average
efficientnet_b0               :      6.346 [ms] in average
ese_vovnet19b_dw              :      4.429 [ms] in average
mixnet_s                      :      7.463 [ms] in average
mnasnet_a1                    :      3.864 [ms] in average
mobilenetv2_050               :      2.323 [ms] in average
mobilenetv2_100               :      3.098 [ms] in average
mobilenetv2_110d              :      4.140 [ms] in average
mobilenetv3_large_100         :      3.888 [ms] in average
mobilenetv3_small_050         :      2.481 [ms] in average
repvgg_b0                     :      7.968 [ms] in average
resnet18                      :      3.580 [ms] in average
shufflenet_v2_x1_0            :      3.099 [ms] in average
vgg13                         :     18.442 [ms] in average
```
