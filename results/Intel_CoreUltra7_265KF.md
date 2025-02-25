## 構成

- CPU: intel core ultra7 265KF (3.3 GHz - 5.5GHz)
- GPU: RTX4090
- OS: Linux

## 実行コマンド

- NPU

```
python3 classification/benchmark_openvino.py onnx-224 --mode npu |grep average
```

- CPU

```
python3 classification/benchmark_openvino.py onnx-224 --mode cpu |grep average
```

## 結果

単位は [ms]

| model                 | OpenVINO NPU(fp32) | OpenVINO CPU(fp32) | Torch GPU | Torch CPU | ORT CPU | ORT GPU |
| --------------------- | -----------------: | -----------------: | --------: | --------: | ------: | ------: |
| efficientnet_b0       |              3.882 |              2.338 |     1.759 |   107.296 |   3.572 |   1.125 |
| ese_vovnet19b_dw      |              2.172 |              4.177 |     0.843 |    10.096 |   2.883 |   0.654 |
| mixnet_s              |              2.270 |              1.977 |     2.145 |    87.107 |   5.875 |   1.481 |
| mnasnet_a1            |              1.239 |              1.196 |     1.403 |    34.341 |   1.602 |   0.719 |
| mobilenetv2_050       |              0.855 |              0.508 |     1.103 |     3.456 |   0.795 |   0.523 |
| mobilenetv2_100       |              1.082 |              1.145 |     1.168 |    10.125 |   1.053 |   0.545 |
| mobilenetv2_110d      |              1.469 |              1.713 |     1.492 |     8.805 |   1.532 |   0.746 |
| mobilenetv3_large_100 |              1.880 |              1.800 |     1.358 |     6.186 |   3.327 |   0.693 |
| mobilenetv3_small_050 |              1.100 |              0.410 |     1.064 |    10.353 |   1.656 |   0.626 |
| repvgg_b0             |              3.107 |              7.223 |     2.159 |    12.905 |   5.242 |   1.200 |
| resnet18              |              1.509 |              4.796 |     1.174 |     5.923 |   2.797 |   0.537 |
| shufflenet_v2_x1_0    |              1.309 |              1.115 |     1.316 |     4.713 |   3.130 |   0.610 |
| vgg13                 |             11.961 |             29.189 |     3.047 |   100.642 |  20.248 |   2.475 |


## モデルサイズ

```
Model size: 20.06 MB
Model size: 24.96 MB
Model size: 15.76 MB
Model size: 14.74 MB
Model size: 7.49 MB
Model size: 13.29 MB
Model size: 17.11 MB
Model size: 20.85 MB
Model size: 6.08 MB
Model size: 60.34 MB
Model size: 32.62 MB
Model size: 8.70 MB
Model size: 507.52 MB
```

### 量子化実行結果

量子化前と量子化後の推論速度 (NPU)。単位は [ms]

| Model                 |   FP32 |  INT8 |
| --------------------- | -----: | ----: |
| efficientnet_b0       |  4.738 | 4.456 |
| ese_vovnet19b_dw      |  3.190 | 1.675 |
| mixnet_s              |  3.533 |   N/A |
| mnasnet_a1            |  1.855 | 1.666 |
| mobilenetv2_050       |  1.183 | 1.104 |
| mobilenetv2_100       |  1.704 | 1.041 |
| mobilenetv2_110d      |  2.240 | 1.552 |
| mobilenetv3_large_100 |  2.784 | 2.377 |
| mobilenetv3_small_050 |  1.832 | 1.858 |
| repvgg_b0             |  4.870 | 2.984 |
| resnet18              |  2.139 | 1.770 |
| shufflenet_v2_x1_0    |  1.862 | 1.984 |
| vgg13                 | 11.947 | 6.769 |

量子化前と量子化後の imagenette AP (NPU). 単位は 100 倍すると %, つまり 0.768 なら 76.8%

| Model                 | FP32 AP | INT8 AP | 量子化前後のラベル一致度 |
| --------------------- | ------: | ------: | -----------------------: |
| efficientnet_b0       |   0.768 |   0.529 |                    0.607 |
| ese_vovnet19b_dw      |   0.767 |   0.755 |                    0.935 |
| mixnet_s              |   0.761 |     N/A |                      N/A |
| mnasnet_a1            |   0.758 |   0.749 |                    0.941 |
| mobilenetv2_050       |   0.665 |   0.653 |                    0.929 |
| mobilenetv2_100       |   0.712 |   0.708 |                    0.952 |
| mobilenetv2_110d      |   0.730 |   0.725 |                    0.958 |
| mobilenetv3_large_100 |   0.768 |   0.644 |                    0.778 |
| mobilenetv3_small_050 |   0.603 |   0.378 |                    0.498 |
| repvgg_b0             |   0.732 |   0.004 |                    0.004 |
| resnet18              |   0.697 |   0.688 |                    0.950 |
| shufflenet_v2_x1_0    |   0.700 |   0.689 |                    0.938 |
| vgg13                 |   0.714 |   0.713 |                    0.977 |

また、量子化オプションで `fast_bias_correction=False` を指定した場合 resnet などが途中で失敗した。
精度もほぼ変わらなかった。