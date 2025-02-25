## 構成

- CPU: Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz
- GPU: NVIDIA GeForce RTX 3080
- OS: Linux

## 早見表

| model                 | Torch CPU | Torch GPU | ORT CPU | ORT GPU |
| --------------------- | --------: | --------: | ------: | ------: |
| efficientnet_b0       |    18.448 |     6.090 |   7.372 |   3.216 |
| ese_vovnet19b_dw      |    16.909 |     2.663 |   5.433 |   5.353 |
| mixnet_s              |    18.994 |     7.045 |  12.004 |   4.238 |
| mnasnet_a1            |    11.111 |     4.454 |   2.399 |   2.201 |
| mobilenetv2_050       |     7.565 |     3.455 |   0.997 |   1.677 |
| mobilenetv2_100       |     9.835 |     3.686 |   1.842 |   2.129 |
| mobilenetv2_110d      |    12.808 |     4.800 |   2.625 |   2.154 |
| mobilenetv3_large_100 |    10.850 |     4.301 |   3.727 |   2.166 |
| mobilenetv3_small_050 |     4.929 |     3.140 |   1.588 |   1.911 |
| repvgg_b0             |    18.169 |     5.428 |  10.298 |   3.207 |
| resnet18              |     8.206 |     1.361 |   5.935 |   2.141 |
| shufflenet_v2_x1_0    |     9.126 |     4.251 |   3.112 |   2.166 |
| vgg13                 |    86.552 |     1.266 |  39.957 |   5.399 |

## PyTorch

### CPU

```text
$ python3 classification/benchmark_torch.py | grep average
efficientnet_b0               :     18.448 [ms] in average
ese_vovnet19b_dw              :     16.909 [ms] in average
mixnet_s                      :     18.994 [ms] in average
mnasnet_a1                    :     11.111 [ms] in average
mobilenetv2_050               :      7.565 [ms] in average
mobilenetv2_100               :      9.835 [ms] in average
mobilenetv2_110d              :     12.808 [ms] in average
mobilenetv3_large_100         :     10.850 [ms] in average
mobilenetv3_small_050         :      4.929 [ms] in average
repvgg_b0                     :     18.169 [ms] in average
resnet18                      :      8.206 [ms] in average
shufflenet_v2_x1_0            :      9.126 [ms] in average
vgg13                         :     86.552 [ms] in average
```

### GPU

```text
$ python3 classification/benchmark_torch.py --mode gpu | grep average
efficientnet_b0               :      6.090 [ms] in average
ese_vovnet19b_dw              :      2.663 [ms] in average
mixnet_s                      :      7.045 [ms] in average
mnasnet_a1                    :      4.454 [ms] in average
mobilenetv2_050               :      3.455 [ms] in average
mobilenetv2_100               :      3.686 [ms] in average
mobilenetv2_110d              :      4.800 [ms] in average
mobilenetv3_large_100         :      4.301 [ms] in average
mobilenetv3_small_050         :      3.140 [ms] in average
repvgg_b0                     :      5.428 [ms] in average
resnet18                      :      1.361 [ms] in average
shufflenet_v2_x1_0            :      4.251 [ms] in average
vgg13                         :      1.266 [ms] in average
```

## ORT

### CPU (ORT)

```text
$ python3 classification/benchmark_ort.py onnx-224 | grep average
efficientnet_b0               :      7.372 [ms] in average
ese_vovnet19b_dw              :      5.433 [ms] in average
mixnet_s                      :     12.004 [ms] in average
mnasnet_a1                    :      2.399 [ms] in average
mobilenetv2_050               :      0.997 [ms] in average
mobilenetv2_100               :      1.842 [ms] in average
mobilenetv2_110d              :      2.625 [ms] in average
mobilenetv3_large_100         :      3.727 [ms] in average
mobilenetv3_small_050         :      1.588 [ms] in average
repvgg_b0                     :     10.298 [ms] in average
resnet18                      :      5.935 [ms] in average
shufflenet_v2_x1_0            :      3.112 [ms] in average
vgg13                         :     39.957 [ms] in average
```

### GPU (ORT)

```text
$ python3 classification/benchmark_ort.py onnx-224 --mode gpu | grep average
efficientnet_b0               :      3.216 [ms] in average
ese_vovnet19b_dw              :      5.353 [ms] in average
mixnet_s                      :      4.238 [ms] in average
mnasnet_a1                    :      2.201 [ms] in average
mobilenetv2_050               :      1.677 [ms] in average
mobilenetv2_100               :      2.129 [ms] in average
mobilenetv2_110d              :      2.154 [ms] in average
mobilenetv3_large_100         :      2.166 [ms] in average
mobilenetv3_small_050         :      1.911 [ms] in average
repvgg_b0                     :      3.207 [ms] in average
resnet18                      :      2.141 [ms] in average
shufflenet_v2_x1_0            :      2.166 [ms] in average
vgg13                         :      5.399 [ms] in average
```

### GPU (TensorRT)

未実施
