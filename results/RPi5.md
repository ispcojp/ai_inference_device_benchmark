# Raspberry Pi 5

## 構成

<https://www.raspberrypi.com/products/raspberry-pi-5/>

- CPU: Broadcom BCM2712 2.4GHz quad-core 64-bit Arm Cortex-A76 CPU
- GPU: VideoCore VII GPU
- OS: Linux

## Result

| model                 | CPU Torch | ORT CPU |
| --------------------- | --------: | ------: |
| efficientnet_b0       |   144.172 |  63.013 |
| ese_vovnet19b_dw      |   162.753 |  63.804 |
| mixnet_s              |   128.724 |  66.758 |
| mnasnet_a1            |   107.078 |  34.858 |
| mobilenetv2_050       |    54.870 |  13.970 |
| mobilenetv2_100       |   107.574 |  30.221 |
| mobilenetv2_110d      |   135.057 |  44.584 |
| mobilenetv3_large_100 |    83.445 |  29.030 |
| mobilenetv3_small_050 |    23.906 |   7.580 |
| repvgg_b0             |   249.838 | 108.095 |
| resnet18              |   136.053 |  50.285 |
| shufflenet_v2_x1_0    |    43.186 |   9.432 |
| vgg13                 |  1502.155 | 421.593 |

## CPU Torch

```text
efficientnet_b0               :    144.172 [ms] in average
ese_vovnet19b_dw              :    162.753 [ms] in average
mixnet_s                      :    128.724 [ms] in average
mnasnet_a1                    :    107.078 [ms] in average
mobilenetv2_050               :     54.870 [ms] in average
mobilenetv2_100               :    107.574 [ms] in average
mobilenetv2_110d              :    135.057 [ms] in average
mobilenetv3_large_100         :     83.445 [ms] in average
mobilenetv3_small_050         :     23.906 [ms] in average
repvgg_b0                     :    249.838 [ms] in average
resnet18                      :    136.053 [ms] in average
shufflenet_v2_x1_0            :     43.186 [ms] in average
vgg13                         :   1502.155 [ms] in average
```

## CPU ONNX

```text
efficientnet_b0               :     63.013 [ms] in average
ese_vovnet19b_dw              :     63.804 [ms] in average
mixnet_s                      :     66.758 [ms] in average
mnasnet_a1                    :     34.858 [ms] in average
mobilenetv2_050               :     13.970 [ms] in average
mobilenetv2_100               :     30.221 [ms] in average
mobilenetv2_110d              :     44.584 [ms] in average
mobilenetv3_large_100         :     29.030 [ms] in average
mobilenetv3_small_050         :      7.580 [ms] in average
repvgg_b0                     :    108.095 [ms] in average
resnet18                      :     50.285 [ms] in average
shufflenet_v2_x1_0            :      9.432 [ms] in average
vgg13                         :    421.593 [ms] in average
```
