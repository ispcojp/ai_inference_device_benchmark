# Raspberry Pi4

## 構成

<https://www.raspberrypi.com/products/raspberry-pi-4-model-b/>

- CPU: Broadcom BCM2711, Quad core Cortex-A72 (ARM v8) 64-bit SoC @ 1.8GHz
- GPU:
- OS: Linux

## 早見表

(224, 224) の推論時間。単位は ms

| model                 | torch CPU |  ORT CPU |
| --------------------- | --------: | -------: |
| efficientnet_b0       |   517.608 |  200.081 |
| ese_vovnet19b_dw      |   343.219 |  249.765 |
| mixnet_s              |   448.068 |  194.713 |
| mnasnet_a1            |   385.575 |  113.957 |
| mobilenetv2_050       |   201.655 |   46.726 |
| mobilenetv2_100       |   380.062 |  103.894 |
| mobilenetv2_110d      |   526.031 |  146.234 |
| mobilenetv3_large_100 |   295.356 |   97.349 |
| mobilenetv3_small_050 |    81.429 |   20.494 |
| repvgg_b0             |   485.943 |  542.541 |
| resnet18              |   263.473 |  274.989 |
| shufflenet_v2_x1_0    |   141.679 |   40.200 |
| vgg13                 |  1782.008 | 1738.197 |

## Torch CPU

```bash
$ python3 classification/benchmark_torch.py  | grep avera
efficientnet_b0               :    517.608 [ms] in average
ese_vovnet19b_dw              :    343.219 [ms] in average
mixnet_s                      :    448.068 [ms] in average
mnasnet_a1                    :    385.575 [ms] in average
mobilenetv2_050               :    201.655 [ms] in average
mobilenetv2_100               :    380.062 [ms] in average
mobilenetv2_110d              :    526.031 [ms] in average
mobilenetv3_large_100         :    295.356 [ms] in average
mobilenetv3_small_050         :     81.429 [ms] in average
repvgg_b0                     :    485.943 [ms] in average
resnet18                      :    263.473 [ms] in average
shufflenet_v2_x1_0            :    141.679 [ms] in average
vgg13                         :   1782.008 [ms] in average
```

## ORT CPU

```text
$ python3 classification/benchmark_ort.py onnx-224 | grep average
efficientnet_b0               :    200.081 [ms] in average
ese_vovnet19b_dw              :    249.765 [ms] in average
mixnet_s                      :    194.713 [ms] in average
mnasnet_a1                    :    113.957 [ms] in average
mobilenetv2_050               :     46.726 [ms] in average
mobilenetv2_100               :    103.894 [ms] in average
mobilenetv2_110d              :    146.234 [ms] in average
mobilenetv3_large_100         :     97.349 [ms] in average
mobilenetv3_small_050         :     20.494 [ms] in average
repvgg_b0                     :    542.541 [ms] in average
resnet18                      :    274.989 [ms] in average
shufflenet_v2_x1_0            :     40.200 [ms] in average
vgg13                         :   1738.197 [ms] in average
```
