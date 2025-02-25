## 構成

- CPU: Intel(R) Core(TM) i5-5200U CPU @ 2.20GHz
- GPU: なし
- OS: Linux

## 結果

(224, 224) の推論時間。単位は [ms]

| model                 | Torch CPU | ORT CPU |
| --------------------- | --------: | ------: |
| efficientnet_b0       |   120.512 |  34.642 |
| ese_vovnet19b_dw      |   126.616 |  49.043 |
| mixnet_s              |    96.209 |  45.049 |
| mnasnet_a1            |    65.216 |  17.303 |
| mobilenetv2_050       |    35.525 |   6.919 |
| mobilenetv2_100       |    62.089 |  16.557 |
| mobilenetv2_110d      |    80.935 |  18.203 |
| mobilenetv3_large_100 |    40.677 |  14.239 |
| mobilenetv3_small_050 |    16.464 |   3.226 |
| repvgg_b0             |   176.104 |  73.577 |
| resnet18              |    99.518 |  41.907 |
| shufflenet_v2_x1_0    |    35.124 |  11.224 |
| vgg13                 |   692.254 | 368.626 |

## Torch CPU

```text
$ python3 classification/benchmark_torch.py | grep average
efficientnet_b0               :    120.512 [ms] in average
ese_vovnet19b_dw              :    126.616 [ms] in average
mixnet_s                      :     96.209 [ms] in average
mnasnet_a1                    :     65.216 [ms] in average
mobilenetv2_050               :     35.525 [ms] in average
mobilenetv2_100               :     62.089 [ms] in average
mobilenetv2_110d              :     80.935 [ms] in average
mobilenetv3_large_100         :     40.677 [ms] in average
mobilenetv3_small_050         :     16.464 [ms] in average
repvgg_b0                     :    176.104 [ms] in average
resnet18                      :     99.518 [ms] in average
shufflenet_v2_x1_0            :     35.124 [ms] in average
vgg13                         :    692.254 [ms] in average
```

## ORT CPU

```text
$ python3 classification/benchmark_ort.py onnx-224 | grep average
efficientnet_b0               :     34.642 [ms] in average
ese_vovnet19b_dw              :     49.043 [ms] in average
mixnet_s                      :     45.049 [ms] in average
mnasnet_a1                    :     17.303 [ms] in average
mobilenetv2_050               :      6.919 [ms] in average
mobilenetv2_100               :     16.557 [ms] in average
mobilenetv2_110d              :     18.203 [ms] in average
mobilenetv3_large_100         :     14.239 [ms] in average
mobilenetv3_small_050         :      3.226 [ms] in average
repvgg_b0                     :     73.577 [ms] in average
resnet18                      :     41.907 [ms] in average
shufflenet_v2_x1_0            :     11.224 [ms] in average
vgg13                         :    368.626 [ms] in average
```
