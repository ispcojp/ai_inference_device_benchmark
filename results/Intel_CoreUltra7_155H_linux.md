## 構成 

- CPU: intel core ultra7 155H (1.4-4.8 GHz)
- GPU: intel HD graphics (intel Arc Graphics)
- OS: Linux

## benchmark_torch.py

```text
efficientnet_b0               :     24.347 [ms] in average
ese_vovnet19b_dw              :     34.093 [ms] in average
mixnet_s                      :     32.188 [ms] in average
mnasnet_a1                    :     23.677 [ms] in average
mobilenetv2_050               :      6.124 [ms] in average
mobilenetv2_100               :     12.117 [ms] in average
mobilenetv2_110d              :     16.386 [ms] in average
mobilenetv3_large_100         :     28.203 [ms] in average
mobilenetv3_small_050         :     10.271 [ms] in average
repvgg_b0                     :     57.567 [ms] in average
resnet18                      :     27.257 [ms] in average
shufflenet_v2_x1_0            :     13.683 [ms] in average
vgg13                         :    134.243 [ms] in average
```

## classification/benchmark_ort.py

```text
efficientnet_b0               :      9.074 [ms] in average
ese_vovnet19b_dw              :     10.980 [ms] in average
mixnet_s                      :     14.850 [ms] in average
mnasnet_a1                    :      3.849 [ms] in average
mobilenetv2_050               :      1.428 [ms] in average
mobilenetv2_100               :      3.095 [ms] in average
mobilenetv2_110d              :      4.567 [ms] in average
mobilenetv3_large_100         :      4.719 [ms] in average
mobilenetv3_small_050         :      1.888 [ms] in average
repvgg_b0                     :     25.707 [ms] in average
resnet18                      :     12.317 [ms] in average
shufflenet_v2_x1_0            :      3.871 [ms] in average
vgg13                         :     66.228 [ms] in average
```

## OpenVINO CPU FP16 12 thread

```text
efficientnet_b0               :      6.127 [ms] in average
ese_vovnet19b_dw              :      9.227 [ms] in average
mixnet_s                      :      5.314 [ms] in average
mnasnet_a1                    :      2.992 [ms] in average
mobilenetv2_050               :      1.326 [ms] in average
mobilenetv2_100               :      2.525 [ms] in average
mobilenetv2_110d              :      3.932 [ms] in average
mobilenetv3_large_100         :     15.773 [ms] in average
mobilenetv3_small_050         :      4.231 [ms] in average
repvgg_b0                     :     18.928 [ms] in average
resnet18                      :     10.146 [ms] in average
shufflenet_v2_x1_0            :      2.409 [ms] in average
vgg13                         :     62.004 [ms] in average
```

## OpenVINO iGPU FP16 12 thread

```text
efficientnet_b0               :      4.161 [ms] in average
ese_vovnet19b_dw              :      2.994 [ms] in average
mixnet_s                      :      6.591 [ms] in average
mnasnet_a1                    :      2.686 [ms] in average
mobilenetv2_050               :      2.060 [ms] in average
mobilenetv2_100               :      2.236 [ms] in average
mobilenetv2_110d              :      3.127 [ms] in average
mobilenetv3_large_100         :     16.194 [ms] in average
mobilenetv3_small_050         :      3.398 [ms] in average
repvgg_b0                     :      6.546 [ms] in average
resnet18                      :      2.293 [ms] in average
shufflenet_v2_x1_0            :      2.033 [ms] in average
vgg13                         :      8.607 [ms] in average
```

## OpenVINO NPU FP16 12 thread

```text
efficientnet_b0               :      5.286 [ms] in average
ese_vovnet19b_dw              :      3.807 [ms] in average
mixnet_s                      :      4.715 [ms] in average
mnasnet_a1                    :      2.485 [ms] in average
mobilenetv2_050               :      1.503 [ms] in average
mobilenetv2_100               :      1.963 [ms] in average
mobilenetv2_110d              :      2.359 [ms] in average
mobilenetv3_large_100         :     16.186 [ms] in average
mobilenetv3_small_050         :      4.236 [ms] in average
repvgg_b0                     :      5.214 [ms] in average
resnet18                      :      2.931 [ms] in average
shufflenet_v2_x1_0            :      2.405 [ms] in average
vgg13                         :     17.068 [ms] in average
```
