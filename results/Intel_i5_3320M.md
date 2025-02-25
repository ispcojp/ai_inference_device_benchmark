## 構成

- CPU: Intel(R) Core(TM) i5-3320M CPU @ 2.60GHz
- GPU: なし
- OS: Linux

## 結果

(224, 224) の推論時間。単位は [ms]

| model                 | Torch CPU | ORT CPU | ORT CPU (int8) | OpenVINO (※1) |
| --------------------- | --------: | ------: | -------------: | -------------: |
| efficientnet_b0       |    85.499 |  27.335 |         51.709 |         29.561 |
| ese_vovnet19b_dw      |    74.479 |  41.205 |        101.422 |         69.546 |
| mixnet_s              |    75.231 |  30.410 |         51.722 |         29.696 |
| mnasnet_a1            |    39.880 |  11.253 |         19.481 |         20.280 |
| mobilenetv2_050       |    23.860 |   4.385 |          7.874 |          7.194 |
| mobilenetv2_100       |    42.667 |  10.779 |         17.753 |         15.641 |
| mobilenetv2_110d      |    55.541 |  15.291 |         24.242 |         20.941 |
| mobilenetv3_large_100 |    48.336 |  10.972 |         17.248 |         15.805 |
| mobilenetv3_small_050 |    11.932 |   2.641 |          4.739 |          3.904 |
| repvgg_b0             |   128.620 |  79.238 |        131.674 |         86.304 |
| resnet18              |    69.987 |  41.988 |         74.548 |         86.730 |
| shufflenet_v2_x1_0    |    25.405 |   7.040 |         10.015 |         11.262 |
| vgg13                 |   487.892 | 290.000 |        486.714 |        532.871 |

※1: CPU, FP32

## CPU spec

```
flags           : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm cpuid_fault epb pti ssbd ibrs ibpb stibp tpr_shadow flexpriority ept vpid fsgsbase smep erms xsaveopt dtherm ida arat pln pts vnmi md_clear flush_l1d
vmx flags       : vnmi preemption_timer invvpid ept_x_only flexpriority tsc_offset vtpr mtf vapic ept vpid unrestricted_guest
```

## OpenVINO

`get_available_device()` で CPU のみ
