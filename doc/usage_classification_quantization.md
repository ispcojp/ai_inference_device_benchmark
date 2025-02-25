# Usage: classification + quantization

## はじめに

- ONNX Runtime および OpenVINO の量子化機能を使います

## キャリブレーション用データの準備

[classification](./doc/usage_classification.md) に従い、あらかじめ ONNX モデルを作成しておいてください。

量子化のキャリブレーションおよび精度確認の為 Imagenette データセットをダウンロードしておいてください。
360 MB 程度の容量が必要です。
エッジデバイスなどディスク容量が少ないデバイスで計測する場合はあらかじめ PC で量子化して ONNX ファイルをコピーするのでも大丈夫です。

```bash
# デバイス容量の節約の為 320x320 で
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar zxf imagenette2-320.tgz
```

## ONNX Runtime による量子化と精度の確認

### ONNX Runtime による量子化手順

onnx-224-quant に量子化後の ONNX ファイルを配置します。

```bash
mkdir -p onnx-224-prep onnx-224-quant

# preprocess
for i in onnx-224/*.onnx; do
   echo $i
   python -m onnxruntime.quantization.preprocess \
       --input  $i \
       --output onnx-224-prep/$(basename $i)
done

# quantize
python classification/quantize.py onnx-224-prep onnx-224-quant
```

速度計測は ONXX ファイルを配置したディレクトリを指定して `benchmark_ort.py` を実行して下さい。

```
# benchmark quantized model
python3 classification/benchmark_ort.py onnx-224-quant | grep average
```

### ONNX Runtime による量子化後の精度

精度を計測するには benchmark_ort.py 実行時に `--verify-imagenette` を追加してください。

```bash
# FP32 モデルの精度確認
python3 classification/benchmark_ort.py onnx-224 --verify-imagenette

# 量子化したモデルの精度確認
python3 classification/benchmark_ort.py onnx-224-quant --verify-imagenette
```

2024/12/23 実施時の imagenette の精度は以下の通りでした (CPUExecutionProvider).

| model                 | FP32 ORT AP | INT8 ORT AP | 量子化による精度劣化 |
| --------------------- | ----------: | ----------: | -------------------- |
| efficientnet_b0       |      0.7678 |      0.0000 | 甚大                 |
| ese_vovnet19b_dw      |      0.7671 |      0.7393 |                      |
| mixnet_s              |      0.7607 |      0.5543 | 大                   |
| mnasnet_a1            |      0.7579 |      0.7533 |                      |
| mobilenetv2_050       |      0.6652 |      0.0127 | 甚大                 |
| mobilenetv2_100       |      0.7118 |      0.6950 |                      |
| mobilenetv2_110d      |      0.7294 |      0.7064 |                      |
| mobilenetv3_large_100 |      0.7676 |      0.1245 | 甚大                 |
| mobilenetv3_small_050 |      0.6038 |      0.0000 | 甚大                 |
| repvgg_b0             |      0.7312 |      0.0099 | 甚大                 |
| resnet18              |      0.6970 |      0.6608 |                      |
| shufflenet_v2_x1_0    |      0.6996 |      0.6163 | 中                   |
| vgg13                 |      0.7136 |      0.7189 |                      |

## OpenVINO による量子化と精度の確認

### OpenVINO による量子化手順

下記コマンドにより onnx-224-quant-openvino に量子化後の ONNX ファイルが配置されます。

```bash
# quantize
python classification/quantize_openvino.py onnx-224 onnx-224-quant-openvino
```

### OpenVINO による量子化後の精度

精度を計測するに `benchmark_openvino.py` に `--verify-imagenette` オプションを追加してください。
さらに `--comparison-model-dir <FP32_ONNX_directory>` オプションにより FP32 ONNX ファイルを配置したディレクトリを指定することで量子化前後の推論ラベルの一致率を計測できます。

```bash
# quantize
python classification/quantize_openvino.py onnx-224 onnx-224-quant-openvino

# benchmark quantized model
python3 classification/benchmark_openvino.py onnx-224-quant-openvino --mode npu --verify-imagenette --comparison-model-dir onnx-224
```

Intel CoreUltra7 265KF NPU 利用時の量子化の精度は [../results/Intel_CoreUltra7_265KF.md](../results/Intel_CoreUltra7_265KF.md) 参照。