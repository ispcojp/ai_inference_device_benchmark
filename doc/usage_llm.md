# Usage: llm

## 概要
このスクリプトはLLM系のモデルを実行した際（text to text又はimage&text to text）の評価ができます。

サポートしているミドルウエアは次の2つです。
- Ollama
- Open VINO

使用可能モデルに制約があります。詳細は各ミドルウエアの項目をご確認ください。

評価指標は以下の通りです。
- FTL(First Token Latency)
   - テキストを入力してから、最初のトークンが出るまでの時間
- TPS(Tokens per second)
   - 1s当たりの平均トークン数
- Peak VRAM
   - モデルロード時の最大メモリ量
   - NVIDIA使用時のみ測定可能。

## 画像について
- image&text to textの推論時に使用します。
- 画像はImageNetから一枚取得したものです。
- 任意の値でリサイズが可能です。（デフォルトは320×320）

## Ollama

### 概要
- Ollamaを使った推論をします。
- install.mdを参考にOllama等のインストールしてください。

## 使用モデルの制約
次の条件をすべて満たすモデルであれば、使用可能です。
- `llama.cpp`がロード可能な形式（主にGGUF）である。
- Ollama用のメタ情報（Modelfile）で定義できる。

### コマンド
- ollamaの起動
```bash
ollama serve
cd ai_inference_device_benchmark
```

- text to textの場合
```bash
python3 -m llm.ollama.benchmark_ollama
又は
python3 -m llm.ollama.benchmark_ollama --inference-mode text
```

- image&text to text(VLM)の場合
```bash
python3 -m llm.ollama.benchmark_ollama --inference-mode image
```

- NVIDIA GPUを使用している場合
`--is-nvidia True`を追加してください。

例
```bash
python3 -m llm.ollama.benchmark_ollama --inference-mode text --is-nvidia True
```

- 画像のリサイズ
```bash
python3 -m llm.ollama.benchmark_ollama --inference-mode image --image-size-h <number> --image-size-w <number>
```
   - `--image-size-h`で画像の縦、`--image-size-w`で画像の横のサイズを指定できます。
   - `<number>`はint型の数字を入力してください。

- modelの追加

llm/ollama/ollama_models.pyを開き、追加したいモデル名を記載してください。

   - TEXT_MODELS -> text to textでのみ使用するモデル
   - IMAGE_MODELS -> image&text to textでのみ使用するモデル
   - TEXT_IMG_MODELS -> 両方の場合で使用するモデル


## Open VINO

### 概要
- Open VINOを使った推論をします。
- 使用可能なデバイスはNPU、CPU、GPUです。
   - 任意で一つまたは複数の指定が可能（デフォルトはCPU）

### 使用モデルの制約
- optimum対応のモデルであれば使用可能です。以下のサイトを参考にしてください。

[Optimum対応モデル一覧](https://huggingface.co/docs/optimum-intel/openvino/models)

- 既に量子化されたモデルを使うこともできます。
   - int4で量子化されたモデル
[4-bit (INT4) GPTQ models](https://huggingface.co/models?other=gptq,4-bit&sort=trending)
   - NPU向けに最適化されたモデル
[LLMs optimized for NPU](https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu)

### modelのOpen VINO IR変換と量子化
```powershell
optimum-cli export openvino  \
       -m <model name> \
       --weight-format <int4 or nf4> \ 
       --sym \
       --ratio <number> \
       --group-size <number> \
      <directory name>
```
- `-m`：変換したいモデル名の指定。
- `--weight-format`：重みの形式を指定。int4またはnf4に変換できる（nf4はIntel® Core Ultra Processors Series 2のNPUで使用可能）
- `--sym`：対称量子化をする場合に入れる
- `--ratio`：どのくらいの重みを量子化するかの指定。1.0なら100%量子化される。
- `--group-size`：量子化の単位サイズ。Group quantization (GQ)の場合は整数値（128など）、Channel-wise quantization (CW) の場合は-1とする。
- `<directory name>`：出力先ディレクトリ名。

#### IR変換・量子化の例
- microsoft/Phi-3.5-mini-instructをIR変換・量子化するときのコマンド
```powershell
optimum-cli export openvino  \
       -m microsoft/Phi-3.5-mini-instruct \
       --weight-format int4 \ 
       --sym \
       --ratio 1.0 \
       --group-size 128 \
      Phi-3.5-mini-instruct
```

#### 既に量子化されたモデルを使う場合
`<model name>`に使用するモデルの名前を入れ、次のコマンドを実行してください。
```powershell
optimum-cli export openvino -m <model name>
```

### 実行方法
```powershell
python -m llm.openvino.benchmark_openvino \
        --model-path <your model path> \
        --inference-mode <text or image> \
        --model-type <llm or vlm> \
        --device <NPU, CPU, GPU> \
        --image-size-h <number> \
        --image-size-w <number>
```
- `--model-path`：IR変換・量子化したモデルのパス。必須項目。
- `--inference-mode`：text to text（text）かimage&text to text（imgae）の指定。デフォルトはimage。
- `--model-type`：テキスト専用モデルの場合は`llm`、マルチモーダルモデルの場合は`vlm`と指定する。デフォルトは`llm`。
- `--device`：使用デバイス（NPU, CPU, GPU）の指定。複数選択も可能（`--device NPU CPU GPU`）。デフォルトはCPU。
- `image-size-h`：画像のリサイズをする際に使用。画像の縦の値を指定できる（<number>はint型の数字を入力する）。デフォルトは320。
- `image-size-h`：画像のリサイズをする際に使用。画像の横の値を指定できる（<number>はint型の数字を入力する）。デフォルトは320。

#### 実行例
- Phi-3.5-mini-instruct(テキスト専用モデル)でtext to textを行う場合（デバイスはNPUとGPU）。
```powershell
python -m llm.openvino.benchmark_openvino \
        --model-path .\LLM_benchmark\openvino_models\Phi-3.5-mini-instruct \
        --inference-mode text \
        --model-type llm \
        --device NPU GPU
```

- gemma3(マルチモーダルモデル)でtext to textを行う場合（デバイスはGPU）。
```powershell
python -m llm.openvino.benchmark_openvino \
        --model-path .\LLM_benchmark\openvino_models\gemma-3-4b-it \
        --inference-mode text \
        --model-type vlm \
        --device GPU
```

- gemma3(マルチモーダルモデル)でimage&text to textを行う場合（デバイスはGPU。画像を300×300にリサイズ。）。
```powershell
python -m llm.openvino.benchmark_openvino \
        --model-path .\LLM_benchmark\openvino_models\gemma-3-4b-it \
        --inference-mode image \
        --model-type vlm \
        --device GPU \
        --image-size-h 300 \
        --image-size-w 300
```