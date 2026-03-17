# install

## PyTorch

PyTorch は環境に合わせて GPU/CPU を選んでください。

```bash
# TODO: fix me
git clone 

cd dl_infer_benchmark

python3 -m venv venv
. venv/bin/activate

# torch GPU
pip install torch torchvision torchaudio

# example of torch CPU
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

Windows の場合 ORT を使うために Visual Studio 2019 のランタイムパッケージが必要です。
→ [install document](https://onnxruntime.ai/docs/install/) 参照

## OpenVINO

### 共通

Linux は pip は下記 pip コマンドの他 [Additional Configurations](https://docs.openvino.ai/2025/get-started/install-openvino/configurations.html) に従い GPU driver, NPU driver をインストールして下さい。

Windows は下記手順に従い事前に OpenVINO をインストールして下さい。
参考: <https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-windows.html>

- `C:\Program Files (x86)\Intel` を作成
- 上記 URL に記載のリンクから zip を取得、解凍
- 上記の Intel フォルダに openvino_2024.6.0 というフォルダ名でコピー
- 同梱の setupvars.bat を実行 (環境変数の設定)

### OpenVINO (non ORT API)

`classification/bentimark_openvino.py` を使用する場合は openvino をインストールしてください。

```bash
pip install openvino
```

### OpenVINO (ORT API)

`pip install onnxruntime-openvino` して下さい。

## OpenVINO Gen AI
- OpenVINO Gen AIを使ったLLM系のモデル推論をするための環境構築をします。
- 以下のようにモデルの変換に必要な依存関係をインストールしてください。

- Pythonのインストール
Python 3.12.10を以下のサイトからインストールしてください。
https://www.python.org/downloads/release/python-31210/

- windows仮想環境の作成
```powershell
python -m venv npu-env
npu-env\Scripts\activate
```

- 依存関係のインストール
```powershell
pip install nncf==2.19.0 onnx==1.20.0 optimum-intel==1.27.0 transformers==4.57.3
pip install torchvision==0.20.1                                               
pip install openvino==2025.4 openvino-tokenizers==2025.4 openvino-genai==2025.4
```
※versionが異なると動かない可能性があります。

## ollama
環境に合ったollamaをインストールしてください。

- ollamaのinstall (Linux)
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

- 使用するmodelの準備
```bash
ollama pull <model>
```
※`<model>`に使用したいモデル名を入れてください。

デフォルトで使用しているモデル
- "gpt-oss:20b"
- "moondream:1.8b"
- "gemma3:12b"
- "qwen3-vl:8b"

- Ollama Python Libraryのinstall
```bash
pip install ollama
```