## 構成

- CPU: Intel(R) Core(TM) i9-13900K
- GPU: NVIDIA GeForce RTX 4090
- OS: Linux

## benchmark_ollama.py
- text to text

|  model name | FTL[ms] |   TPS   | Peak VRAM during load[MB] |
| ------------|---------|---------|---------------------------|
| gemma3:12b  | 275.82  |  98.61  |          9226.44          |
| gpt-oss:20b | 391.73  | 189.80  |         13000.44          |
| qwen3-vl:8b | 193.43  | 156.30  |          7746.44          |

- image&text to text
画像は320×320にリサイズ

|   model name   | FTL[ms] |   TPS   | Peak VRAM during load[MB] |
| ---------------|---------|---------|---------------------------|
| gemma3:12b     | 270.10  |  99.37  |          9228.44          |
| moondream:1.8b | 288.06  |  474.13 |          2726.44          |
| qwen3-vl:8b    | 163.63  |  157.76 |          7746.44          |