## 構成

- CPU: Intel® Core™ Ultra7 155H
- GPU: Intel® Arc™ graphics
- NPU: Intel® AI Boost
- OS: Windows

## benchmark_openvino.py
- text to text

|         model name             | device | FTL[s] |   TPS   | 
| -------------------------------|--------|--------|---------|
| microsoft/Phi-3.5-mini-instruct|   NPU  | 6.5732 | 10.4052 |
|                                |   CPU  | 3.5664 | 25.4957 |
|                                |   GPU  | 0.2218 | 27.5077 |
| google/gemma-3-4b-it           |   NPU  | 4.7171 | 9.6081  |
|                                |   CPU  | 3.3711 | 19.9057 |
|                                |   GPU  | 0.2870 | 19.4119 |
| Qwen/Qwen2.5-7B                |   NPU  | 8.8057 | 6.6966  |
|                                |   CPU  | 6.6613 | 10.0158 |
|                                |   GPU  | 0.3997 | 15.1637 |

- image&text to text
画像は320×320にリサイズ

|         model name             | device | FTL[s] |   TPS   | 
| -------------------------------|--------|--------|---------|
| google/gemma-3-4b-it           |   NPU  | 15.1334| 9.0554  |
|                                |   CPU  | 13.9380| 15.7524 |
|                                |   GPU  | 4.6351 | 19.7767 |