# AI Inference Device Benchmark

- 2025 年現在 Deep Learning の利用はますます加速し、PC/エッジ/クラウドなど様々な環境で Deep Learning による推論処理を実行する機会が増えています。
- 特にプロジェクト開始時、モデルやハードウェアの選定にあたり推論速度の目安を知りたいことがあります。モデルが決まれば FLOPS やパラメタ数などのモデル規模は分かりますが、必ずしも推論速度と関連しないことが知られています。
  また、同じモデルでもハードウェアやミドルウェアが違うと速度の傾向が異なることもあります。
- そこで本レポジトリでは、[株式会社システム計画研究所／ISP](https://www.isp.co.jp) で用いているベンチマーク用のスクリプトを公開し、様々な環境での推論速度を共有することを目指します。

## 狙い

- 様々なハードウェア・ミドルウェアで、様々な Deep Learning モデルを実行した際の推論速度を計測します。
  2025/01 現在は画像認識タスクだけですが、様々なタスクを追加したいと考えています。
- AI 推論向けのハードウェアでは量子化を必要とするものが少なくありません。一方で、量子化により精度劣化が発生することもあります。
  そこで同じデータセットを用いて量子化前後の精度を比較できる様にします。量子化の設定により精度劣化を緩和できる場合はその設定も共有します。
  量子化やコンパイルに特別な手順が必要になるものについてはコマンドや設定等も共有します。
- 各種実験結果を [results](./results) に集約します。
  将来的にはグラフ化など見やすい形で公開できる様検討中です。

## サポートしている Deep Learning 推論ミドルウェア

- [PyTorch](https://pytorch.org)
- [ONNX Runtime (ORT)](https://onnxruntime.ai)
- [OpenVINO](https://github.com/openvinotoolkit/openvino)

ONNX Runtime の場合 ExecutionProvider により様々なハードウェアに対応可能です。現在、以下のものを評価しています。

- CPUExecutionProvider
- CUDAExecutionProvider
- TensorrtExecutionProvider
- OpenVINOExecutionProvider

## ドキュメント

- [インストール](./doc/install.md)
- 使い方
  - [画像認識タスク](./doc/usage_classification.md)
  - [画像認識タスク + 量子化](./doc/usage_classification_quantization.md)

## Contributing

- issues や PullRequest など歓迎です
- 質問・バグ報告・機能改善要求などは issues にお願いします
- 新たな環境での実験結果は PullRequest 歓迎です。
  その際、既存のものにならいハードウェアや OS を記載下さい。

## License

[LICENSE](LICENSE) 参照
