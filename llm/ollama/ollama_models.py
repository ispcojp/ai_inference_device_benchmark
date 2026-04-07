# text to textでのみ使用するモデル
TEXT_MODELS = [
    "gpt-oss:20b"
]

# image&text to textでのみ使用するモデル
IMAGE_MODELS = [
    "moondream:1.8b"
]

# text to textとimage&text to textの両方で使用するモデル
TEXT_IMG_MODELS = [
    "gemma3:12b",
    "qwen3-vl:8b"
]

MODELS_TEXT = sorted(TEXT_MODELS + TEXT_IMG_MODELS)

MODELS_IMG = sorted(IMAGE_MODELS + TEXT_IMG_MODELS)
