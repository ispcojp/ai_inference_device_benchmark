import time
import argparse
import sys

from PIL import Image
import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from openvino import Tensor
from openvino_genai import LLMPipeline
from typing import Callable

from .. import image_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None, help="your model path")
    parser.add_argument(
        "--inference-mode", default="text", help="Inference mode: 'text' or 'image'"
    )
    parser.add_argument(
        "--model-type", default="llm", help="model type: 'llm' or 'vlm'"
    )
    parser.add_argument(
        "--device",
        nargs="+",
        default=["CPU"],
        choices=["CPU", "GPU", "NPU"],
        help="device: 'CPU' or 'NPU' or 'GPU'",
    )
    parser.add_argument(
        "--image-size-h", default=320, type=int, help="image height: integer"
    )
    parser.add_argument(
        "--image-size-w", default=320, type=int, help="image width: integer"
    )
    args = parser.parse_args()
    return args


def load_image(image_file: str) -> Tensor:
    image = Image.open(image_file).convert("RGB")
    image_data = (
        np.array(image.get_flattened_data())
        .reshape(1, image.size[1], image.size[0], 3)
        .astype(np.uint8)
    )
    return ov.Tensor(image_data)


def make_streamer(state: dict) -> Callable[[str], bool]:
    def streamer(subword: str) -> bool:
        state["token_count"] += 1
        if state["first_token_time"] is None:
            state["first_token_time"] = time.time() - state["start_time"]
        return False

    return streamer


def make_pipeline(
    device: str,
    model_path: str,
    inference_mode: str,
    model_type: str,
    img_width: int,
    img_height: int,
) -> tuple[LLMPipeline, Tensor, str]:
    if inference_mode == "text":
        if model_type == "llm":
            pipe = ov_genai.LLMPipeline(model_path, device)
        elif model_type == "vlm":
            pipe = ov_genai.VLMPipeline(model_path, device)
        else:
            print("check your comand")
            sys.exit()

        input_text = "Explain quantum computing in about 100 tokens."
        image_tensor = None

    elif inference_mode == "image":
        pipe = ov_genai.VLMPipeline(model_path, device)
        input_text = "What is this picture?"
        image_path = image_utils.get_image()
        resized_image_path = image_utils.resize_image(image_path, img_width, img_height)
        image_tensor = load_image(resized_image_path)
    else:
        print("--inference-modeでtext又はimageが指定されていません。")

    return pipe, image_tensor, input_text


def main():
    args = parse_args()
    device = args.device
    model_path = args.model_path

    inference_mode = args.inference_mode
    model_type = args.model_type
    img_width = args.image_size_w
    img_height = args.image_size_h

    if not model_path:
        print("model pathを指定してください")
        sys.exit()
    print(f"model path: {model_path}")

    for d in device:
        pipe, image_tensor, input_text = make_pipeline(
            d, model_path, inference_mode, model_type, img_width, img_height
        )

        state = {"first_token_time": None, "start_time": None, "token_count": 0}

        streamer = make_streamer(state)
        kwargs = {"max_new_tokens": 100, "streamer": streamer}

        if image_tensor is not None:
            kwargs["image"] = image_tensor

        pipe.start_chat()

        state["start_time"] = time.time()
        pipe.generate(input_text, **kwargs)
        end_time = time.time()

        pipe.finish_chat()
        elapsed_time = end_time - state["start_time"] - state["first_token_time"]
        tps = state["token_count"] / elapsed_time

        print(f"device: {d}")
        print(f"First Token Latency(FTL): {state["first_token_time"]:.4f}s")
        print(f"Tokens Per Second(TPS): {tps:.4f}")
        print("---")


if __name__ == "__main__":
    main()
