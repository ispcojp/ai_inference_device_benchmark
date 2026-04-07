from ollama import chat
import time
import argparse
from typing import Iterable

from .. import vram_utils
from .. import image_utils
from .ollama_models import MODELS_TEXT, MODELS_IMG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference-mode", default="text", help="Inference mode: 'text' or 'image'"
    )
    parser.add_argument(
        "--image-size-h", default=320, type=int, help="image height: integer"
    )
    parser.add_argument(
        "--image-size-w", default=320, type=int, help="image width: integer"
    )
    parser.add_argument(
        "--is-nvidia",
        default=False,
        type=bool,
        help="Do you use an NVIDIA GPU?: 'True' or 'False'",
    )
    args = parser.parse_args()
    return args


def calculate_metrics(all_chunks: Iterable[dict]) -> tuple[float, float]:
    for chunk in all_chunks:
        if getattr(chunk, "done", False):
            total_tokens = getattr(chunk, "eval_count", 0)
            eval_duration_ns = getattr(chunk, "eval_duration", 0)
            total_duration_ns = getattr(chunk, "total_duration", 0)
            load_duration_ns = getattr(chunk, "load_duration", 0)
            break

    tps = total_tokens / (eval_duration_ns / 1e9)
    ftl = (total_duration_ns - eval_duration_ns - load_duration_ns) / 1e6

    return tps, ftl


def main():
    args = parse_args()
    models = []
    input_text = ""
    img_width = args.image_size_w
    img_height = args.image_size_h
    is_nvidia = args.is_nvidia
    resize_img = None

    if args.inference_mode == "text":
        models = MODELS_TEXT
        input_text = "Explain quantum computing in about 500 tokens."
    elif args.inference_mode == "image":
        models = MODELS_IMG

        image_path = image_utils.get_image()
        resized_image_path = image_utils.resize_image(image_path, img_width, img_height)
        resize_img_byte = image_utils.get_image_bytes(resized_image_path)
        resize_img = [resize_img_byte]
        input_text = "What is this picture?"
    else:
        print("--inference-modeでtext又はimageが指定されていません。")

    for model_name in models:

        if is_nvidia:
            handle = vram_utils.first_vram_setting()
            first_peak_vram = vram_utils.get_used_vram(handle)
            vram_holder, monitor_thread, stop_event = vram_utils.vram_monitor_start(
                handle
            )

        stream = chat(
            model=model_name,
            messages=[{"role": "user", "content": input_text, "images": resize_img}],
            stream=True,
            keep_alive=0,
        )

        tps, ftl = calculate_metrics(stream)

        print(f"model name：{model_name}")
        print(f"Tokens Per Second(TPS)：{tps:.2f}")
        print(f"First Token Latency(FTL)：{ftl:.2f}ms")

        if is_nvidia:
            peak_vram = vram_utils.vram_monitor_end(
                stop_event, monitor_thread, vram_holder, first_peak_vram
            )
            vram = peak_vram / (1024**2)
            print(f"Peak VRAM during load: {vram:.2f} MB")

        print("---")
        time.sleep(5)


if __name__ == "__main__":
    main()
