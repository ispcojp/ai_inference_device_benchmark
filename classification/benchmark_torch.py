#!/usr/bin/python3

import argparse
import time

import torch

from models import (
    MODELS,
    get_model,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="cpu",
                        help="cpu or gpu")
    parser.add_argument("--models", nargs="+",
                        default=MODELS)
    parser.add_argument("--size", default=224, type=int,
                        help="size of input square")
    parser.add_argument("--times", default=10, type=int,
                        help="how many times to infer")
    parser.add_argument("--batch", default=1, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print(f"{args.size=}")
    tensor = torch.randn(args.batch, 3, args.size, args.size)

    device = "cpu"
    if args.mode in ("gpu", "tensorrt"):
        if not torch.cuda.is_available():
            print("GPU not supported")
            return
        print("gpu mode")
        device = "cuda:0"

    for model_name in args.models:
        print(f"{model_name}")

        model = get_model(model_name)
        model.to(device)
        model.eval()
        tensor = tensor.to(device)

        # dry run
        with torch.no_grad():
            _ = model(tensor)

        total_ms = 0.0
        for _ in range(args.times):
            st = time.time()
            with torch.no_grad():
                _ = model(tensor)
            et = time.time()
            elasped_ms = (et - st) * 1000
            print(f"{elasped_ms} [ms]")
            total_ms += elasped_ms

        avg_ms = total_ms / args.times
        print(f"{model_name:30}: {avg_ms:>10.3f} [ms] in average")


if __name__ == "__main__":
    main()
