#!/usr/bin/python3

import argparse
from pathlib import Path

import torch

from models import (
    MODELS,
    get_model,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("outdir", default="classification_models")
    parser.add_argument("--models",
                        default=MODELS, nargs="+")
    parser.add_argument("--size", default=224, type=int,
                        help="size of input square")
    parser.add_argument("--times", default=10, type=int,
                        help="how many times to infer")
    parser.add_argument("--batch", default=1, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print(f"size: {args.size}")
    dummy_input = torch.randn(args.batch, 3, args.size, args.size)

    outdir = Path(f"{args.outdir}-{args.size}")
    outdir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        print(f"{model_name}")
        model = get_model(model_name)
        model.eval()

        input_names = ['input']
        output_names = ['score']

        outfile = outdir / f"{model_name}.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            str(outfile),
            verbose=False,
            input_names=input_names,
            output_names=output_names)


if __name__ == "__main__":
    main()
