import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime
from onnxruntime.quantization import (
    QuantFormat,
    QuantType,
    quantize_static,
)

from models import get_transform
from data_loader import ImagenetteCalibrationDataReader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model_dir", help="input model directory")
    parser.add_argument("output_model_dir", help="output model directory")
    parser.add_argument(
        "--calib-dir", default="./imagenette2-320/train", help="calibration data set"
    )
    parser.add_argument(
        "--quant-format",
        default=QuantFormat.QDQ,
        type=QuantFormat.from_string,
        choices=list(QuantFormat),
    )
    parser.add_argument("--per-channel", default=False, type=bool)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_models = Path(args.input_model_dir).glob("*")
    output_model_dir = Path(args.output_model_dir)

    output_model_dir.mkdir(parents=True, exist_ok=True)

    # TODO: avoid hard coding
    input_name = "input"

    for model in input_models:
        data_reader = ImagenetteCalibrationDataReader(
            args.calib_dir,
            input_name,
            get_transform(model.stem),
            data_per_class=10)

        print(model.name)
        output_path = output_model_dir / model.name
        quantize_static(
            str(model),
            str(output_path),
            data_reader,
            quant_format=args.quant_format,
            per_channel=args.per_channel,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            optimize_model=False,
            # nodes_to_quantize=[f'Conv_{i}' for i in range(600)]
        )
        data_reader.rewind()


if __name__ == "__main__":
    main()
