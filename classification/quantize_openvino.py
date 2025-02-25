import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import onnx
import nncf
from models import get_transform
from data_loader_nncf import NNCFImagenetteCalibrationDataset
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model_dir", help="input model directory")
    parser.add_argument("output_model_dir", help="output model directory")
    parser.add_argument(
        "--calib-dir", default="./imagenette2-320/train", help="calibration data set"
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
        dataset = NNCFImagenetteCalibrationDataset(
            args.calib_dir,
            input_name,
            get_transform(model.stem),
            data_per_class=10
        )
        calibration_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        def transform_fn(data_item):
            images, _ = data_item
            return {input_name: images.numpy()}

        calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)
        onnx_model = onnx.load(model)
        print(f"start to quantize {model.name}.")
        try:
            quantized_model = nncf.quantize(onnx_model,
                                            calibration_dataset,
                                            target_device=nncf.TargetDevice.NPU,
                                            # fast_bias_correction=False
                                            )
        except:
            print(f"failed to quantize {model.name}")
        output_path = output_model_dir / model.name
        onnx.save(quantized_model, output_path)
        dataset.rewind()
if __name__ == "__main__":
    main()
