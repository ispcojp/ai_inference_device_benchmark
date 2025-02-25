from itertools import groupby, islice

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from onnxruntime.quantization import CalibrationDataReader


IMAGENETTE_TO_IMAGENET = [
    0,     # n01440764
    207,   # n02102040
    482,   # n02979186
    491,   # n03000684
    497,   # n03028079
    566,   # n03394916
    569,   # n03417042
    571,   # n03425413
    574,   # n03445777
    701,    # n03888257
]


def target_transform_imagenette_to_imagenet(i):
    return IMAGENETTE_TO_IMAGENET[i]


class ImagenetteCalibrationDataReader(CalibrationDataReader):
    def __init__(self, imagenette_dir, input_name, transform,
                 image_size=(224, 224),
                 data_per_class=-1):
        self.input_name = input_name
        self.image_size = image_size
        self.current_index = 0

        self.dataset = ImageFolder(imagenette_dir,
                                   transform)
        if data_per_class > 0:
            self.get_k_data_per_class(data_per_class, self.dataset)

    def get_next(self):
        if self.current_index == len(self.dataset):
            return None

        tensor, _ = self.dataset[self.current_index]
        image_data = tensor.unsqueeze(0).numpy()
        self.current_index += 1

        return {self.input_name: image_data}

    def get_k_data_per_class(self, k, image_folder):
        group = groupby(image_folder.samples, lambda x: x[1])
        new_samples = sum(
            map(lambda x: list(islice(x[1], 0, k)), group),
            [])
        image_folder.samples = new_samples

    def rewind(self):
        self.current_index = 0
