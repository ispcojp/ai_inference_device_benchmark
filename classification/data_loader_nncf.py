from itertools import groupby, islice
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
class NNCFImagenetteCalibrationDataset(Dataset):
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

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        if self.current_index == len(self.dataset):
            return None
        tensor, _ = self.dataset[idx]
        image_data = tensor.numpy()
        self.current_index += 1
        return (image_data, _)
    def get_k_data_per_class(self, k, image_folder):
        group = groupby(image_folder.samples, lambda x: x[1])
        new_samples = sum(
            map(lambda x: list(islice(x[1], 0, k)), group),
            [])
        image_folder.samples = new_samples
    def rewind(self):
        self.current_index = 0
