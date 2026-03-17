import os

from torchvision.datasets.utils import download_url
from PIL import Image


def get_image() -> str:
    url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
    img_dir_path = "./llm/images"
    image_path = "./llm/images/sample.jpg"

    if not os.path.exists(img_dir_path):
        os.makedirs(img_dir_path)

    try:
        with open(image_path, "rb"):
            pass
    except FileNotFoundError:
        download_url(url, img_dir_path, "sample.jpg")

    return image_path


def resize_image(image_path: str, width: int, height: int) -> str:
    resized_image_path = "./llm/images/resized_sample.jpg"
    img = Image.open(image_path)
    resized_image = img.resize((width, height))
    resized_image.save(resized_image_path)
    return resized_image_path


def get_image_bytes(resized_image_path: str) -> bytes:
    with open(resized_image_path, "rb") as f:
        return f.read()
