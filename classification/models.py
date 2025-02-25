import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
from torchvision import transforms


TIMM_MODELS = [
    "mobilenetv3_large_100",
    "mobilenetv3_small_050",
    "resnet18",
    "mobilenetv2_110d",
    "mobilenetv2_100",
    "mobilenetv2_050",
    "mixnet_s",
    "ese_vovnet19b_dw",
    "efficientnet_b0",
    "mnasnet_a1",
    "vgg13",
    "repvgg_b0",
]
TORCH_HUB_MODELS = [
    "shufflenet_v2_x1_0",
]
MODELS = sorted(TIMM_MODELS + TORCH_HUB_MODELS)


def get_model(model_name, pretrained=True):
    if model_name in TORCH_HUB_MODELS:
        model = model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=pretrained)
    else:
        model = timm.create_model(model_name, pretrained=pretrained)

    return model


def get_transform(model_name):
    if model_name in TIMM_MODELS:
        data_config = resolve_data_config({}, model=timm.create_model(model_name))
        return create_transform(**data_config)

    if model_name == TORCH_HUB_MODELS[0]:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    raise Exception("oops")

