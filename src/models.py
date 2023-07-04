from torch import nn
from torchvision import models

from .config import *


def valid_model_name(model_name: str):
    """
    Функция проверки поддержки модели
    """
    assert model_name in models_names, f"'{model_name}'не поддерживается. Доступные модели: {models_names}"


def get_image_model(name: str,
                    pretrained: bool = True,
                    freeze_weight: bool = False,
                    num_classes: int = 3
                    ):
    """
    Функция получения сконфигурированного классификатора изображений
    """
    valid_model_name(name)

    models_package = models.__dict__

    last_layer_name = last_layer_replace_dict[name]
    import_name = get_models_dict[name]
    weight_name = get_weights_dict[f"{import_name}_weights"]

    if "E2E" in name:
        weights = models_package[weight_name].IMAGENET1K_SWAG_E2E_V1
    else:
        weights = models_package[weight_name].DEFAULT

    model = models_package[import_name](weights=weights if pretrained else None)

    if freeze_weight:
        for param in model.parameters():
            param.requiresGrad = False

    if "." not in last_layer_name:
        in_features = model._modules[last_layer_name].in_features
        model._modules[last_layer_name] = nn.Linear(in_features=in_features, out_features=8)

    else:
        last_layer_arr = last_layer_name.split(".")

        try:
            last_layer_arr[1] = int(last_layer_arr[1])
        except ValueError:
            last_layer_arr[1] = 0

        in_features = model._modules[last_layer_arr[0]][last_layer_arr[1]].in_features
        model._modules[last_layer_arr[0]][last_layer_arr[1]] = nn.Linear(in_features=in_features,
                                                                         out_features=num_classes)

    return model, weights.transforms()
