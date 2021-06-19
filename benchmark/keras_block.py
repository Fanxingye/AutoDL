from typing import Optional
from tensorflow.keras import applications
from autokeras.blocks.basic import KerasApplicationBlock


class ResNet50V1(KerasApplicationBlock):
    """Block for ResNetV1.
    """

    def __init__(self, pretrained: Optional[bool] = True, **kwargs):
        super().__init__(
            pretrained=pretrained,
            models={"resnet50v1": applications.ResNet50},
            min_size=32,
            **kwargs)


class ResNet50V2(KerasApplicationBlock):
    """Block for ResNetV2.
    """

    def __init__(self, pretrained: Optional[bool] = True, **kwargs):
        super().__init__(
            pretrained=pretrained,
            models={"resnet50v2": applications.ResNet50V2},
            min_size=32,
            **kwargs)
    