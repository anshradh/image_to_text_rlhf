from dataclasses import dataclass
import torch
import torch.nn as nn
from imagen_pytorch import Unet, Imagen, ElucidatedImagen, ElucidatedImagenConfig
from dataclasses import dataclass, field
from typing import Sequence, Tuple, List
from einops import rearrange
from einops.layers.torch import Rearrange
from fancy_einsum import einsum
from utils import clones


@dataclass
class ElucidatedImagenValueHeadConfig(ElucidatedImagenConfig):
    intermediate_channels: int = 64
    intermediate_kernel_size: int = 3
    intermediate_stride: int = 1


class ElucidatedImagenValueHead(nn.Module):
    def __init__(self, config: ElucidatedImagenValueHeadConfig):
        super().__init__()
        self.config = config
        image_sizes = config.image_sizes
        in_channels = config.channels
        out_channels = config.intermediate_channels
        kernel_size = config.intermediate_kernel_size
        stride = config.intermediate_stride
        output_image_size = image_sizes[-1]
        flattened_size = out_channels * output_image_size ** 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding="same",
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            Rearrange("batch channels height width -> batch (channels height width)"),
            nn.Linear(flattened_size, flattened_size * 4),
            nn.GELU(),
            nn.Linear(flattened_size * 4, flattened_size),
            nn.GELU(),
            nn.Linear(flattened_size, 1),
        )

    def forward(self, x):
        return self.block(x)


class ElucidatedImagenWithValueHead:
    def __init__(
        self,
        model_config: ElucidatedImagenConfig,
        value_head_config: ElucidatedImagenValueHeadConfig,
    ):
        self.model = model_config.create()
        self.value_head = ElucidatedImagenValueHead(value_head_config)

    def forward(self, images, text_embeds, unet_number):
        model_output = self.model(
            images=images, text_embeds=text_embeds, unet_number=unet_number
        )
