#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/20 18:52
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_pt_02_pooling_layer.py
# @Desc     :   

from torch import Tensor, nn

from utils.config import EXAMPLE_IMG_PATH
from utils.CV import (read_image, display_image,
                      image_to_tensor, tensor_to_image)
from utils.helper import Timer


def main() -> None:
    """ Main Function """
    with Timer("Convolutional Layer"):
        # Get image properties
        img = read_image(str(EXAMPLE_IMG_PATH))

        # Show image
        # display_image(img)

        # Transform image to tensor
        t: Tensor = image_to_tensor(img)
        channels, height, width = t.shape

        # Define a convolution layer
        convolutional_layer = nn.Conv2d(
            in_channels=channels, out_channels=3,
            kernel_size=6, stride=3, padding=0,
            bias=False
        )

        # Get the output tensor
        out = convolutional_layer(t)

    with Timer("Pooling Layer"):
        pooling_layer = nn.MaxPool2d(2, 2, padding=1)
        out = pooling_layer(out)

        # Show output image
        out_img = tensor_to_image(out)
        display_image(out_img)


if __name__ == "__main__":
    main()
