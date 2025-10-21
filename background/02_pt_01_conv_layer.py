#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/20 15:42
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_pt_01_conv_layer.py
# @Desc     :   

from torch import Tensor, nn

from utils.config import EXAMPLE_IMG_PATH
from utils.CV import (read_image, display_image,
                      image2tensor, tensor2image)
from utils.helper import Timer


def main() -> None:
    """ Main Function """
    with Timer("Convolutional Layer"):
        # Get image properties
        img = read_image(str(EXAMPLE_IMG_PATH))

        # Show image
        display_image(img)

        # Transform image to tensor
        t: Tensor = image2tensor(img)
        channels, height, width = t.shape

        # Define a convolution layer
        layer = nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=6, stride=3, padding=0, bias=False)

        # Get the output tensor
        out = layer(t)

        # Show output image
        out_img = tensor2image(out)
        display_image(out_img)


if __name__ == "__main__":
    main()
