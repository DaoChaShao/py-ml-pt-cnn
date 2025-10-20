#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/20 16:07
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   CV.py
# @Desc     :   

from cv2 import (imread, imshow, waitKey, destroyAllWindows,
                 cvtColor, COLOR_BGR2RGB)
from numpy import ndarray, uint8
from torch import Tensor, from_numpy

from utils.decorator import timer


@timer
def read_image(image_path: str) -> ndarray:
    """ Read an image from the given path """
    img = imread(image_path)
    height, width, channels = img.shape

    print(f"The image's height and width are {height} and {width}, and its channels are {channels}.")
    print(f"The image shape is {img.shape}.")

    return img


@timer
def display_image(image) -> None:
    """ Display an image from the given path """
    print("Click ESC to close the image window.")

    imshow("Image Window", image)
    waitKey(0)
    destroyAllWindows()


@timer
def image_to_tensor(image) -> Tensor:
    """ Reshape the image to the specified width and height """
    img = cvtColor(image, COLOR_BGR2RGB)

    # Transform the numpy image to tensor (channels, height, width)
    new = from_numpy(img).permute(2, 0, 1).float() / 255.0

    print(f"The image numpy shape is {img.shape}.")
    print(f"The image tensor shape is {new.shape}.")

    return new


@timer
def tensor_to_image(image: Tensor) -> ndarray:
    """ Reshape the tensor back to image format """
    img = image - image.min()
    img = img / img.max() * 255

    # Transform the tensor back to numpy image (height, width, channels)
    new = img.permute(1, 2, 0).detach().numpy().astype(uint8)

    print(f"The image tensor shape is {image.shape}.")
    print(f"The image numpy shape is {new.shape}.")

    return new
