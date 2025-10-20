#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/20 22:04
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   03_pt_04_resnet.py
# @Desc     :   

from torchvision.models import resnet50

from utils.helper import Timer


def main() -> None:
    """ Main Function """
    with Timer("Load ResNet50 Model"):
        resnet = resnet50()
        print(resnet)


if __name__ == "__main__":
    main()
