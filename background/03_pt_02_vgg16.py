#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/20 19:41
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   03_pt_02_vgg16.py
# @Desc     :   

from torchvision.models import vgg16

from utils.helper import Timer


def main() -> None:
    """ Main Function """
    with Timer("Load AlexNet Model"):
        vgg = vgg16()
        print(vgg)


if __name__ == "__main__":
    main()
