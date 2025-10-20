#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/20 19:35
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   03_pt_01_alex_net.py
# @Desc     :

from torchvision.models import alexnet

from utils.helper import Timer

def main() -> None:
    """ Main Function """
    with Timer("Load AlexNet Model"):
        alex = alexnet()
        print(alex)


if __name__ == "__main__":
    main()
