#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/20 22:01
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   03_pt_03_google_net.py
# @Desc     :   

from torchvision.models import googlenet

from utils.helper import Timer


def main() -> None:
    """ Main Function """
    with Timer("Load GoogleNet Model"):
        google_net = googlenet(init_weights=True)
        print(google_net)


if __name__ == "__main__":
    main()
