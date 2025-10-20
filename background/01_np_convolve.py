#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/20 13:18
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   01_np_convolve.py
# @Desc     :   

from numpy import ndarray, array, convolve
from pprint import pprint

from utils.helper import Beautifier


def main() -> None:
    """ Main Function """
    signals: ndarray = array([1, 2, 3, 4, 5])  # length N=5
    V: ndarray = array([0, 0.5, 1])  # length M=3,  V is reversed, like v[::-1]

    # full: return the convolution at each point of overlap, with an output shape of (N+M-1,)
    # same: return the central part of the convolution that is the same size as a
    # valid: return only those parts of the convolution that are computed without the zero-padded edges
    full: ndarray = convolve(signals, V, mode="full")
    same: ndarray = convolve(signals, V, mode="same")
    valid: ndarray = convolve(signals, V, mode="valid")

    """
    0   [1]         × [0]           1×0                 0.0
    1   [1, 2]      × [0.5, 0]      1×0.5 + 2×0         0.5
    2   [1, 2, 3]   × [1, 0.5, 0]   1×1 + 2×0.5 + 3×0   2.0
    3   [2, 3, 4]   × [1, 0.5, 0]   2×1 + 3×0.5 + 4×0   3.5
    4   [3, 4, 5]   × [1, 0.5, 0]   3×1 + 4×0.5 + 5×0   5.0
    5   [4, 5]      × [1, 0.5]      4×1 + 5×0.5         6.5
    6   [5]         × [1]           5×1                 5.0

    """

    with Beautifier("NumPy Convolve Example"):
        print(f"Full Convolution of length {len(full)}:")
        pprint(full)
        print(f"Same Convolution of length {len(same)}:")
        pprint(same)
        print(f"Valid Convolution of length {len(valid)}:")
        pprint(valid)


if __name__ == "__main__":
    main()
