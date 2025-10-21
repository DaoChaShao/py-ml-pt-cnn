#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/21 19:29
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   predictor.py
# @Desc     :   

from os import path
from random import randint
from torch import load

from utils.config import (MODEL_SAVE_PATH,
                          TEST_DATASET_PATH, )
from utils.helper import Timer
from utils.highlighter import red, green
from utils.model import ConvolutionalModel
from utils.PT import df2tensor, GrayTensorReshaper
from utils.stats import load_data


def main() -> None:
    """ Main Function """
    with Timer("Fashion MNIST Prediction"):
        if path.exists(MODEL_SAVE_PATH):
            print("The model has already been trained.\n")

            X, y = load_data(TEST_DATASET_PATH)

            X_tensor = df2tensor(X)
            y_test = df2tensor(y, is_label=True)

            X_test = GrayTensorReshaper(X_tensor)
            print(X_test.shape)

            # Due to the saved model structure, we need to define the model architecture again
            in_channels: int = X_test.shape[1]
            model = ConvolutionalModel(in_channels, 10)
            state_dict = load(MODEL_SAVE_PATH)
            model.load_state_dict(state_dict)
            model.eval()
            print("The model has been loaded successfully!")

            index: int = randint(0, len(X_test) - 1)
            print(f"The random index is {index}.")
            # Due to the input image size must be [batch, channel, height, width], resize the image shape
            input_tensor = X_test[index].unsqueeze(0)
            predictions = model(input_tensor)
            pred = predictions.argmax(axis=1)
            print(f"The original label is {y_test[index]}, and the predicted label is {pred}.")
            print(f"The prediction is {green("True") if pred == y_test[index] else red("False")}.")
        else:
            print("The model has not been trained.")


if __name__ == "__main__":
    main()
