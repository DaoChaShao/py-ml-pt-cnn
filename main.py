#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/20 12:47
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   main.py
# @Desc     :   

from random import randint
from torch import Tensor, device, optim, nn

from utils.config import (TRAIN_DATASET_PATH,
                          FASHION_CLASSES,
                          ACCELERATOR,
                          ALPHA,
                          EPOCHS, MODEL_SAVE_PATH)
from utils.CV import tensor2image, display_image
from utils.helper import Timer, RandomSeed
from utils.model import ConvolutionalModel
from utils.PT import (df2tensor, GrayTensorReshaper,
                      TorchDataset, TorchDataLoader, )
from utils.stats import load_data, split_data
from utils.trainer import TorchTrainer


def data_preparation() -> tuple[TorchDataLoader, TorchDataLoader]:
    # Load train dataset
    # raw: DataFrame = read_csv(TRAIN_DATASET_PATH)
    # print(type(raw), raw.shape)
    # print(raw)

    # Reload the data into two categories, such as X, y
    X, y = load_data(TRAIN_DATASET_PATH)
    # print(X.head(5))
    # print(y.head(5))

    # Spilt the data
    X_train_df, X_valid_df, y_train_df, y_valid_df = split_data(X, y)

    # Transform df into tensor
    X_train_tensor = df2tensor(X_train_df)
    X_valid_tensor = df2tensor(X_valid_df)
    y_train: Tensor = df2tensor(y_train_df, is_label=True)
    y_valid: Tensor = df2tensor(y_valid_df, is_label=True)
    print(y_train[0])

    # Reshape the flat data into image data for display and conv model training
    X_train = GrayTensorReshaper(X_train_tensor)
    # print(f"X_train.shape = {X_train.shape}")
    X_valid = GrayTensorReshaper(X_valid_tensor)
    # Display a random image
    # index: int = randint(0, len(X_train) - 1)
    # y_lbl = FASHION_CLASSES[int(y_train[index])]
    # print(y_lbl)
    # print()
    # x = X_train[index]
    # x = tensor2image(x)
    # display_image(x)

    # Setup dataset
    dataset_train = TorchDataset(X_train, y_train)
    print(dataset_train[0][1])
    dataset_valid = TorchDataset(X_valid, y_valid)
    # print(dataset_valid)

    batch_loader_train = TorchDataLoader(dataset_train)
    batch_loader_valid = TorchDataLoader(dataset_valid)

    return batch_loader_train, batch_loader_valid


def main() -> None:
    """ Main Function """
    with Timer("Data Preparation"):
        loader_train, loader_valid = data_preparation()
        print(loader_train)
        print(loader_valid)

    with RandomSeed("Model Training"):
        # Get the input_channels number
        in_channels: int = loader_train[0][0].shape[0]
        # print(in_channels)
        # print(loader_train[0][1])
        model = ConvolutionalModel(in_channels, 10)
        # print(model)
        model.to(device(ACCELERATOR))
        # Set up an optimiser
        optimizer = optim.Adam(model.parameters(), lr=ALPHA)
        # Set up a criterion
        criterion = nn.CrossEntropyLoss()
        # Initialise a trainer
        trainer = TorchTrainer(model, optimizer, criterion, ACCELERATOR)
        trainer.fit(loader_train, loader_valid, EPOCHS, MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
