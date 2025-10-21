#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/20 12:53
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   config.py
# @Desc     :   


from pathlib import Path

# Set base directory
BASE_DIRECTORY = Path(__file__).resolve().parent.parent
# Data file paths
EXAMPLE_IMG_PATH = BASE_DIRECTORY / "data/Nemo.png"
# Model save path
MODEL_SAVE_PATH = BASE_DIRECTORY / "models/model.pth"
# Train and test dataset path
TRAIN_DATASET_PATH = BASE_DIRECTORY / "data/fashion-mnist_train.csv"
TEST_DATASET_PATH = BASE_DIRECTORY / "data/fashion-mnist_test.csv"

# Data map
FASHION_CLASSES = {
    0: "T恤/上衣 (T-shirt/top)",
    1: "裤子 (Trouser)",
    2: "套头衫 (Pullover)",
    3: "连衣裙 (Dress)",
    4: "外套 (Coat)",
    5: "凉鞋 (Sandal)",
    6: "衬衫 (Shirt)",
    7: "运动鞋 (Sneaker)",
    8: "包 (Bag)",
    9: "短靴 (Ankle boot)"
}

# Data processing parameters
RANDOM_STATE: int = 27
VALID_SIZE: float = 0.2
IS_SHUFFLE: bool = True

# PCA parameters
PCA_VARIANCE_THRESHOLD: float = 0.95

# Dataset & Dataloader settings
BATCHES: int = 32

# Training hyperparameters
HIDDEN_UNITS: int = 128
ALPHA: float = 0.01
ALPHA4REDUCTION: float = 0.3
EPOCHS: int = 20
ACCELERATOR: str = "cpu"
