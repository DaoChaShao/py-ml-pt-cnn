#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/20 16:46
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   conv_layer.py
# @Desc     :

from numpy import ndarray, ascontiguousarray, uint8
from torch import Tensor, nn
from utils.config import EXAMPLE_IMG_PATH
from utils.CV import (read_image,
                      image_to_tensor, tensor_to_image)

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel)
from sys import argv, exit


def train() -> tuple[ndarray, ndarray]:
    """ Main Function """
    # Get image properties
    img = read_image(str(EXAMPLE_IMG_PATH))

    # Transform image to tensor
    t: Tensor = image_to_tensor(img)
    channels, _, _ = t.shape

    # Define a convolution layer
    layer = nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=6, stride=3, padding=0, bias=False)

    # Get the output tensor
    out = layer(t)

    # Show output image
    out_img = tensor_to_image(out)

    return img, out_img


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Convolution Example")
        self.resize(800, 400)
        self._widget = QWidget(self)
        self.setCentralWidget(self._widget)

        self._btn_labels = ["Plot", "Clear", "Exit"]
        self._buttons = []

        self._lbl_names = ["Original Image", "Convoluted Image"]
        self._images = [train()[0], train()[1]]
        self._labels = []

        self._setup()

    def _setup(self) -> None:
        _layout = QVBoxLayout(self._widget)
        _row_lbl = QHBoxLayout()
        _row_btn = QHBoxLayout()

        for lbl in self._lbl_names:
            label = QLabel(lbl, self)
            label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            label.setMinimumSize(400, 200)
            self._labels.append(label)
            _row_lbl.addWidget(label)
        _layout.addLayout(_row_lbl)

        funcs = [
            self._click2plot,
            self._click2clear,
            self.close,
        ]
        for i, label in enumerate(self._btn_labels):
            button = QPushButton(label, self)
            button.clicked.connect(funcs[i])
            if button.text() == "Clear":
                button.setEnabled(False)
            self._buttons.append(button)
            _row_btn.addWidget(button)
        _layout.addLayout(_row_btn)

        self._widget.setLayout(_layout)

    def _click2plot(self) -> None:
        """ Plot random data points """
        for i, lbl in enumerate(self._labels):
            arr = self._images[i]
            print(f"Image {i} - Min: {arr.min()}, Max: {arr.max()}, Dtype: {arr.dtype}")
            arr = ascontiguousarray(arr).astype(uint8)
            h, w, c = arr.shape
            img = QImage(arr.data, w, h, w * c, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(img)
            lbl.setPixmap(pixmap)
            lbl.setScaledContents(True)

        for button in self._buttons:
            if button.text() == "Clear":
                button.setEnabled(True)
        for button in self._buttons:
            if button.text() == "Plot":
                button.setEnabled(False)

    def _click2clear(self) -> None:
        """ Clear the chart """
        for lbl in self._labels:
            lbl.clear()

        for button in self._buttons:
            if button.text() == "Clear":
                button.setEnabled(False)
        for button in self._buttons:
            if button.text() == "Plot":
                button.setEnabled(True)


def main() -> None:
    """ Main Function """
    app = QApplication(argv)
    window = MainWindow()
    window.show()
    exit(app.exec())


if __name__ == "__main__":
    main()
