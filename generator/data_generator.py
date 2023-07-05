import math
import random

import albumentations as alb
import cv2
import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    """Sample data generator."""

    def __init__(
        self: "DataGenerator",
        batch_size: int,
        transforms: alb.Compose,
        data_list: list[tuple[str, int]] = [],
    ) -> None:
        self.__batch_size = batch_size
        self.__transforms = transforms
        # TODO: Prepare data. Sample data: image path and label list.
        self.__data_list: list[tuple[str, int]] = data_list
        random.shuffle(self.__data_list)

    def __len__(self: "DataGenerator"):
        return int(math.ceil(len(self.__data_list) / self.__batch_size))

    def __getitem__(self: "DataGenerator", index: int):
        start: int = index * self.__batch_size
        end: int = (index + 1) * self.__batch_size
        batch_data = self.__data_list[start:end]
        batch_x, batch_y = self.__get_data(batch_data)
        return batch_x, batch_y

    def __get_data(
        self, batch_data: list[tuple[str, int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build sample data."""
        batch_x = []
        batch_y = []
        for data in batch_data:
            img = cv2.imread(data[0])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            transformed = self.__transforms(image=img)
            transformed_image = transformed["image"]
            batch_x.append(transformed_image)
            batch_y.append(data[1])
        return np.array(batch_x), np.array(batch_y)
