import random
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def resize_and_padding(image: cv2.Mat, size: int, color=(0, 0, 0)):
    h, w = image.shape[:2]
    aspect = w / h
    nh = nw = size
    if 1 >= aspect:
        nw = round(nh * aspect)
    else:
        nh = round(nw / aspect)
    resized = cv2.resize(image, dsize=(nw, nh))
    h, w = resized.shape[:2]
    x = y = 0
    if h < w:
        y = abs(size - h) // 2
    else:
        x = abs(size - w) // 2
    resized = Image.fromarray(resized)
    canvas = Image.new(resized.mode, (size, size), color)
    canvas.paste(resized, (x, y))
    background_image = np.array(canvas)
    return background_image


def rotate_image(image, angle):
    height, width = image.shape[:2]
    image_center = (width / 2, height / 2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]
    rotated_mat = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def blend(
    foreground_image: cv2.Mat,
    background_image: cv2.Mat,
    src_size_ratio=0.3,
) -> Tuple[
    cv2.Mat,
    Tuple[
        int | float,
        int | float,
        int | float,
        int | float,
    ],
]:
    """
    src を dst に画像合成を行う
    png 画像の透過部分をマスク画像に利用して背景の上から貼り付けるように合成する
        src: 合成したい画像のパス（最前面に表示される）
        dst: 合成先の背景画像のパス
    return: 合成された結果の画像、(x_min, y_min, x_max, y_max)
    """
    foreground_image = rotate_image(foreground_image, random.randint(0, 360))
    foreground_image = cv2.cvtColor(foreground_image, cv2.COLOR_BGRA2RGBA)
    foreground_h, foreground_w = foreground_image.shape[:2]
    background_h, background_w = background_image.shape[:2]
    src_size = int(min(background_h, background_w) * src_size_ratio)
    if foreground_w >= foreground_h:
        ratio = src_size / foreground_w
    else:
        ratio = src_size / foreground_h
    new_width = int(foreground_w * ratio)
    new_height = int(foreground_h * ratio)
    foreground_image = cv2.resize(
        foreground_image,
        dsize=(new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )
    foreground_h, foreground_w = foreground_image.shape[:2]
    background_h, background_w = background_image.shape[:2]
    scale = random.uniform(0.3, 1.0)
    x = random.randint(0, background_image.shape[1] - int(foreground_w * scale))
    y = random.randint(0, background_image.shape[0] - int(foreground_h * scale))
    foreground_image = cv2.resize(
        foreground_image, (int(foreground_w * scale), int(foreground_h * scale))
    )
    alpha = foreground_image[:, :, 3]
    foreground_image = foreground_image[:, :, 0:3]
    roi = background_image[
        y : y + foreground_image.shape[0], x : x + foreground_image.shape[1]
    ]
    for c in range(0, 3):
        roi[:, :, c] = foreground_image[:, :, c] * (alpha / 255.0) + roi[
            :, :, c
        ] * (1.0 - alpha / 255.0)
    background_image[
        y : y + foreground_image.shape[0], x : x + foreground_image.shape[1]
    ] = roi
    return background_image, (
        x,
        y,
        x + foreground_image.shape[1],
        y + foreground_image.shape[0],
    )
