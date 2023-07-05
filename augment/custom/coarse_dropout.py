import albumentations as A
import numpy as np
from albumentations.augmentations.dropout.functional import cutout


class CoarseDropout(A.CoarseDropout):
    def apply(self, image, fill_value="random", holes=(), **params):
        if fill_value == "random":
            fill_value = np.random.randint(low=0, high=256, size=3)
        return cutout(image, holes, fill_value)

    def apply_to_mask(self, image, mask_fill_value=0, holes=(), **params):
        if mask_fill_value is None:
            return image
        if mask_fill_value == "random":
            mask_fill_value = np.random.randint(low=0, high=256, size=3)
        return cutout(image, holes, mask_fill_value)

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def apply_to_bbox(self, bbox, **params):
        return bbox
