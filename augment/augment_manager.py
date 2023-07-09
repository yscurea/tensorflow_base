from pathlib import Path

import albumentations as alb
import cv2


# TODO: Reconsider how to manage augment processing. About also other than image processing.
class AugmentManager:
    def load_augment(self, file_name: str) -> alb.Compose:
        """Load albumentations Compose from file. Use for reproduction.

        Args:
        ----
        file_name (str): Be loaded file name.
        """
        return alb.load(filepath=file_name)

    def build_alb_augment(self, image_shape: tuple[int, int, int], filepath: Path | None = None) -> alb.Compose:
        """Build albumentations and save config file.

        Args:
        ----
        file_name (str): Save file name
        """
        transforms = self.__build_alb_augment(image_shape)
        if filepath is not None:
            file_ext = filepath.suffix[1:]
            alb.save(
                transform=transforms,
                filepath=str(filepath),
                data_format=file_ext,
            )
        return transforms

    def __build_alb_augment(self, image_shape: tuple[int, int, int]) -> alb.Compose:
        """Build albumentations augment. TODO: Implement albumentations transform."""
        interpolation = cv2.INTER_CUBIC
        border_mode = cv2.BORDER_REPLICATE
        height = image_shape[0]
        width = image_shape[1]
        return alb.Compose(
            [
                alb.LongestMaxSize(
                    max_size=max(height, width),
                    interpolation=interpolation,
                    always_apply=True,
                ),
                alb.PadIfNeeded(
                    min_height=height,
                    min_width=width,
                    border_mode=border_mode,
                    value=None,
                    always_apply=True,
                ),
                alb.ShiftScaleRotate(
                    shift_limit=0.10,
                    scale_limit=0.15,
                    rotate_limit=5,
                    interpolation=interpolation,
                    border_mode=border_mode,
                    value=None,
                    mask_value=None,
                    shift_limit_x=None,
                    shift_limit_y=None,
                    p=0.95,
                ),
            ]
        )
