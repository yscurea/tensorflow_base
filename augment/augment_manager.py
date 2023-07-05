import albumentations as alb
import cv2

# from .custom import CoarseDropout


class AugmentManager:
    def load_augment(self, file_name: str) -> alb.Compose:
        """Load albumentations Compose from file. Use for reproduction.

        Args:
        ----
        file_name (str): Be loaded file name.
        """
        return alb.load(filepath=file_name)

    def build_alb_augment(self, file_name: str, image_shape: tuple[int, int, int]) -> alb.Compose:
        """Build albumentations and save config file.

        Args:
        ----
        file_name (str): Save file name
        """
        transforms = self.__build_alb_augment(image_shape)
        alb.save(transform=transforms, filepath=file_name, data_format="yaml")
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
                    interpolation=cv2.INTER_CUBIC,
                    always_apply=True,
                ),
                alb.PadIfNeeded(
                    min_height=height,
                    min_width=width,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=None,
                    always_apply=True,
                ),
                alb.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=45,
                    interpolation=interpolation,
                    border_mode=border_mode,
                    value=None,
                    mask_value=None,
                    shift_limit_x=None,
                    shift_limit_y=None,
                    p=0.95,
                ),
                # alb.OneOf(
                #     [
                #         alb.HueSaturationValue(
                #             hue_shift_limit=20,
                #             sat_shift_limit=30,
                #             val_shift_limit=20,
                #             p=1.0,
                #         ),
                #         alb.RGBShift(
                #             r_shift_limit=20,
                #             g_shift_limit=20,
                #             b_shift_limit=20,
                #             p=1.0,
                #         ),
                #     ],
                #     p=0.2,
                # ),
                # alb.RandomBrightnessContrast(
                #     brightness_limit=0.2,
                #     contrast_limit=0.2,
                #     brightness_by_max=True,
                #     p=1,
                # ),
                # alb.OneOf(
                #     [
                #         CoarseDropout(
                #             max_holes=8,
                #             min_holes=1,
                #             max_height=height // 8,
                #             min_height=height // 32,
                #             max_width=width // 8,
                #             min_width=width // 32,
                #             fill_value="random",
                #             p=1.0,
                #         ),
                #         CoarseDropout(
                #             max_holes=4,
                #             min_holes=1,
                #             max_height=height // 4,
                #             min_height=height // 8,
                #             max_width=width // 4,
                #             min_width=width // 8,
                #             fill_value="random",
                #             p=1.0,
                #         ),
                #     ],
                #     p=0.8,
                # ),
                # alb.OneOf(
                #     [
                #         alb.OneOf(
                #             [
                #                 alb.Blur(blur_limit=7, p=1.0),
                #                 alb.GaussianBlur(
                #                     blur_limit=(3, 7), sigma_limit=0, p=1.0
                #                 ),
                #                 alb.GlassBlur(
                #                     sigma=0.7,
                #                     max_delta=2,
                #                     iterations=1,
                #                     mode="fast",
                #                     p=1.0,
                #                 ),
                #                 alb.MedianBlur(blur_limit=7, p=1.0),
                #             ],
                #             p=0.5,
                #         ),
                #         alb.MotionBlur(blur_limit=7, p=0.5),
                #     ],
                #     p=0.3,
                # ),
                # alb.OneOf(
                #     [
                #         alb.ISONoise(
                #             color_shift=(0.01, 0.05),
                #             intensity=(0.1, 0.5),
                #             p=1.0,
                #         ),
                #         alb.GaussNoise(var_limit=(10, 50), mean=0, p=1.0),
                #         alb.MultiplicativeNoise(
                #             multiplier=(0.9, 1.1), elementwise=False, p=1.0
                #         ),
                #     ],
                #     p=0.8,
                # ),
                # alb.OneOf(
                #     [
                #         alb.ImageCompression(
                #             quality_lower=99,
                #             quality_upper=100,
                #             compression_type=alb.augmentations.transforms.ImageCompression.ImageCompressionType.JPEG,
                #             p=1.0,
                #         ),
                #         alb.Downscale(
                #             scale_min=0.4,
                #             scale_max=0.4,
                #             interpolation=interpolation,
                #             p=1.0,
                #         ),
                #     ],
                #     p=0.2,
                # ),
            ]
        )
