from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    summarize: bool

    input_image_shape: tuple[int, int, int]
    num_classes: int
