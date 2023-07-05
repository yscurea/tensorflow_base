from dataclasses import dataclass

import tensorflow as tf

from .callback_config import CallbackConfig
from .data_config import DataConfig
from .model_config import ModelConfig
from .optimizer_config import OptimizerConfig


@dataclass
class TrainConfig:
    comment: str
    log_dir: str
    epochs: int
    verbose: int
    batch_size: int
    workers: int
    use_multiprocessing: bool
    use_mixed_float16: bool

    callback_config: CallbackConfig
    model_config: ModelConfig
    data_config: DataConfig

    optimizer_config: OptimizerConfig
    loss: dict[str, str] | str | tf.losses.Loss | dict[str, tf.losses.Loss]
    loss_weights: dict[str, float] | None
    metrics: list[str] | str | None
