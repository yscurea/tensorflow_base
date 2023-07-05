import tensorflow as tf

from config.type import OptimizerConfig


def build_optimizer(
    optimizer_config: OptimizerConfig,
) -> tf.optimizers.Optimizer:
    optimizer: str | tf.optimizers.Optimizer = optimizer_config.optimizer
    if isinstance(optimizer, str):
        if optimizer == "Adam":
            return tf.optimizers.Adam(
                learning_rate=optimizer_config.learning_rate
            )
        else:
            raise NotImplementedError()
    else:
        return optimizer
