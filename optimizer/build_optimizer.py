import tensorflow as tf

from config.type import OptimizerConfig

from .gradient_accumulation import GradientAccumulator

# TODO: Implement clipnorm, clipvalue, etc.


def build_optimizer(
    optimizer_config: OptimizerConfig,
) -> tf.optimizers.Optimizer:
    optimizer: str | tf.optimizers.Optimizer = optimizer_config.optimizer
    if isinstance(optimizer, tf.optimizers.Optimizer):
        return optimizer
    if optimizer == "Adam":
        optimizer = tf.optimizers.Adam(
            learning_rate=optimizer_config.learning_rate,
        )
        if optimizer_config.use_gradient_accumulation:
            return GradientAccumulator(
                optimizer,
                optimizer_config.accumulation_steps,
            )
        return tf.optimizers.Adam(learning_rate=optimizer_config.learning_rate)
    elif optimizer == "sgd" or optimizer == "SGD" or optimizer == "Sgd":
        optimizer = tf.optimizers.SGD(
            learning_rate=optimizer_config.learning_rate,
        )
        if optimizer_config.use_gradient_accumulation:
            return GradientAccumulator(
                optimizer,
                optimizer_config.accumulation_steps,
            )
        return tf.optimizers.Adam(
            learning_rate=optimizer_config.learning_rate,
        )
    else:
        raise NotImplementedError()
