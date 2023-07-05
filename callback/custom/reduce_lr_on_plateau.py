import mlflow
import tensorflow as tf


class CustomReduceLROnPlateau(tf.keras.callbacks.ReduceLROnPlateau):
    """ReduceLROnPlateau to log learning rate with mlflow."""

    def __init__(
        self,
        monitor: str,
        factor: float,
        patience: int,
        verbose: int,
        min_lr: float,
    ):
        super(CustomReduceLROnPlateau, self).__init__(
            monitor=monitor,
            factor=factor,
            patience=patience,
            verbose=verbose,
            min_lr=min_lr,
        )

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        mlflow.log_metric("learning rate", current_lr, step=epoch)
