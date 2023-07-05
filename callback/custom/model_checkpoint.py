from pathlib import Path

import mlflow
import tensorflow as tf


class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """ModelCheckpoint to log val_loss or something with mlflow.
    And remove non best weights even not same name.
    """

    def __init__(
        self,
        filepath: str,
        monitor: str,
        remove_non_best_weights: bool,
        verbose: int,
        save_best_only: bool,
        save_weights_only: bool,
        *args,
        **kwargs
    ):
        super(CustomModelCheckpoint, self).__init__(
            filepath,
            monitor=monitor,
            verbose=verbose,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            *args,
            **kwargs,
        )
        self.remove_non_best_weights = remove_non_best_weights
        self.prev_filepath: Path | None = None

    def on_epoch_end(self: "CustomModelCheckpoint", epoch: int, logs=None):
        super().on_epoch_end(epoch, logs)
        if logs is None:
            return

        val_loss = logs.get("val_loss")
        val_accuracy = logs.get("val_accuracy")
        mlflow.log_metrics(
            {
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            },
            epoch + 1,
        )
        current = logs.get(self.monitor)
        if self.prev_filepath is None:
            self.prev_filepath = Path(
                self.filepath.format(epoch=epoch + 1, **logs)
            )
        elif (
            self.remove_non_best_weights
            and self.save_best_only
            and self.monitor_op(current, self.best)
        ):
            self.prev_filepath.unlink(missing_ok=False)
            self.prev_filepath = Path(
                self.filepath.format(epoch=epoch + 1, **logs)
            )
