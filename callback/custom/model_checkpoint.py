import logging  # noqa: D100
from pathlib import Path
from typing import Any

import mlflow
import tensorflow as tf


class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """ModelCheckpoint to log val_loss or something with mlflow, and remove non best weights even not same name."""

    def __init__(
        self: "CustomModelCheckpoint",
        filepath: str,
        monitor: str,
        verbose: int,
        save_best_only: bool,
        save_weights_only: bool,
    ) -> None:
        super().__init__(
            filepath,
            monitor=monitor,
            verbose=verbose,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
        )
        self.prev_filepath: Path | None = None

    def on_epoch_end(self: "CustomModelCheckpoint", epoch: int, logs: Any | None = None) -> None:
        """Call ModelCheckpoint's on_epoch_end and remove the non-best weights file."""
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
                self.filepath.format(
                    epoch=epoch + 1,
                    **logs,
                ),
            )
        elif self.save_best_only and self.monitor_op(current, self.best):
            self.prev_filepath.unlink(missing_ok=False)
            logging.info("Remove the file: %s", self.prev_filepath)
            self.prev_filepath = Path(
                self.filepath.format(
                    epoch=epoch + 1,
                    **logs,
                ),
            )

        super().on_epoch_end(epoch, logs)
