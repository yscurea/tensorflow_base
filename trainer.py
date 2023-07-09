from pathlib import Path

import albumentations as alb
import cv2
import hydra
import mlflow
import tensorflow as tf
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from augment.augment_manager import AugmentManager
from callback import build_callback_list
from config.type import TrainConfig
from generator.data_generator import DataGenerator
from model.model import build_model
from optimizer import build_optimizer
from utils import flatten, get_best_weights


def train(train_config: TrainConfig) -> float:
    mlflow.set_tracking_uri(uri=f"file://{hydra.utils.get_original_cwd()}/logs/mlruns")
    mlflow.set_experiment(train_config.comment)
    log_dir = HydraConfig.get().runtime.output_dir
    print(log_dir)

    with mlflow.start_run():
        mlflow.log_params(dict(flatten(OmegaConf.to_container(train_config))))
        mlflow.log_artifact(log_dir + "/.hydra/config.yaml")

        image_height = train_config.model_config.input_image_shape[0]
        image_width = train_config.model_config.input_image_shape[1]

        # Build augment and save setting file
        augment_manager = AugmentManager()
        train_data_transforms = augment_manager.build_alb_augment(
            image_shape=train_config.model_config.input_image_shape,
            filepath=Path(log_dir) / train_config.data_config.augment_setting_path_to_save,
        )
        validation_data_transforms = alb.Compose(
            [
                alb.LongestMaxSize(
                    max_size=max(image_height, image_width),
                    interpolation=cv2.INTER_CUBIC,
                    always_apply=True,
                ),
                alb.PadIfNeeded(
                    min_height=image_height,
                    min_width=image_width,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=None,
                    always_apply=True,
                ),
            ]
        )

        # Build model
        model: tf.keras.Model = build_model(train_config.model_config)
        optimizer = build_optimizer(train_config.optimizer_config)
        model.compile(
            optimizer=optimizer,
            loss=train_config.loss,
            loss_weights=train_config.loss_weights,
            metrics=train_config.metrics,
        )

        # Define callbacks
        callbacks = build_callback_list(train_config.callback_config)

        # Prepare data and generator.
        train_data_generator = DataGenerator(
            train_config.batch_size,
            transforms=train_data_transforms,
        )
        validation_data_generator = DataGenerator(
            train_config.batch_size,
            transforms=validation_data_transforms,
        )

        # Reduce the amount of calculation. NOTE: that accuracy may be compromised.
        if train_config.use_mixed_float16:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

        # Train
        _ = model.fit(
            train_data_generator,
            validation_data=validation_data_generator,
            epochs=train_config.epochs,
            shuffle=train_config.epochs,
            callbacks=callbacks,
            verbose=train_config.verbose,
            workers=train_config.workers,
            use_multiprocessing=train_config.use_multiprocessing,
        )

        # Test best model. Change test method as needed.
        best_weights_path = get_best_weights(Path("weights"))
        if best_weights_path is None:
            raise ValueError("Weights path is not found in weights directory")
        model.load_weights(str(best_weights_path))
        loss, accuracy = model.evaluate(validation_data_generator)

        # Log best score
        # mlflow.log_metrics({"best_loss": loss, "best_accuracy": accuracy})
        mlflow.tensorflow.log_model(model)

        return loss
