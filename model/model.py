import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from config.type.model_config import ModelConfig


def build_model(model_config: ModelConfig) -> tf.keras.Model:
    input_shape = model_config.input_image_shape
    input_layer = tf.keras.layers.Input(shape=input_shape)
    preprocessed_input = preprocess_input(input_layer)
    backbone = MobileNetV2(
        include_top=False,
        input_shape=input_shape,
        input_tensor=preprocessed_input,
        weights="imagenet",
        pooling="avg",
        alpha=0.35,
    )
    backbone.trainable = True
    x = backbone.output
    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.models.Model(
        inputs=input_layer,
        outputs=output_layer,
    )
    return model
