from keras.callbacks import Callback, EarlyStopping, TensorBoard  # noqa: D100

from config.type import CallbackConfig

from .custom import CustomModelCheckpoint, CustomReduceLROnPlateau


def build_callback_list(callback_config: CallbackConfig) -> list[Callback]:
    """Build callback list from callback configs.

    Returns:
        list[Callback]: keras.callbacks.Callback list
    """
    callback_list: list[Callback] = []
    if callback_config.early_stopping_config is not None:
        callback_list.append(
            EarlyStopping(
                monitor=callback_config.early_stopping_config.monitor,
                patience=callback_config.early_stopping_config.patience,
                verbose=callback_config.early_stopping_config.verbose,
            ),
        )
    if callback_config.model_checkpoint_config is not None:
        callback_list.append(
            CustomModelCheckpoint(
                filepath=callback_config.model_checkpoint_config.filepath,
                monitor=callback_config.model_checkpoint_config.monitor,
                verbose=callback_config.model_checkpoint_config.verbose,
                save_best_only=callback_config.model_checkpoint_config.save_best_only,
                save_weights_only=callback_config.model_checkpoint_config.save_weights_only,
            ),
        )
    if callback_config.reduce_lr_on_plateau_config is not None:
        callback_list.append(
            CustomReduceLROnPlateau(
                monitor=callback_config.reduce_lr_on_plateau_config.monitor,
                factor=callback_config.reduce_lr_on_plateau_config.factor,
                patience=callback_config.reduce_lr_on_plateau_config.patience,
                min_lr=callback_config.reduce_lr_on_plateau_config.min_lr,
                verbose=callback_config.reduce_lr_on_plateau_config.verbose,
            ),
        )
    if callback_config.tensor_board_config is not None:
        callback_list.append(
            TensorBoard(
                callback_config.tensor_board_config.log_dir,
            ),
        )

    return callback_list
