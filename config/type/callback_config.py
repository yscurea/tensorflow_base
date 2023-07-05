from dataclasses import dataclass


@dataclass
class EarlyStoppingConfig:
    monitor: str = "val_loss"
    patience: int = 50
    verbose: int = 1


@dataclass
class TensorBoardConfig:
    log_dir: str


@dataclass
class ModelCheckpointConfig:
    filepath: str = "${log_dir}/${hydra:job.num}/{epoch}_{val_loss:.4f}.h5"
    save_best_only: bool = True
    save_weights_only: bool = True
    monitor: str = "val_loss"
    verbose: int = 1
    remove_non_best_weights: bool = True


@dataclass
class ReduceLROnPlateauConfig:
    monitor: str = "val_loss"
    factor: float = 0.1
    patience: int = 16
    min_lr: float = 1.0e-7
    verbose: int = 1


@dataclass
class CallbackConfig:
    early_stopping_config: EarlyStoppingConfig | None
    tensor_board_config: TensorBoardConfig | None
    model_checkpoint_config: ModelCheckpointConfig | None
    reduce_lr_on_plateau_config: ReduceLROnPlateauConfig | None
