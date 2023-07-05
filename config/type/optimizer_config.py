from dataclasses import dataclass

from keras.optimizers import Optimizer


@dataclass
class OptimizerConfig:
    learning_rate: float | None
    optimizer: str | Optimizer
    use_gradient_accumulation: bool
    accumulation_steps: int
