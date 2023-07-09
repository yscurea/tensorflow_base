from pathlib import Path
from typing import Callable


def val_loss_operator(old: Path, new: Path) -> bool:
    return float(old.stem.split("_")[-1]) > float(new.stem.split("_")[-1])


def get_best_weights(weight_dir: Path, operator: Callable[[Path, Path], bool] = val_loss_operator) -> Path | None:
    """Get best weights. expected file format: ${epoch}_${val_loss}.h5"""
    best_weight_path: Path | None = None
    for weight_path in weight_dir.glob("*.h5"):
        print(str(weight_path))
        if best_weight_path is None:
            best_weight_path = weight_path
        elif operator(best_weight_path, weight_path):
            best_weight_path = weight_path
    print(f"\033[94mBest weight: {best_weight_path}\033[0m")
    return best_weight_path


if __name__ == "__main__":
    get_best_weights(Path("logs/2023-07-05_22-52-08/0"))
