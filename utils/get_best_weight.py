import os
from pathlib import Path


def get_best_weights(weight_dir: Path) -> Path | None:
    """Get best weights. expected file format: ${epoch}_${val_loss}.h5"""
    best_weight_path = None
    for weight_path in weight_dir.glob("*.h5"):
        if best_weight_path is None:
            best_weight_path = weight_path
        elif int(weight_path.split(os.sep)[-1].split("_")[0]) > int(
            best_weight_path.split(os.sep)[-1].split("_")[0]
        ):
            best_weight_path = weight_path
    print(f"\033[94m Best weight: {best_weight_path}\033[0m")
    return best_weight_path
