from dataclasses import dataclass


@dataclass
class DataConfig:
    shuffle: bool
    augment_all_combination: bool
    augment_setting_path_to_save: str
