import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from config.type import TrainConfig
from trainer import train


def as_tuple(*args):
    return tuple(args)


@hydra.main(
    version_base=None,
    config_path="config/",
    config_name="default_config.yaml",
)
def main(config: TrainConfig):
    config = instantiate(config)
    return train(config)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("as_tuple", as_tuple)
    main()
