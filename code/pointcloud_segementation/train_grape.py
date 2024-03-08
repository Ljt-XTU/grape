# Author:ljt
# Time:2023/8/27 10:40
# Illustration:
import os
import hydra
import copy
import logging
from omegaconf import OmegaConf
from class_store.mytrainer import MyTrainer

log = logging.getLogger(__name__)

def debug(trainer):
    train_dataset=trainer._dataset.train_dataloader


@hydra.main(config_path="conf_grape", config_name="config")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    print("Configuration Done!")
    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))
    trainer = MyTrainer(cfg)
    #debug(trainer)
    trainer.train()
    return 0


if __name__ == "__main__":
    main()
