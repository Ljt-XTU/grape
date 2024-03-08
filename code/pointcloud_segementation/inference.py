# Author:ljt
# Time:2023/8/11 8:20
# Illustration:
import hydra
import os
import copy
import numpy as np
import torch

from omegaconf import OmegaConf
from class_store.inference_class import Inferencer
#data for inference

@hydra.main(config_path="conf_infer_grape", config_name="config")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    print("Configuration Done!")
    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))

    rawdata_dir = '../../../data/shapenet/raw/02222222/'
    trainer = Inferencer(cfg)
    trainer.inference_in_test()
    #trainer.load_raw_data(rawdata_dir)
    #trainer.inference_on_rawdata()
    # # https://github.com/facebookresearch/hydra/issues/440
    #GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()