import numpy as np
import torch
import torch.nn as nn
import random
import os

import wandb
import hydra
from omegaconf import OmegaConf

from color.model_constructor import construct_model
from color.dataset_constructor import construct_dataloaders
from color import utils, trainer, tester
try:
    from color.ada_aug.search import AdaAug
except ModuleNotFoundError as e:
    print("AdaAug has ERROR!")

# @hydra.main(version_base=None, config_path="cfg", config_name="config.yaml")
def main_color(cfg: OmegaConf):
    set_manual_seed(cfg.seed)

    wandb.init(
        project=cfg.wandb.project,
        config=utils.flatten_configdict(cfg),
        entity=cfg.wandb.entity,
        settings=wandb.Settings(start_method="thread"),
        mode=cfg.wandb.mode
    )
    # wandb.define_metric("accuracy_train", summary="max")
    wandb.define_metric("accuracy_valid", summary="max")
    
    print("GPU's available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    if (cfg.device == "cuda") and torch.cuda.is_available():
        cfg.device = "cuda:0"
    else:
        cfg.device = "cpu"

    dataloaders = construct_dataloaders(cfg)

    cfg.train_length = len(dataloaders["train"].dataset)
    model = construct_model(cfg)
    model = model.to(cfg.device)

    if cfg.pretrained:
        path = cfg.pretrained
        model.load_state_dict(
            torch.load(path, map_location=cfg.device)["model"]
        )

    # Train AdaAug
    if cfg.ada_aug.do:
        print("Training AdaAug")
        
        print("AdaAug training complete")

    # Train the model
    if cfg.train.do:
        trainer.train(model, dataloaders, cfg)

    # Test the model
    tester.test(model, dataloaders, cfg)
    
def set_manual_seed(
    seed: int,
):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

