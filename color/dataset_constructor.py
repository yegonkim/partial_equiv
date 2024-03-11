from typing import Dict, Tuple

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, random_split

from .generate_data import generate_colormnist_longtailed, generate_102flower_data

def construct_datasets(
    cfg: OmegaConf,
):
    if cfg.dataset == "Flowers102":
        datasets = generate_102flower_data(size=224, root='data')
        train_set = datasets['train']
        test_set = datasets['test']
    elif cfg.dataset == "MNIST":
        train_set, test_set = generate_colormnist_longtailed(datapath='data')
    else:
        raise ValueError(f"Dataset {cfg.dataset} not recognized.")
    
    return {"train": train_set, "validation": test_set, "test": test_set}


def construct_dataloaders(
    cfg: OmegaConf,
):
    """
    Construct DataLoaders for the selected dataset
    :return dict("train": train_loader, "validation": val_loader , "test": test_loader)
    """
    datasets = construct_datasets(cfg)

    num_workers = cfg.no_workers
    num_workers = num_workers * torch.cuda.device_count()
    dataloaders = dict()
    for key in datasets:
        dataloaders[key] = DataLoader(
            datasets[key],
            batch_size=cfg.train.batch_size,
            shuffle=True if key == "train" else False,
            num_workers=num_workers,
            pin_memory=False,
        )

    return dataloaders
