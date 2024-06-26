from typing import Dict, Tuple

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, random_split

from .generate_data import generate_colormnist_longtailed, generate_102flower_data, generate_cifar10, generate_stl10

def construct_datasets(
    cfg: OmegaConf,
):
    if cfg.dataset == "Flowers102":
        datasets = generate_102flower_data(size=224, root='data')
        train_set = datasets['train']
        test_set = datasets['test']
        datasets = {"train": train_set, "validation": test_set, "test": test_set}
        cfg.num_classes = 102
    elif cfg.dataset == "MNIST":
        train_set, test_set = generate_colormnist_longtailed(datapath='data')
        datasets = {"train": train_set, "validation": test_set, "test": test_set}
        cfg.num_classes = 30
    elif cfg.dataset == "CIFAR10":
        training_set, validation_set, test_set = generate_cifar10(cfg)
        datasets = {"train": training_set, "validation": validation_set, "test": test_set}
        cfg.num_classes = 10
    elif cfg.dataset == "STL10":
        training_set, test_set = generate_stl10(cfg)
        datasets = {"train": training_set, "validation": test_set, "test": test_set}
        cfg.num_classes = 10
    else:
        raise ValueError(f"Dataset {cfg.dataset} not recognized.")
    
    return datasets


def construct_dataloaders(
    cfg,
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

    return dataloaders # dictionary of dataloaders {"train": train_loader, "validation": val_loader , "test": test_loader}

