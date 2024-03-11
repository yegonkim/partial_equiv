# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 David W. Romero & Robert-Jan Bruintjes
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
#
# Code adapted from https://github.com/rjbruin/flexconv -- MIT License

import copy
import datetime
import os

# typing
from typing import Dict, List

# torch
import torch

# logger
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import optim, utils

def test(
    model: torch.nn.Module,
    dataloaders: Dict[str, DataLoader],
    cfg: OmegaConf,
):
    criterion = torch.nn.CrossEntropyLoss()

    test_function = classification_test if not cfg.model.insta else classification_test_insta

    test_function(
        model,
        criterion,
        dataloaders,
        cfg,
    )

def classification_test(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloaders: Dict[str, DataLoader],
    cfg: OmegaConf,
):
    device = cfg.device
    model.eval()
    
    with torch.no_grad():
        running_loss = 0
        running_corrects = 0
        total = 0
        for data in dataloaders["test"]:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total

    # log statistics of the epoch
    metrics = {
        "accuracy_test": epoch_acc,
        "loss_test": epoch_loss,
    }
    wandb.log(metrics)
    print(f"Accuracy of the network on test samples: {(100 * epoch_acc)}%")

def classification_test_insta(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloaders: Dict[str, DataLoader],
    cfg: OmegaConf,
):
    device = cfg.device
    lambda_entropy = cfg.model.insta_params.lambda_entropy
    model.eval()

    with torch.no_grad():
        running_loss = 0
        running_corrects = 0
        total = 0
        for data in dataloaders["test"]:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs, inv_param = model(inputs)
            entropy = utils.entropy_param(inv_param)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels) - lambda_entropy * entropy.mean()

            running_loss += loss.item() * inputs.size(0)
            running_entropy += entropy.sum().item()
            running_corrects += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total

    # log statistics of the epoch
    metrics = {
        "accuracy_test": epoch_acc,
        "loss_test": epoch_loss,
    }
    wandb.log(metrics)
    print(f"Accuracy of the network on test samples: {(100 * epoch_acc)}%")
