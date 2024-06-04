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

def get_entropy(model):
    entropy = 0
    for m in model.modules():
        if getattr(m, "entropy", None) is None:
            continue
        entropy += m.entropy
    return entropy

def get_variance(model):
    variance = 0
    for m in model.modules():
        if getattr(m, "variance", None) is None:
            continue
        variance += m.variance
    return variance

def get_last_sample(model):
    samples = None
    for m in model.modules():
        if getattr(m, "samples", None) is None:
            continue
        samples = m.samples.clone()
    return samples

def train(
    model: torch.nn.Module,
    dataloaders: Dict[str, DataLoader],
    cfg: OmegaConf,
    adaaug=None,
):
    # Define optimizer and scheduler
    optimizer = optim.construct_optimizer(model, cfg)
    lr_scheduler = optim.construct_scheduler(optimizer, cfg)
    criterion = torch.nn.CrossEntropyLoss()
    train_function = classification_train if not cfg.model.insta else classification_train_insta

    # Train model
    train_function(
        model,
        criterion,
        optimizer,
        dataloaders,
        lr_scheduler,
        cfg,
        adaaug=adaaug,
    )

    # Save the final model
    save_model_to_wandb(model, optimizer, lr_scheduler, name="final_model")


def classification_train(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloaders: Dict[str, DataLoader],
    lr_scheduler,
    cfg: OmegaConf,
    adaaug=None,
):
    # Training parameters
    epochs = cfg.train.epochs
    device = cfg.device

    # Save best performing weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')

    # bin for group samples
    samples_list = []
    samples_label_list = []

    # iterate over epochs
    for epoch in tqdm(range(epochs)):
        model.train()

        # Accumulate accuracy and loss
        running_loss = 0
        running_corrects = 0
        total = 0

        # iterate over data
        for data in dataloaders["train"]:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = adaaug(inputs, mode='exploit') if adaaug is not None else inputs

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            if cfg.model.variational:
                entropy = get_entropy(model)
                loss -= cfg.train.lamda*entropy
                variance = get_variance(model)
                loss -= cfg.train.lamda2*variance
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()
            total += labels.size(0)

        # statistics of the epoch
        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        # print(
        #     f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        # print(datetime.datetime.now())

        # log statistics of the epoch
        metrics = {
            "train/accuracy": epoch_acc,
            "train/loss": epoch_loss,
            "train/epoch": epoch + 1,
        }
        wandb.log(
            metrics,
        )

        if (epoch+1) % cfg.train.valid_every==0:
            model.eval()
            with torch.no_grad():
                running_loss = 0
                running_corrects = 0
                total = 0
                samples = None
                samples_label = None
                for data in dataloaders["validation"]:
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += (preds == labels).sum().item()
                    total += labels.size(0)

                    if cfg.model.variational and torch.any(labels == 9):
                        samples = get_last_sample(model)[labels==9][0]
                        samples_label = labels[labels==9][0]

                epoch_loss_valid = running_loss / total
                epoch_acc_valid = running_corrects / total

                if cfg.model.variational:
                    samples_list.append(samples)
                    samples_label_list.append(samples_label)

            # log statistics of the epoch
            metrics = {
                "valid/accuracy": epoch_acc_valid,
                "valid/loss": epoch_loss_valid,
                "valid/epoch": epoch + 1,
            }
            wandb.log(
                metrics,
            )
            print(
                f"Validation Loss: {epoch_loss_valid:.4f} Acc: {epoch_acc_valid:.4f}")
            print(datetime.datetime.now())

            # If better validation accuracy, replace best weights and compute the test performance
            if epoch_acc_valid > best_acc or (epoch_acc_valid == best_acc and epoch_loss_valid <= best_loss):
                best_acc = epoch_acc_valid
                best_loss = epoch_loss_valid

                best_model_wts = copy.deepcopy(model.state_dict())
                save_model_to_wandb(
                    model, optimizer, lr_scheduler, epoch=epoch + 1)

                # Log best results so far and the weights of the model.
                wandb.run.summary["best_val_accuracy"] = best_acc
                wandb.run.summary["best_val_loss"] = best_loss

        lr_scheduler.step()

    # Report best results
    print("Best Val Acc: {:.4f}".format(best_acc))
    # Load best model weights
    model.load_state_dict(best_model_wts)

    if cfg.model.variational:
        for l, s in zip(samples_label_list, samples_list):
            print(f"{l},,{','.join([str(x) for x in s.tolist()])}")

    # Return model
    return model

def classification_train_insta(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloaders: Dict[str, DataLoader],
    lr_scheduler,
    cfg: OmegaConf,
    adaaug=None,
):
    # Training parameters
    epochs = cfg.train.epochs
    device = cfg.device
    lambda_entropy = cfg.model.insta_params.lambda_entropy

    # Save best performing weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')

    # iterate over epochs
    for epoch in tqdm(range(epochs)):
        model.train()

        # Accumulate accuracy and loss
        running_loss = 0
        running_entropy = 0
        running_corrects = 0
        total = 0

        # iterate over data
        for data in dataloaders["train"]:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = adaaug(inputs, mode='exploit') if adaaug is not None else inputs

            optimizer.zero_grad()

            outputs, inv_param = model(inputs)
            # outputs, inv_param = model.output_with_param(inputs)
            entropy = utils.entropy_param(inv_param)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels) - lambda_entropy * entropy.mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_entropy += entropy.sum().item()
            running_corrects += (preds == labels).sum().item()
            total += labels.size(0)


        # statistics of the epoch
        epoch_loss = running_loss / total
        epoch_entropy = running_entropy / total
        epoch_acc = running_corrects / total
        print(
            f"Train Loss: {epoch_loss:.4f} Entropy: {epoch_entropy:.4f} Acc: {epoch_acc:.4f}")
        print(datetime.datetime.now())

        if epoch_entropy < cfg.model.insta_params.h_min:
            lambda_entropy *= 1.2
        elif epoch_entropy > cfg.model.insta_params.h_max:
            lambda_entropy /= 1.2

        # log statistics of the epoch
        metrics = {
            "train/accuracy": epoch_acc,
            "train/entropy": epoch_entropy,
            "train/loss": epoch_loss,
            "train/epoch": epoch + 1,
        }
        wandb.log(
            metrics,
        )

        if (epoch+1) % cfg.train.valid_every==0:
            model.eval()
            with torch.no_grad():
                running_loss = 0
                running_corrects = 0
                total = 0
                for data in dataloaders["validation"]:
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

                epoch_loss_valid = running_loss / total
                epoch_entropy = running_entropy / total
                epoch_acc_valid = running_corrects / total

            # log statistics of the epoch
            metrics = {
                "valid/accuracy": epoch_acc_valid,
                "valid/entropy": epoch_entropy,
                "valid/loss": epoch_loss_valid,
                "valid/epoch": epoch + 1,
            }
            wandb.log(
                metrics,
            )
            print(
                f"Validation Loss: {epoch_loss_valid:.4f} Entropy: {epoch_entropy:.4f} Acc: {epoch_acc_valid:.4f}")
            print(datetime.datetime.now())

            # If better validation accuracy, replace best weights and compute the test performance
            if epoch_acc_valid > best_acc or (epoch_acc_valid == best_acc and epoch_loss_valid <= best_loss):
                best_acc = epoch_acc_valid
                best_loss = epoch_loss_valid

                best_model_wts = copy.deepcopy(model.state_dict())
                save_model_to_wandb(
                    model, optimizer, lr_scheduler, epoch=epoch + 1)

                # Log best results so far and the weights of the model.
                wandb.run.summary["best_val_accuracy"] = best_acc
                wandb.run.summary["best_val_loss"] = best_loss

        lr_scheduler.step()

    # Report best results
    print("Best Val Acc: {:.4f}".format(best_acc))
    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Return model
    return model

def save_model_to_wandb(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler=None,
    name: str = "model",
    epoch=None,
):
    filename = f"{name}.pt"
    if epoch is not None:
        filename = "checkpoint.pt"
    path = os.path.join(wandb.run.dir, filename)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "epoch": epoch,
        },
        path,
    )
    # Call wandb to save the object, syncing it directly
    wandb.save(path)
    wandb.summary["save_path"] = path
    print(f"Saved at {path}")
