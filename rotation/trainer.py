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
from hydra import utils
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from .models.probckresnet import ProbCKResBlock, ProbCKResNet

from . import optim
from .partial_equiv import general as gral
from .partial_equiv import partial_gconv as partial_gconv
from .partial_equiv.partial_gconv.conv import PartConv
from .partial_equiv.partial_gconv.probconv import ProbConv, ProbGroupConv, ProbLiftingConv
from .partial_equiv.partial_gconv.expconv import ExpConv
from .partial_equiv.partial_gconv.varconv import VarConv

# project
from . import tester
from .globals import IMG_DATASETS
from .partial_equiv import ck


def train(
    model: torch.nn.Module,
    dataloaders: Dict[str, DataLoader],
    cfg: OmegaConf,
):

    # Define criterion and training function
    if cfg.dataset in IMG_DATASETS:
        criterion = torch.nn.CrossEntropyLoss().to(cfg.device)
        if cfg.net.type == 'InstaCKResNet':
            train_function = classification_train_insta
        else:
            train_function = classification_train
    else:
        raise NotImplementedError(
            f"No training criterion and training function found for dataset {cfg.dataset}.")

    # Define optimizer and scheduler
    optimizer = optim.construct_optimizer(model, cfg)
    total_opt = optimizer
    if isinstance(optimizer, tuple):
        optimizer, suboptimizer = optimizer
    lr_scheduler = optim.construct_scheduler(optimizer, cfg)

    # Train model
    train_function(
        model,
        criterion,
        total_opt,
        dataloaders,
        lr_scheduler,
        cfg,
    )

    # Save the final model
    save_model_to_wandb(model, optimizer, lr_scheduler, name="final_model")

    if cfg.debug:
        # If running on debug mode, also save the model locally.
        torch.save(model.state_dict(), os.path.join(
            utils.get_original_cwd(), "saved/model.pt"))

    return


def classification_train(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloaders: Dict[str, DataLoader],
    lr_scheduler,
    cfg: OmegaConf,
):
    suboptimizer = None
    if isinstance(optimizer, tuple):
        optimizer, suboptimizer = optimizer
    # Construct weight_regularizer
    weight_regularizer = gral.nn.loss.LnLoss(
        weight_loss=cfg.train.weight_decay, norm_type=2)

    # Construct decay_regularizer
    mono_decay_loss = gral.nn.loss.MonotonicPartialEquivarianceLoss(
        weight_loss=cfg.train.monotonic_decay_loss,
    )

    # equivariance loss for localliftingconv
    equiv_loss = gral.nn.loss.EquivLoss()

    # Training parameters
    epochs = cfg.train.epochs
    device = cfg.device

    # Save best performing weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 999

    # Counter for epochs without improvement
    epochs_no_improvement = 0
    max_epochs_no_improvement = 100

    # iterate over epochs
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("-" * 30)
        # Print current learning rate
        for param_group in optimizer.param_groups:
            print("Learning Rate: {}".format(param_group["lr"]))
        print("-" * 30)
        # log learning_rate of the epoch
        wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=epoch + 1)

        # Each epoch consist of training and validation
        for phase in ["train", "validation"]:

            if phase == "train":
                model.train()
            else:
                model.eval()

            # Accumulate accuracy and loss
            running_loss = 0
            running_entropy = 0
            running_corrects = 0
            total = 0

            # iterate over data
            for data in tqdm(dataloaders[phase], desc=f"Epoch {epoch} / {phase}"):

                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                _labels = labels.clone()

                optimizer.zero_grad()
                if suboptimizer is not None:
                    suboptimizer.zero_grad()
                train = phase == "train"

                with torch.set_grad_enabled(train):
                    # Fwrd pass:
                    outputs = model(inputs)
                    # Rinputs = torch.rot90(inputs,k=2)
                    # Routputs = model(Rinputs)
                    # if torch.any(labels==0) and torch.any(labels==6):
                    #     mask0 = labels == 0
                    #     mask6 = labels == 6
                    #     outputs0 = torch.argmax(outputs[mask0][0], dim=-1).detach().cpu()
                    #     outputs6 = torch.argmax(outputs[mask6][0], dim=-1).detach().cpu()
                    #     Routputs0 = torch.argmax(Routputs[mask0][0], dim=-1).detach().cpu()
                    #     Routputs6 = torch.argmax(Routputs[mask6][0], dim=-1).detach().cpu()
                    #     print(" outputs", outputs0, outputs6, flush=True)
                    #     print("Routputs", Routputs0, Routputs6, flush=True)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # Regularization:
                    if cfg.train.weight_decay > 0.0 and cfg.train.optimizer != "AdamW":
                        loss = loss + weight_regularizer(model)
                    if cfg.train.monotonic_decay_loss > 0.0:
                        loss = loss + mono_decay_loss(model)
                    if cfg.train.equiv_loss > 0.:
                        loss = loss + cfg.train.equiv_loss * \
                            equiv_loss(model, inputs)

                    if phase == "train":

                        entropy = torch.zeros_like(loss)
                        for m in model.modules():
                            if isinstance(m, (ProbConv, ExpConv, VarConv)):
                                entropy += m.entropy
                        loss += -cfg.train.lamda*entropy
                        # Backward pass
                        loss.backward()

                        # Gradient clip
                        if cfg.train.gradient_clip != 0.0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                cfg.train.gradient_clip,
                            )

                        # Optimizer step
                        optimizer.step()
                        if suboptimizer is not None:
                            suboptimizer.step()

                        # update the lr_scheduler
                        if isinstance(
                            lr_scheduler,
                            (
                                torch.optim.lr_scheduler.CosineAnnealingLR,
                                gral.lr_scheduler.LinearWarmUp_LRScheduler,
                            ),
                        ):
                            lr_scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_entropy += entropy.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                total += labels.size(0)

            # statistics of the epoch
            epoch_loss = running_loss / total
            epoch_entropy = running_entropy / total
            epoch_acc = running_corrects / total
            print(
                f"{phase} Loss: {epoch_loss:.4f} Entropy: {epoch_entropy:.4f} Acc: {epoch_acc:.4f}")
            print(datetime.datetime.now())

            # log statistics of the epoch
            metrics = {
                "accuracy" + "_" + phase: epoch_acc,
                "loss" + "_" + phase: epoch_loss,
            }
            metrics[f"entropy_{phase}"] = epoch_entropy
            wandb.log(
                metrics,
                step=epoch + 1,
            )

            # If better validation accuracy, replace best weights and compute the test performance
            if phase == "validation" and epoch_acc >= best_acc:

                # Updates to the weights will not happen if the accuracy is equal but loss does not diminish
                if (epoch_acc == best_acc) and (epoch_loss > best_loss):
                    pass
                else:
                    best_acc = epoch_acc
                    best_loss = epoch_loss

                    best_model_wts = copy.deepcopy(model.state_dict())
                    save_model_to_wandb(
                        model, optimizer, lr_scheduler, epoch=epoch + 1)

                    # Log best results so far and the weights of the model.
                    wandb.run.summary["best_val_accuracy"] = best_acc
                    wandb.run.summary["best_val_loss"] = best_loss

                    # Clean CUDA Memory
                    del inputs, outputs, labels
                    torch.cuda.empty_cache()

                    # # Perform test and log results
                    # if cfg.dataset in ["rotMNIST", "CIFAR10", "CIFAR10", "CIFAR100", "PCam"]:
                    #     test_acc = tester.test(model, dataloaders["test"], cfg)
                    # else:
                    #     test_acc = best_acc
                    # wandb.run.summary["best_test_accuracy"] = test_acc
                    # wandb.log(
                    #     {"accuracy_test": test_acc},
                    #     step=epoch + 1,
                    # )

                    # Reset counter of epochs without progress
                    epochs_no_improvement = 0

            elif phase == "validation" and epoch_acc < best_acc:
                # Otherwise, increase counter
                epochs_no_improvement += 1

            # if phase == "validation" and (cfg.conv.partial_equiv or cfg.conv.lift_partial_equiv):
            if phase == "validation":
                # Log to wandb and print
                log_and_print_probabilities(
                    model, step=epoch + 1, labels=_labels)

            # Log omega_0
            if cfg.kernel.type == "SIREN" and cfg.kernel.learn_omega0:
                log_and_print_omega_0s(model, epoch + 1, log_to_wandb=True)

            # Update scheduler
            if phase == "validation" and isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(epoch_acc)

        # Update scheduler
        if isinstance(
            lr_scheduler,
            (
                torch.optim.lr_scheduler.MultiStepLR,
                torch.optim.lr_scheduler.ExponentialLR,
            ),
        ):
            lr_scheduler.step()
        print()

        #  Check how many epochs without improvement have passed, and, if required, stop training.
        if epochs_no_improvement == max_epochs_no_improvement:
            print(
                f"Stopping training due to {epochs_no_improvement} epochs of no improvement in validation accuracy.")
            break

    # Report best results
    print("Best Val Acc: {:.4f}".format(best_acc))
    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Print learned w0s
    if cfg.kernel.type == "SIREN" and cfg.kernel.learn_omega0:
        log_and_print_omega_0s(model, step=-1, log_to_wandb=False)

    # Return model
    return model


def save_model_to_wandb(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
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


def log_and_print_probabilities(
    model: torch.nn.Module,
    step: int,
    labels: torch.Tensor
):
    probs = {}
    counter = 0
    for m in model.modules():
        if isinstance(m, PartConv):
            # get
            prob = m.probs.detach().cpu()
            # print
            print(prob, flush=True)
            # add to probs dict
            probs[str(counter)] = prob
            # increase counter
            counter += 1
        elif isinstance(m, (ProbConv, ExpConv, VarConv)):
            if getattr(m, "probs_all", None) is None:
                print("No probs_all")
                counter += 1
                continue
            for digit in range(10):
                prob = m.probs_all.detach().cpu()
                prob = prob[labels == digit]
                if len(prob) > 0:
                    prob = prob[-1].detach().cpu()
                print(f"prob {digit}", prob, "entropy",
                      m.entropy.detach().cpu(), flush=True)
                if probs.get(str(counter)) is None:
                    probs[str(counter)] = dict()
                else:
                    probs[str(counter)][str(digit)] = prob
            # increase counter
            counter += 1
    # log probs:
    wandb.log({"probs": probs}, step=step)


def log_and_print_omega_0s(
    model: torch.nn.Module,
    step: int,
    log_to_wandb: bool,
):
    w0s = {}
    counter = 0
    for m in model.modules():
        if isinstance(
            m,
            (
                ck.siren.SIRENLayer1d,
                ck.siren.SIRENLayer2d,
                ck.siren.SIRENLayer3d,
                ck.siren.SIRENLayerNd,
            ),
        ):
            w0 = m.omega_0.detach().cpu().item()
            # print
            print(w0)
            # add to probs dict
            w0s[str(counter)] = w0
            # increase counter
            counter += 1
    if log_to_wandb:
        # log probs:
        wandb.log({"w0s": w0s}, step=step)


def classification_train_insta(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloaders: Dict[str, DataLoader],
    lr_scheduler,
    cfg: OmegaConf,
):
    suboptimizer = None
    if isinstance(optimizer, tuple):
        optimizer, suboptimizer = optimizer
    # Construct weight_regularizer
    weight_regularizer = gral.nn.loss.LnLoss(
        weight_loss=cfg.train.weight_decay, norm_type=2)

    # Construct decay_regularizer
    mono_decay_loss = gral.nn.loss.MonotonicPartialEquivarianceLoss(
        weight_loss=cfg.train.monotonic_decay_loss,
    )

    # equivariance loss for localliftingconv
    equiv_loss = gral.nn.loss.EquivLoss()

    # Training parameters
    epochs = cfg.train.epochs
    device = cfg.device

    # Save best performing weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 999

    # Counter for epochs without improvement
    epochs_no_improvement = 0
    max_epochs_no_improvement = 100

    # iterate over epochs
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("-" * 30)
        # Print current learning rate
        for param_group in optimizer.param_groups:
            print("Learning Rate: {}".format(param_group["lr"]))
        print("-" * 30)
        # log learning_rate of the epoch
        wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=epoch + 1)

        # Each epoch consist of training and validation
        for phase in ["train", "validation"]:

            if phase == "train":
                model.train()
            else:
                model.eval()

            # Accumulate accuracy and loss
            running_loss = 0
            running_entropy = 0
            running_corrects = 0
            total = 0

            # iterate over data
            for data in tqdm(dataloaders[phase], desc=f"Epoch {epoch} / {phase}"):

                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                _labels = labels.clone()

                optimizer.zero_grad()
                if suboptimizer is not None:
                    suboptimizer.zero_grad()
                train = phase == "train"

                with torch.set_grad_enabled(train):
                    # Fwrd pass:
                    outputs, entropy = model(inputs)
                    # Rinputs = torch.rot90(inputs,k=2)
                    # Routputs = model(Rinputs)
                    # if torch.any(labels==0) and torch.any(labels==6):
                    #     mask0 = labels == 0
                    #     mask6 = labels == 6
                    #     outputs0 = torch.argmax(outputs[mask0][0], dim=-1).detach().cpu()
                    #     outputs6 = torch.argmax(outputs[mask6][0], dim=-1).detach().cpu()
                    #     Routputs0 = torch.argmax(Routputs[mask0][0], dim=-1).detach().cpu()
                    #     Routputs6 = torch.argmax(Routputs[mask6][0], dim=-1).detach().cpu()
                    #     print(" outputs", outputs0, outputs6, flush=True)
                    #     print("Routputs", Routputs0, Routputs6, flush=True)
                    loss = criterion(outputs, labels) - cfg.train.lamda * entropy
                    _, preds = torch.max(outputs, 1)

                    # Regularization:
                    if cfg.train.weight_decay > 0.0 and cfg.train.optimizer != "AdamW":
                        loss = loss + weight_regularizer(model)
                    if cfg.train.monotonic_decay_loss > 0.0:
                        loss = loss + mono_decay_loss(model)
                    if cfg.train.equiv_loss > 0.:
                        loss = loss + cfg.train.equiv_loss * \
                            equiv_loss(model, inputs)

                    if phase == "train":

                        entropy = torch.zeros_like(loss)
                        for m in model.modules():
                            if isinstance(m, (ProbConv, ExpConv, VarConv)):
                                entropy += m.entropy
                        loss += -cfg.train.lamda*entropy
                        # Backward pass
                        loss.backward()

                        # Gradient clip
                        if cfg.train.gradient_clip != 0.0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                cfg.train.gradient_clip,
                            )

                        # Optimizer step
                        optimizer.step()
                        if suboptimizer is not None:
                            suboptimizer.step()

                        # update the lr_scheduler
                        if isinstance(
                            lr_scheduler,
                            (
                                torch.optim.lr_scheduler.CosineAnnealingLR,
                                gral.lr_scheduler.LinearWarmUp_LRScheduler,
                            ),
                        ):
                            lr_scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_entropy += entropy.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                total += labels.size(0)

            # statistics of the epoch
            epoch_loss = running_loss / total
            epoch_entropy = running_entropy / total
            epoch_acc = running_corrects / total
            print(
                f"{phase} Loss: {epoch_loss:.4f} Entropy: {epoch_entropy:.4f} Acc: {epoch_acc:.4f}")
            print(datetime.datetime.now())

            # log statistics of the epoch
            metrics = {
                "accuracy" + "_" + phase: epoch_acc,
                "loss" + "_" + phase: epoch_loss,
            }
            metrics[f"entropy_{phase}"] = epoch_entropy
            wandb.log(
                metrics,
                step=epoch + 1,
            )

            # If better validation accuracy, replace best weights and compute the test performance
            if phase == "validation" and epoch_acc >= best_acc:

                # Updates to the weights will not happen if the accuracy is equal but loss does not diminish
                if (epoch_acc == best_acc) and (epoch_loss > best_loss):
                    pass
                else:
                    best_acc = epoch_acc
                    best_loss = epoch_loss

                    best_model_wts = copy.deepcopy(model.state_dict())
                    save_model_to_wandb(
                        model, optimizer, lr_scheduler, epoch=epoch + 1)

                    # Log best results so far and the weights of the model.
                    wandb.run.summary["best_val_accuracy"] = best_acc
                    wandb.run.summary["best_val_loss"] = best_loss

                    # Clean CUDA Memory
                    del inputs, outputs, labels
                    torch.cuda.empty_cache()

                    # # Perform test and log results
                    # if cfg.dataset in ["rotMNIST", "CIFAR10", "CIFAR10", "CIFAR100", "PCam"]:
                    #     test_acc = tester.test(model, dataloaders["test"], cfg)
                    # else:
                    #     test_acc = best_acc
                    # wandb.run.summary["best_test_accuracy"] = test_acc
                    # wandb.log(
                    #     {"accuracy_test": test_acc},
                    #     step=epoch + 1,
                    # )

                    # Reset counter of epochs without progress
                    epochs_no_improvement = 0

            elif phase == "validation" and epoch_acc < best_acc:
                # Otherwise, increase counter
                epochs_no_improvement += 1

            # if phase == "validation" and (cfg.conv.partial_equiv or cfg.conv.lift_partial_equiv):
            if phase == "validation":
                # Log to wandb and print
                log_and_print_probabilities(
                    model, step=epoch + 1, labels=_labels)

            # Log omega_0
            if cfg.kernel.type == "SIREN" and cfg.kernel.learn_omega0:
                log_and_print_omega_0s(model, epoch + 1, log_to_wandb=True)

            # Update scheduler
            if phase == "validation" and isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(epoch_acc)

        # Update scheduler
        if isinstance(
            lr_scheduler,
            (
                torch.optim.lr_scheduler.MultiStepLR,
                torch.optim.lr_scheduler.ExponentialLR,
            ),
        ):
            lr_scheduler.step()
        print()

        #  Check how many epochs without improvement have passed, and, if required, stop training.
        if epochs_no_improvement == max_epochs_no_improvement:
            print(
                f"Stopping training due to {epochs_no_improvement} epochs of no improvement in validation accuracy.")
            break

    # Report best results
    print("Best Val Acc: {:.4f}".format(best_acc))
    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Print learned w0s
    if cfg.kernel.type == "SIREN" and cfg.kernel.learn_omega0:
        log_and_print_omega_0s(model, step=-1, log_to_wandb=False)

    # Return model
    return model