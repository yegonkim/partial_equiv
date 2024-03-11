# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch
import math

# from models import CKResNet
from partial_equiv.partial_gconv.conv import GroupConv, LiftingConv, PointwiseGroupConv
from partial_equiv.partial_gconv.module import get_rotation_matrix_from_radian
from partial_equiv.partial_gconv.probconv import LocalLiftingConv, ProbGroupConv, ProbLiftingConv
import torchvision


class EquivLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model: torch.nn.Module, x: torch.Tensor):
        loss = 0.
        bs = x.size(0)
        u = torch.rand(1).to(x)
        angle = 360*u.item()
        R = get_rotation_matrix_from_radian(2*math.pi*u) # (2,2)
        Rx = torchvision.transforms.functional.rotate(
            x, angle, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        _x = torch.cat([x, Rx])
        loss = 0.
        for m in model.modules():
            if isinstance(m, LocalLiftingConv):
                _, filter, arrow = m.probs(_x)
                if arrow is None:
                    continue
                Rarrow = torch.einsum("ij,bj->bi", R, arrow[:bs])
                equiv = (Rarrow-arrow[bs:]).pow(2).sum(-1).mean()
                inv = (filter[:bs]-filter[bs:]).pow(2).mean()
                loss += equiv + inv
        return loss



class LnLoss(torch.nn.Module):
    def __init__(
        self,
        weight_loss: float,
        norm_type: int,
    ):
        """
        Computes the Ln loss on the CKConv kernels in a CKCNN.
        :param weight_loss: Specifies the weight with which the loss will be summed to the total loss.
        :param norm_type: Type of norm, e.g., 1 = L1 loss, 2 = L2 loss, ...
        """
        super().__init__()
        self.weight_loss = weight_loss
        self.norm_type = norm_type

    def forward(
        self,
        model: torch.nn.Module,
    ):
        loss = 0.0
        # Go through modules that are instances of CKConvs
        for m in model.modules():
            if isinstance(m, (
                LiftingConv,
                ProbLiftingConv,
                GroupConv,
                ProbGroupConv,
                PointwiseGroupConv
            )):
                loss += m.conv_kernel.norm(self.norm_type)

                if m.bias is not None:
                    loss += m.bias.norm(self.norm_type)
        loss = self.weight_loss * loss
        return loss


class MonotonicPartialEquivarianceLoss(torch.nn.Module):
    def __init__(
        self,
        weight_loss: float,
    ):
        """
        Computes the Ln loss on the learned subset values for a partial equivariant network.
        It implicitly penalizes networks whose learned subsets increase during traning.
        """
        super().__init__()
        self.weight_loss = weight_loss

    def forward(
        self,
        model: torch.nn.Module,
    ):
        # if not isinstance(model.module, CKResNet):
        #     raise NotImplementedError(f"Model of type {model.__class__.__name__} not a CKResNet.")

        # Only calculated with partial_equivariant models
        if not model.module.partial_equiv:
            return 0.0

        learned_equivariances = []
        for m in model.modules():
            if isinstance(m, (
                GroupConv,
                LiftingConv,
                ProbGroupConv,
                ProbLiftingConv
                )):
                if m.probs is not None and (m.probs.nelement() != 0):
                    learned_equivariances.append(m.probs)
        # Take the difference between the next element, and the previous.
        # Then check if it's larger than 0. If that;'s the case, then
        # the network is increasing, and thus, must be penalized.
        differences = torch.relu(
            torch.stack(
                [y - x for (x, y) in zip(learned_equivariances[:-1], learned_equivariances[1:])],
                dim=0,
            )
        )
        loss = self.weight_loss * differences.sum()
        return loss
