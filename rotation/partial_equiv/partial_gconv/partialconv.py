import copy
import math

# typing
from typing import Tuple

import torch
import torch.nn.functional as torch_F
from omegaconf import OmegaConf

import partial_equiv.ck as ck
from partial_equiv.ck.mlp import MLP

# project
import partial_equiv.general.utils as g_utils
from partial_equiv.groups import Group, SamplingMethods
from partial_equiv.partial_gconv.conv import LiftingConvBase
from partial_equiv.partial_gconv.module import Filter, MLPFilter
from partial_equiv.partial_gconv.probconv import ProbConv

class PartialLiftingConv(LiftingConvBase, ProbConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group: Group,
        base_group_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
    ):
        conv_type = "lifting"
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_type=conv_type,
            group=group,
            base_group_config=base_group_config,
            kernel_config=kernel_config,
            conv_config=conv_config,
        )
        self.filter_hidden = conv_config.filter_hidden

        self.version = conv_config.version
        if self.version == "v1.0":
            eps = False
        else:
            eps = True
        self.probs = MLPFilter(in_channels, 10, self.filter_hidden, "v1.5", eps)
        self.entropy = 0

    def forward(self, *args, **kwargs):
        if self.version == "v1.0":
            return self._v1_0(*args, **kwargs)
        elif self.version == "v1.1":
            return self._v1_1(*args, **kwargs)
        else:
            raise NotImplementedError

    def _v1_0(self, x):
        """
        :param x: Input: Function on Rd of size [batch_size, in_channels, * ]
        :return: Output: Function on the group of size [batch_size, out_channels, no_lifting_samples, * ]
        """

        # Get input grid of conv kernel
        rel_pos = self.handle_rel_positions_on_Rd(x)

        # Define the number of independent samples to take from the group. If self.sample_per_batch_element == True, and
        # self.group_sampling_method == RANDOM, then batch_size independent self.group_no_samples samples from the group
        # will be taken for each batch element. Otherwise, the same rotations are used across batch elements.

        if (
            self.lift_sample_per_batch_element
            # and self.group_sampling_method == SamplingMethods.RANDOM
        ):
            no_samples = x.shape[0]  # batch size
        else:
            no_samples = 1

        probs = self.probs(x)
        # probs = torch.sigmoid(probs)
        probs = probs.abs()
        # Sample no_group_elements from the group, batch_size times.
        g_elems, filter = self.group.sample_from_stabilizer_given_x(
            no_samples=no_samples,
            no_elements=self.group_no_samples,
            method=self.group_sampling_method,
            device=x.device,
            partial_equivariance=self.lift_partial_equiv,  # We always lift first to the group.
            probs=probs,
        )  # [no_samples, output_g_no_elems]
        # mask [no_samples, output_g_no_elems]

        self.entropy = torch.log(torch.minimum(probs+1e-9,torch.ones_like(probs))).mean()
        self.probs_all = probs

        output_g_no_elems = g_elems.shape[-1]

        # For R2, we don't need to go to the LieAlgebra. We can parameterize the kernel directly on the input space
        acted_rel_pos = self.group.left_action_on_Rd(
            self.group.inv(g_elems.view(-1, self.group.dimension_stabilizer)), rel_pos
        )

        self.acted_rel_pos = acted_rel_pos

        # Get the kernel
        conv_kernel = self.kernelnet(acted_rel_pos).view(
            no_samples * output_g_no_elems,
            self.out_channels,
            self.in_channels,
            *acted_rel_pos.shape[-2:],
        )

        # Filter values outside the sphere
        mask = (torch.norm(acted_rel_pos[:, :2], dim=1) > 1.0).unsqueeze(1).unsqueeze(2)
        mask = mask.expand_as(conv_kernel)
        conv_kernel[mask] = 0

        # Fiter values outside the p(u|x)
        conv_kernel[filter.view(-1)] = 0

        self.conv_kernel = conv_kernel

        # Reshape conv_kernel for convolution
        conv_kernel = conv_kernel.view(
            no_samples * output_g_no_elems * self.out_channels,
            self.in_channels,
            *acted_rel_pos.shape[-2:],
        )

        # Convolution:
        if no_samples == 1:
            out = torch_F.conv2d(
                input=x,
                weight=conv_kernel,
                padding=self.padding,
                groups=1,
            )
        else:
            # If each batch element has independent rotations, we need to perform them as grouped convolutions.
            out = torch_F.conv2d(
                input=x.view(1, -1, *x.shape[2:]),  # Shift batch_size to input
                weight=conv_kernel,
                padding=self.padding,
                groups=no_samples,
            )

        out = out.view(-1, output_g_no_elems, self.out_channels, *out.shape[-2:]).transpose(1, 2)

        # Add bias:
        if self.bias is not None:
            out = out + self.bias.view(1, -1, *(1,) * (len(out.shape) - 2))

        return out, g_elems


    def _v1_1(self, x):
        """
        :param x: Input: Function on Rd of size [batch_size, in_channels, * ]
        :return: Output: Function on the group of size [batch_size, out_channels, no_lifting_samples, * ]
        """

        # Get input grid of conv kernel
        rel_pos = self.handle_rel_positions_on_Rd(x)

        # Define the number of independent samples to take from the group. If self.sample_per_batch_element == True, and
        # self.group_sampling_method == RANDOM, then batch_size independent self.group_no_samples samples from the group
        # will be taken for each batch element. Otherwise, the same rotations are used across batch elements.

        if (
            self.lift_sample_per_batch_element
            # and self.group_sampling_method == SamplingMethods.RANDOM
        ):
            no_samples = x.shape[0]  # batch size
        else:
            no_samples = 1

        # Sample no_group_elements from the group, batch_size times.
        g_elems = self.group.sample_from_stabilizer(
            no_samples=no_samples,
            no_elements=self.group_no_samples,
            method=self.group_sampling_method,
            device=x.device,
            partial_equivariance=False,  # We always lift first to the group.
            probs=None,
        )  # [no_samples, output_g_no_elems]
        # mask [no_samples, output_g_no_elems]
        eps = g_elems/(2*math.pi) # (B, k)
        assert len(eps.size()) == 2
        logits = self.probs(x, eps) # (B, k, 1)
        logits = logits.squeeze(-1)
        logw = torch.log_softmax(logits, dim=-1)
        w = torch.exp(logw)

        self.entropy = -(w*logw).sum(-1).mean()
        self.probs_all = w

        output_g_no_elems = g_elems.shape[-1]

        # For R2, we don't need to go to the LieAlgebra. We can parameterize the kernel directly on the input space
        acted_rel_pos = self.group.left_action_on_Rd(
            self.group.inv(g_elems.view(-1, self.group.dimension_stabilizer)), rel_pos
        )

        self.acted_rel_pos = acted_rel_pos

        # Get the kernel
        conv_kernel = self.kernelnet(acted_rel_pos).view(
            no_samples * output_g_no_elems,
            self.out_channels,
            self.in_channels,
            *acted_rel_pos.shape[-2:],
        )

        # Filter values outside the sphere
        mask = (torch.norm(acted_rel_pos[:, :2], dim=1) > 1.0).unsqueeze(1).unsqueeze(2)
        mask = mask.expand_as(conv_kernel)
        conv_kernel[mask] = 0

        self.conv_kernel = conv_kernel

        # Reshape conv_kernel for convolution
        conv_kernel = conv_kernel.view(
            no_samples * output_g_no_elems * self.out_channels,
            self.in_channels,
            *acted_rel_pos.shape[-2:],
        )

        # Convolution:
        if no_samples == 1:
            out = torch_F.conv2d(
                input=x,
                weight=conv_kernel,
                padding=self.padding,
                groups=1,
            )
        else:
            # If each batch element has independent rotations, we need to perform them as grouped convolutions.
            out = torch_F.conv2d(
                input=x.view(1, -1, *x.shape[2:]),  # Shift batch_size to input
                weight=conv_kernel,
                padding=self.padding,
                groups=no_samples,
            )

        out = out.view(-1, output_g_no_elems, self.out_channels, *out.shape[-2:]).transpose(1, 2)

        # Add bias:
        if self.bias is not None:
            out = out + self.bias.view(1, -1, *(1,) * (len(out.shape) - 2))

        B, k = w.shape
        out = w.view(B, 1, k, 1, 1)*out

        return out, g_elems
