# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# torch
import copy
import math

# typing
from typing import Tuple

import torch
import torch.nn.functional as torch_F
from omegaconf import OmegaConf

import partial_equiv.ck as ck

# project
import partial_equiv.general.utils as g_utils
from partial_equiv.groups import Group, SamplingMethods
from partial_equiv.partial_gconv.conv import GroupConvBase
from partial_equiv.partial_gconv.probconv import ProbConv

class ExpConv():
    pass

class ExpGroupConv(GroupConvBase, ExpConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group: Group,
        base_group_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
    ):
        conv_type = "group"
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_type=conv_type,
            group=group,
            base_group_config=base_group_config,
            kernel_config=kernel_config,
            conv_config=conv_config,
        )

    def forward(
        self,
        input_tuple: Tuple[torch.Tensor],
    ):
        """
        :param input_tuple: Consists of two elements:
            (1) The input function in the group [batch_size, in_channels, * ]
            (2) The grid of sampled group elements for which the feature representation is obtained.
        :return: Output: Function on the group of size [batch_size, out_channels, no_lifting_samples, * ]
        """
        # Unpack input tuple
        x, input_g_elems, prev_filter = input_tuple[0], input_tuple[1], input_tuple[2]
        # x: (B, in_ch, prev_no_elem, h, w)
        # input_g_elems (B, prev_no_elems)
        filter_len = prev_filter.size(-1)
        filter, next_filter = torch.split(prev_filter, [1,filter_len-1], dim=-1)

        # Get input grid of conv kernel
        rel_pos = self.handle_rel_positions_on_Rd(x)
        # (2, kernel_size, kernel_size)

        # Define the number of independent samples to take from the group. If self.sample_per_batch_element == True, and
        # self.group_sampling_method == RANDOM, then batch_size independent self.group_no_samples samples from the group
        # will be taken for each batch element. Otherwise, the same rotations are used across batch elements.
        if self.group_sample_per_batch_element and self.group_sampling_method == SamplingMethods.RANDOM:
            no_samples = x.shape[0]  # batch size
        else:
            no_samples = 1

        # If self.group_sample_per_layer == True, and self.group_sampling_method == RANDOM, sample at every layer a
        # different set of rotations. Otherwise, the same group elements are used across the network.
        if (
            self.group_sample_per_layer and self.group_sampling_method == SamplingMethods.RANDOM
        ) or self.partial_equiv:
            # Sample from the group
            g_elems, ood = self.group.sample_from_stabilizer_given_x(
                no_samples=no_samples,
                no_elements=self.group_no_samples,
                method=self.group_sampling_method,
                device=x.device,
                partial_equivariance=self.partial_equiv,
                probs=filter.abs(),
            ) # (B, cur_no_elements)
        else:
            g_elems = input_g_elems # (B, cur_no_elements==prev_no_elements) 

        self.probs_all = filter
        self.entropy = torch.log(torch.minimum(filter, torch.ones_like(filter))).mean()

        output_g_no_elems = g_elems.shape[-1]

        # Act on the grid of positions:

        # Act along the group dimension
        # g_distance
        acted_g_elements = self.group.left_action_on_H(self.group.inv(g_elems), input_g_elems)
        # (B, cur_no_elements, prev_no_elements)

        # Normalize elements to coordinates between -1 and 1
        # normalized g distance
        acted_g_elements = self.group.normalize_g_distance(acted_g_elements).float()
        if self.group.__class__.__name__ == "SE2":
            acted_g_elements = acted_g_elements.unsqueeze(-1)
        # (B, cur_no_elements, prev_no_elements, 1)

        # Act on Rd with the resulting elements
        if acted_g_elements.shape[0] > g_elems.shape[0]:
            _g_elems = g_elems.repeat_interleave(acted_g_elements.shape[0], dim=0)
        else:
            _g_elems = g_elems
        acted_rel_pos_Rd = self.group.left_action_on_Rd(
            self.group.inv(_g_elems.view(-1, self.group.dimension_stabilizer)), rel_pos
        )
        # (B*cur_no_elements, 2, kernel_size, kernel_size)

        # Combine both grids
        # Resulting grid: [no_samples * g_elems, group.dim, self.input_g_elems, kernel_size, kernel_size]
        input_g_no_elems = acted_g_elements.shape[2]
        output_g_no_elems = acted_g_elements.shape[1]
        no_samples = acted_g_elements.shape[0]

        acted_group_rel_pos = torch.cat(
            (
                # Expand the acted rel pos Rd input_g_no_elems times along the "group axis".
                # [no_samples * output_g_no_elems, 2, kernel_size_y, kernel_size_x]
                # +->  [no_samples * output_g_no_elems, 2, input_no_g_elems, kernel_size_y, kernel_size_x]
                acted_rel_pos_Rd.unsqueeze(2).expand(*(-1,) * 2, input_g_no_elems, *(-1,) * 2),
                # Expand the acted g elements along the "spatial axes".
                # [no_samples, output_g_no_elems, input_g_no_elems, self.group.dimension_stabilizer]
                # +->  [no_samples * output_g_no_elems, self.group.dimension_stabilizer, input_no_g_elems, kernel_size_y, kernel_size_x]
                acted_g_elements.transpose(-1, -2)
                .contiguous()
                .view(
                    no_samples * output_g_no_elems,
                    self.group.dimension_stabilizer,
                    input_g_no_elems,
                    1,
                    1,
                )
                .expand(
                    -1,
                    -1,
                    -1,
                    *acted_rel_pos_Rd.shape[2:],
                ),
            ),
            dim=1,
        )
        # torch.cat([(uinv*Rd)_x, (uinv*Rd)_y, uinv*v])
        # (B*cur_no_elements, 2+dimension_stabilizer, prev_no_elements, kernel_size, kernel_size)

        self.acted_rel_pos = acted_group_rel_pos

        # Get the kernel: conv3d [2+dim_stb, prev_no, kernel, kernel]
        # 2+dim_stb -> out_ch*in_ch (no flatten layer in kernelnet)
        conv_kernel = self.kernelnet(acted_group_rel_pos).view(
            no_samples * output_g_no_elems, # batch
            self.out_channels,
            self.in_channels,
            *acted_group_rel_pos.shape[2:],
        )

        # Filter values outside the sphere
        # circle mask
        mask = (torch.norm(acted_group_rel_pos[:, :2], dim=1) > 1.0).unsqueeze(1).unsqueeze(2)
        mask = mask.expand_as(conv_kernel) # expand mask to the same size as conv_kernel
        conv_kernel[mask] = 0

        self.conv_kernel = conv_kernel

        # Reshape conv_kernel for convolution
        conv_kernel = conv_kernel.contiguous().view(
            no_samples * output_g_no_elems * self.out_channels,
            self.in_channels * input_g_no_elems,
            *conv_kernel.shape[-2:],
        ) # (B*cur_no_elem*out_ch, in_ch*prev_no_elem, kernel_size, kernel_size)

        # x: (B, in_ch, prev_no_elem, h, w)
        # Convolution:
        if no_samples == 1:
            out = torch_F.conv2d(
                # (B, in_ch*prev_no_elem, h, w)
                input=x.contiguous().view(-1, self.in_channels * input_g_no_elems, *x.shape[-2:]),
                weight=conv_kernel,
                padding=self.padding,
                groups=1,
            ) # (B, cur_no_elem*out_ch, H, W)
        else:
            # Compute convolution as a 2D convolution
            out = torch_F.conv2d(
                # (1, B*in_ch*prev_no_elem, h, w)
                input=x.contiguous().view(1, -1, *x.shape[3:]),
                weight=conv_kernel,
                padding=self.padding,
                groups=no_samples,
            ) # (1, B*cur_no_elem*out_ch, H, W)
        out = out.view(-1, output_g_no_elems, self.out_channels, *out.shape[2:]).transpose(1, 2)
        # (B, cur_no_elem, out_ch, H, W) tr-> (B, out_ch, cur_no_elem, H, W) 

        # Add bias:
        if self.bias is not None:
            out = out + self.bias.view(1, -1, *(1,) * (len(out.shape) - 2))

        # Filter out of p(u|x)
        out = out.transpose(1,2)
        dims = out.shape
        out = out.view(-1, *dims[2:])
        out[ood.view(-1)] = 0
        out = out.view(*dims)
        out = out.transpose(1,2)
        assert out.size(1) == self.out_channels
        assert out.size(2) == output_g_no_elems
        assert len(out.size()) == 5

        return out, g_elems, next_filter