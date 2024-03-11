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

import rotation.partial_equiv.ck as ck

# project
import rotation.partial_equiv.general.utils as g_utils
from rotation.partial_equiv.groups import Group, SamplingMethods
from rotation.partial_equiv.partial_gconv.conv import GroupConvBase, LiftingConvBase


class VarConv():
    pass


class VarGroupConv(GroupConvBase, VarConv):
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
        self.version = conv_config.version

        # Construct the probability variables given the group.
        probs = self.group.construct_probability_variables(
            self.group_sampling_method,
            base_group_config.no_samples,
        )
        if self.partial_equiv:
            self.probs = torch.nn.Parameter(probs)
        else:
            self.register_buffer("probs", probs)

        # partial equiv encoder
        # self.filter_layer1 = torch.nn.Linear(in_channels, 2*in_channels)
        # self.filter_nonlinear = torch.nn.LeakyReLU(0.1)
        # self.filter_layer2 = torch.nn.Linear(2*in_channels, self.group_no_samples)
        # with torch.no_grad():
        #     self.filter_layer2.weight.fill_(0)
        #     self.filter_layer2.bias.fill_(4)
        if self.version == "v1.0":
            self.filter_q = torch.nn.Linear(in_channels, in_channels)
            self.filter_k = torch.nn.Linear(in_channels, in_channels)
        elif self.version in ["v1.1", "v1.3"]:
            self.filter_q = torch.nn.Linear(1, in_channels)
            self.filter_k = torch.nn.Linear(in_channels, in_channels)
        elif self.version in ["v1.2"]:  # conventional
            self.filter_layer1 = torch.nn.Linear(in_channels, 2*in_channels)
            self.filter_nonlinear = torch.nn.LeakyReLU(0.1)
            self.filter_layer2 = torch.nn.Linear(2*in_channels, 1)
            with torch.no_grad():
                self.filter_layer2.weight.fill_(0)
                self.filter_layer2.bias.fill_(1)
        elif self.version == "v1.4":
            self.filter_layer1 = torch.nn.Linear(in_channels, 2*in_channels)
            self.filter_nonlinear = torch.nn.LeakyReLU(0.1)
            self.filter_layer2 = torch.nn.Linear(
                2*in_channels, self.group_no_samples)
        elif self.version == "v1.5":
            self.filter_layer1 = torch.nn.Linear(in_channels, 2*in_channels)
            self.filter_nonlinear = torch.nn.LeakyReLU(0.1)
            self.filter_layer2 = torch.nn.Linear(2*in_channels, 2)
            with torch.no_grad():
                # self.filter_layer2.bias = torch.nn.Parameter(
                #     torch.tensor([0, 0]).to(self.filter_layer2.bias.data))
                self.filter_layer2.bias.fill_(0)
        elif self.version == "v1.6":  # conventional
            self.filter_conv1 = torch.nn.Conv1d(in_channels, in_channels, 1, 1)
            self.filter_nonlinear = torch.nn.ReLU()
            self.filter_conv2 = torch.nn.Conv1d(in_channels, in_channels, 1, 1)
            self.filter_linear1 = torch.nn.Linear(in_channels, 1)
            with torch.no_grad():
                self.filter_linear1.bias.fill_(5)
        else:
            raise NotImplementedError

    def forward(self, *args, **kwargs):
        if self.version == "v1.0":
            return self._v1_0(*args, **kwargs)
        elif self.version == "v1.1":
            return self._v1_1(*args, **kwargs)
        elif self.version == "v1.2":
            return self._v1_2(*args, **kwargs)
        elif self.version == "v1.3":
            return self._v1_3(*args, **kwargs)
        elif self.version == "v1.4":
            return self._v1_4(*args, **kwargs)
        elif self.version == "v1.5":
            return self._v1_5(*args, **kwargs)
        elif self.version == "v1.6":
            return self._v1_6(*args, **kwargs)
        else:
            raise NotImplementedError

    def _v1_0(
        self,
        input_tuple: Tuple[torch.Tensor, torch.Tensor],
    ):
        """
        :param input_tuple: Consists of two elements:
            (1) The input function in the group [batch_size, in_channels, * ]
            (2) The grid of sampled group elements for which the feature representation is obtained.
        :return: Output: Function on the group of size [batch_size, out_channels, no_lifting_samples, * ]
        """
        # Unpack input tuple
        x, input_g_elems = input_tuple[0], input_tuple[1]
        # x: (B, in_ch, prev_no_elem, h, w)
        # input_g_elems (B, prev_no_elems)

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
            g_elems = self.group.sample_from_stabilizer(
                no_samples=no_samples,
                no_elements=self.group_no_samples,
                method=self.group_sampling_method,
                device=x.device,
                partial_equivariance=self.partial_equiv,
                probs=self.probs,
            )  # (B, cur_no_elements)
        else:
            g_elems = input_g_elems  # (B, cur_no_elements==prev_no_elements)

        # Act on the grid of positions:

        # Act along the group dimension
        # g_distance
        acted_g_elements = self.group.left_action_on_H(
            self.group.inv(g_elems), input_g_elems)
        # (B, cur_no_elements, prev_no_elements)

        # Normalize elements to coordinates between -1 and 1
        # normalized g distance
        acted_g_elements = self.group.normalize_g_distance(
            acted_g_elements).float()
        if self.group.__class__.__name__ == "SE2":
            acted_g_elements = acted_g_elements.unsqueeze(-1)
        # (B, cur_no_elements, prev_no_elements, 1)

        # Act on Rd with the resulting elements
        if acted_g_elements.shape[0] > g_elems.shape[0]:
            _g_elems = g_elems.repeat_interleave(
                acted_g_elements.shape[0], dim=0)
        else:
            _g_elems = g_elems
        acted_rel_pos_Rd = self.group.left_action_on_Rd(
            self.group.inv(
                _g_elems.view(-1, self.group.dimension_stabilizer)), rel_pos
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
                acted_rel_pos_Rd.unsqueeze(2).expand(
                    *(-1,) * 2, input_g_no_elems, *(-1,) * 2),
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
            no_samples * output_g_no_elems,  # batch
            self.out_channels,
            self.in_channels,
            *acted_group_rel_pos.shape[2:],
        )

        # Filter values outside the sphere
        # circle mask
        mask = (torch.norm(acted_group_rel_pos[:, :2], dim=1) > 1.0).unsqueeze(
            1).unsqueeze(2)
        # expand mask to the same size as conv_kernel
        mask = mask.expand_as(conv_kernel)
        conv_kernel[mask] = 0

        self.conv_kernel = conv_kernel

        # Reshape conv_kernel for convolution
        conv_kernel = conv_kernel.contiguous().view(
            no_samples * output_g_no_elems * self.out_channels,
            self.in_channels * input_g_no_elems,
            *conv_kernel.shape[-2:],
        )  # (B*cur_no_elem*out_ch, in_ch*prev_no_elem, kernel_size, kernel_size)

        # x: (B, in_ch, prev_no_elem, h, w)
        # Convolution:
        if no_samples == 1:
            out = torch_F.conv2d(
                # (B, in_ch*prev_no_elem, h, w)
                input=x.contiguous().view(-1, self.in_channels * \
                                          input_g_no_elems, *x.shape[-2:]),
                weight=conv_kernel,
                padding=self.padding,
                groups=1,
            )  # (B, cur_no_elem*out_ch, H, W)
        else:
            # Compute convolution as a 2D convolution
            out = torch_F.conv2d(
                # (1, B*in_ch*prev_no_elem, h, w)
                input=x.contiguous().view(1, -1, *x.shape[3:]),
                weight=conv_kernel,
                padding=self.padding,
                groups=no_samples,
            )  # (1, B*cur_no_elem*out_ch, H, W)
        out = out.view(-1, output_g_no_elems, self.out_channels,
                       *out.shape[2:]).transpose(1, 2)
        # (B, cur_no_elem, out_ch, H, W) tr-> (B, out_ch, cur_no_elem, H, W)

        # Add bias:
        if self.bias is not None:
            out = out + self.bias.view(1, -1, *(1,) * (len(out.shape) - 2))

        z = self.encode(x)
        z = torch.softmax(z, dim=-1)
        out = torch.einsum("bij,bcjhw->bcihw", z, out)
        # self.probs_all = torch.diagonal(z, 0, -2, -1)
        self.probs_all = z
        self.entropy = torch.log(self.probs_all+1e-9).mean()

        return out, g_elems

    def _v1_1(
        self,
        input_tuple: Tuple[torch.Tensor, torch.Tensor],
    ):
        """
        :param input_tuple: Consists of two elements:
            (1) The input function in the group [batch_size, in_channels, * ]
            (2) The grid of sampled group elements for which the feature representation is obtained.
        :return: Output: Function on the group of size [batch_size, out_channels, no_lifting_samples, * ]
        """
        # Unpack input tuple
        x, input_g_elems = input_tuple[0], input_tuple[1]
        # x: (B, in_ch, prev_no_elem, h, w)
        # input_g_elems (B, prev_no_elems)

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
            g_elems = self.group.sample_from_stabilizer(
                no_samples=no_samples,
                no_elements=self.group_no_samples,
                method=self.group_sampling_method,
                device=x.device,
                partial_equivariance=self.partial_equiv,
                probs=self.probs,
            )  # (B, cur_no_elements)
        else:
            g_elems = input_g_elems  # (B, cur_no_elements==prev_no_elements)

        # Act on the grid of positions:

        # Act along the group dimension
        # g_distance
        acted_g_elements = self.group.left_action_on_H(
            self.group.inv(g_elems), input_g_elems)
        # (B, cur_no_elements, prev_no_elements)

        # Normalize elements to coordinates between -1 and 1
        # normalized g distance
        acted_g_elements = self.group.normalize_g_distance(
            acted_g_elements).float()
        if self.group.__class__.__name__ == "SE2":
            acted_g_elements = acted_g_elements.unsqueeze(-1)
        # (B, cur_no_elements, prev_no_elements, 1)

        # Act on Rd with the resulting elements
        if acted_g_elements.shape[0] > g_elems.shape[0]:
            _g_elems = g_elems.repeat_interleave(
                acted_g_elements.shape[0], dim=0)
        else:
            _g_elems = g_elems
        acted_rel_pos_Rd = self.group.left_action_on_Rd(
            self.group.inv(
                _g_elems.view(-1, self.group.dimension_stabilizer)), rel_pos
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
                acted_rel_pos_Rd.unsqueeze(2).expand(
                    *(-1,) * 2, input_g_no_elems, *(-1,) * 2),
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
            no_samples * output_g_no_elems,  # batch
            self.out_channels,
            self.in_channels,
            *acted_group_rel_pos.shape[2:],
        )
        dims = conv_kernel.size()
        score = self.get_score(
            g_elems/(2*math.pi), x
        )
        score = torch.softmax(score, dim=-1)
        self.probs_all = score
        diagprobs = torch.diagonal(score, 0, -2, -1)
        diaglogprobs = torch.log(diagprobs)
        self.entropy = -(diagprobs*diaglogprobs).sum(-1).mean()

        score = score.view(-1, output_g_no_elems, 1, 1, input_g_no_elems, 1, 1)

        conv_kernel = conv_kernel.view(
            no_samples,
            output_g_no_elems,
            self.out_channels,
            self.in_channels,
            input_g_no_elems,
            *acted_group_rel_pos.shape[3:]
        )
        conv_kernel = (score*conv_kernel).view(*dims)

        # Filter values outside the sphere
        # circle mask
        mask = (torch.norm(acted_group_rel_pos[:, :2], dim=1) > 1.0).unsqueeze(
            1).unsqueeze(2)
        # expand mask to the same size as conv_kernel
        mask = mask.expand_as(conv_kernel)
        conv_kernel[mask] = 0

        self.conv_kernel = conv_kernel

        # Reshape conv_kernel for convolution
        conv_kernel = conv_kernel.contiguous().view(
            no_samples * output_g_no_elems * self.out_channels,
            self.in_channels * input_g_no_elems,
            *conv_kernel.shape[-2:],
        )  # (B*cur_no_elem*out_ch, in_ch*prev_no_elem, kernel_size, kernel_size)

        # x: (B, in_ch, prev_no_elem, h, w)
        # Convolution:
        if no_samples == 1:
            out = torch_F.conv2d(
                # (B, in_ch*prev_no_elem, h, w)
                input=x.contiguous().view(-1, self.in_channels * \
                                          input_g_no_elems, *x.shape[-2:]),
                weight=conv_kernel,
                padding=self.padding,
                groups=1,
            )  # (B, cur_no_elem*out_ch, H, W)
        else:
            # Compute convolution as a 2D convolution
            out = torch_F.conv2d(
                # (1, B*in_ch*prev_no_elem, h, w)
                input=x.contiguous().view(1, -1, *x.shape[3:]),
                weight=conv_kernel,
                padding=self.padding,
                groups=no_samples,
            )  # (1, B*cur_no_elem*out_ch, H, W)
        out = out.view(-1, output_g_no_elems, self.out_channels,
                       *out.shape[2:]).transpose(1, 2)
        # (B, cur_no_elem, out_ch, H, W) tr-> (B, out_ch, cur_no_elem, H, W)

        # Add bias:
        if self.bias is not None:
            out = out + self.bias.view(1, -1, *(1,) * (len(out.shape) - 2))

        return out, g_elems

    def _v1_2(
        self,
        input_tuple: Tuple[torch.Tensor, torch.Tensor],
    ):
        """
        :param input_tuple: Consists of two elements:
            (1) The input function in the group [batch_size, in_channels, * ]
            (2) The grid of sampled group elements for which the feature representation is obtained.
        :return: Output: Function on the group of size [batch_size, out_channels, no_lifting_samples, * ]
        """
        # Unpack input tuple
        x, input_g_elems = input_tuple[0], input_tuple[1]
        # x: (B, in_ch, prev_no_elem, h, w)
        # input_g_elems (B, prev_no_elems)

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
        probs = self.encode_prob(x)
        probs = probs.abs()
        if (
            self.group_sample_per_layer and self.group_sampling_method == SamplingMethods.RANDOM
        ) or self.partial_equiv:
            # Sample from the group
            g_elems, ood = self.group.sample_from_stabilizer_given_x(
                no_samples=no_samples,
                no_elements=self.group_no_samples,
                method=self.group_sampling_method,
                device=x.device,
                partial_equivariance=True,
                probs=probs,
            )  # (B, cur_no_elements)
        else:
            g_elems = input_g_elems  # (B, cur_no_elements==prev_no_elements)

        self.probs_all = probs
        self.entropy = torch.log(torch.minimum(
            probs+1e-9, torch.ones_like(probs))).mean()

        # Act on the grid of positions:

        # Act along the group dimension
        # g_distance
        acted_g_elements = self.group.left_action_on_H(
            self.group.inv(g_elems), input_g_elems)
        # (B, cur_no_elements, prev_no_elements)

        # Normalize elements to coordinates between -1 and 1
        # normalized g distance
        acted_g_elements = self.group.normalize_g_distance(
            acted_g_elements).float()
        if self.group.__class__.__name__ == "SE2":
            acted_g_elements = acted_g_elements.unsqueeze(-1)
        # (B, cur_no_elements, prev_no_elements, 1)

        # Act on Rd with the resulting elements
        if acted_g_elements.shape[0] > g_elems.shape[0]:
            _g_elems = g_elems.repeat_interleave(
                acted_g_elements.shape[0], dim=0)
        else:
            _g_elems = g_elems
        acted_rel_pos_Rd = self.group.left_action_on_Rd(
            self.group.inv(
                _g_elems.view(-1, self.group.dimension_stabilizer)), rel_pos
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
                acted_rel_pos_Rd.unsqueeze(2).expand(
                    *(-1,) * 2, input_g_no_elems, *(-1,) * 2),
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
            no_samples * output_g_no_elems,  # batch
            self.out_channels,
            self.in_channels,
            *acted_group_rel_pos.shape[2:],
        )

        # Filter values outside the sphere
        # circle mask
        mask = (torch.norm(acted_group_rel_pos[:, :2], dim=1) > 1.0).unsqueeze(
            1).unsqueeze(2)
        # expand mask to the same size as conv_kernel
        mask = mask.expand_as(conv_kernel)
        conv_kernel[mask] = 0

        self.conv_kernel = conv_kernel

        # Reshape conv_kernel for convolution
        conv_kernel = conv_kernel.contiguous().view(
            no_samples * output_g_no_elems * self.out_channels,
            self.in_channels * input_g_no_elems,
            *conv_kernel.shape[-2:],
        )  # (B*cur_no_elem*out_ch, in_ch*prev_no_elem, kernel_size, kernel_size)

        # x: (B, in_ch, prev_no_elem, h, w)
        # Convolution:
        if no_samples == 1:
            out = torch_F.conv2d(
                # (B, in_ch*prev_no_elem, h, w)
                input=x.contiguous().view(-1, self.in_channels * \
                                          input_g_no_elems, *x.shape[-2:]),
                weight=conv_kernel,
                padding=self.padding,
                groups=1,
            )  # (B, cur_no_elem*out_ch, H, W)
        else:
            # Compute convolution as a 2D convolution
            out = torch_F.conv2d(
                # (1, B*in_ch*prev_no_elem, h, w)
                input=x.contiguous().view(1, -1, *x.shape[3:]),
                weight=conv_kernel,
                padding=self.padding,
                groups=no_samples,
            )  # (1, B*cur_no_elem*out_ch, H, W)
        out = out.view(-1, output_g_no_elems, self.out_channels,
                       *out.shape[2:]).transpose(1, 2)
        # (B, cur_no_elem, out_ch, H, W) tr-> (B, out_ch, cur_no_elem, H, W)

        # Add bias:
        if self.bias is not None:
            out = out + self.bias.view(1, -1, *(1,) * (len(out.shape) - 2))

        out = out.transpose(1, 2)
        dims = out.shape
        out = out.view(-1, *dims[2:])
        out[ood.view(-1)] = 0
        out = out.view(*dims)
        out = out.transpose(1, 2)
        assert out.size(1) == self.out_channels
        assert out.size(2) == output_g_no_elems
        assert len(out.size()) == 5

        return out, g_elems

    def _v1_3(
        self,
        input_tuple: Tuple[torch.Tensor, torch.Tensor],
    ):
        """
        :param input_tuple: Consists of two elements:
            (1) The input function in the group [batch_size, in_channels, * ]
            (2) The grid of sampled group elements for which the feature representation is obtained.
        :return: Output: Function on the group of size [batch_size, out_channels, no_lifting_samples, * ]
        """
        # Unpack input tuple
        x, input_g_elems = input_tuple[0], input_tuple[1]
        # x: (B, in_ch, prev_no_elem, h, w)
        # input_g_elems (B, prev_no_elems)

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
            g_elems = self.group.sample_from_stabilizer(
                no_samples=no_samples,
                no_elements=self.group_no_samples,
                method=self.group_sampling_method,
                device=x.device,
                partial_equivariance=self.partial_equiv,
                probs=self.probs,
            )  # (B, cur_no_elements)
        else:
            g_elems = input_g_elems  # (B, cur_no_elements==prev_no_elements)

        # Act on the grid of positions:

        # Act along the group dimension
        # g_distance
        acted_g_elements = self.group.left_action_on_H(
            self.group.inv(g_elems), input_g_elems)
        # (B, cur_no_elements, prev_no_elements)

        # Normalize elements to coordinates between -1 and 1
        # normalized g distance
        acted_g_elements = self.group.normalize_g_distance(
            acted_g_elements).float()
        if self.group.__class__.__name__ == "SE2":
            acted_g_elements = acted_g_elements.unsqueeze(-1)
        # (B, cur_no_elements, prev_no_elements, 1)

        # Act on Rd with the resulting elements
        if acted_g_elements.shape[0] > g_elems.shape[0]:
            _g_elems = g_elems.repeat_interleave(
                acted_g_elements.shape[0], dim=0)
        else:
            _g_elems = g_elems
        acted_rel_pos_Rd = self.group.left_action_on_Rd(
            self.group.inv(
                _g_elems.view(-1, self.group.dimension_stabilizer)), rel_pos
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
                acted_rel_pos_Rd.unsqueeze(2).expand(
                    *(-1,) * 2, input_g_no_elems, *(-1,) * 2),
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
            no_samples * output_g_no_elems,  # batch
            self.out_channels,
            self.in_channels,
            *acted_group_rel_pos.shape[2:],
        )
        dims = conv_kernel.size()
        score = self.get_score(
            g_elems/(2*math.pi), x
        )
        score = torch.softmax(score, dim=-1)  # (B,out_g_no,in_g_no)
        dist = torch.distributions.RelaxedOneHotCategorical(
            temperature=self.group.get_current_gumbel_temperature(),
            probs=score
        )
        categories = score.size(-1)
        # (in_g_no, B, out_g_no, in_g_no)
        samples = dist.rsample([input_g_no_elems])
        # samples = (samples >= 1/categories).float() + \
        #     samples.detach() - samples
        samples = samples.sum(0)  # (B, out_g_no, in_g_no)
        self.probs_all = samples
        diagprobs = torch.diagonal(samples, 0, -2, -1)
        diaglogprobs = torch.log(diagprobs)
        self.entropy = -(diagprobs*diaglogprobs).sum(-1).mean()

        score = score.view(-1, output_g_no_elems, 1, 1, input_g_no_elems, 1, 1)

        conv_kernel = conv_kernel.view(
            no_samples,
            output_g_no_elems,
            self.out_channels,
            self.in_channels,
            input_g_no_elems,
            *acted_group_rel_pos.shape[3:]
        )
        conv_kernel = (score*conv_kernel).view(*dims)

        # Filter values outside the sphere
        # circle mask
        mask = (torch.norm(acted_group_rel_pos[:, :2], dim=1) > 1.0).unsqueeze(
            1).unsqueeze(2)
        # expand mask to the same size as conv_kernel
        mask = mask.expand_as(conv_kernel)
        conv_kernel[mask] = 0

        self.conv_kernel = conv_kernel

        # Reshape conv_kernel for convolution
        conv_kernel = conv_kernel.contiguous().view(
            no_samples * output_g_no_elems * self.out_channels,
            self.in_channels * input_g_no_elems,
            *conv_kernel.shape[-2:],
        )  # (B*cur_no_elem*out_ch, in_ch*prev_no_elem, kernel_size, kernel_size)

        # x: (B, in_ch, prev_no_elem, h, w)
        # Convolution:
        if no_samples == 1:
            out = torch_F.conv2d(
                # (B, in_ch*prev_no_elem, h, w)
                input=x.contiguous().view(-1, self.in_channels * \
                                          input_g_no_elems, *x.shape[-2:]),
                weight=conv_kernel,
                padding=self.padding,
                groups=1,
            )  # (B, cur_no_elem*out_ch, H, W)
        else:
            # Compute convolution as a 2D convolution
            out = torch_F.conv2d(
                # (1, B*in_ch*prev_no_elem, h, w)
                input=x.contiguous().view(1, -1, *x.shape[3:]),
                weight=conv_kernel,
                padding=self.padding,
                groups=no_samples,
            )  # (1, B*cur_no_elem*out_ch, H, W)
        out = out.view(-1, output_g_no_elems, self.out_channels,
                       *out.shape[2:]).transpose(1, 2)
        # (B, cur_no_elem, out_ch, H, W) tr-> (B, out_ch, cur_no_elem, H, W)

        # Add bias:
        if self.bias is not None:
            out = out + self.bias.view(1, -1, *(1,) * (len(out.shape) - 2))

        return out, g_elems

    def _v1_4(
        self,
        input_tuple: Tuple[torch.Tensor, torch.Tensor],
    ):
        """
        :param input_tuple: Consists of two elements:
            (1) The input function in the group [batch_size, in_channels, * ]
            (2) The grid of sampled group elements for which the feature representation is obtained.
        :return: Output: Function on the group of size [batch_size, out_channels, no_lifting_samples, * ]
        """
        # Unpack input tuple
        x, input_g_elems = input_tuple[0], input_tuple[1]
        # x: (B, in_ch, prev_no_elem, h, w)
        # input_g_elems (B, prev_no_elems)

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
            g_elems = self.group.sample_from_stabilizer(
                no_samples=no_samples,
                no_elements=self.group_no_samples,
                method=self.group_sampling_method,
                device=x.device,
                partial_equivariance=self.partial_equiv,
                probs=self.probs,
            )  # (B, cur_no_elements)
        else:
            g_elems = input_g_elems  # (B, cur_no_elements==prev_no_elements)

        # Act on the grid of positions:

        # Act along the group dimension
        # g_distance
        acted_g_elements = self.group.left_action_on_H(
            self.group.inv(g_elems), input_g_elems)
        # (B, cur_no_elements, prev_no_elements)

        # Normalize elements to coordinates between -1 and 1
        # normalized g distance
        acted_g_elements = self.group.normalize_g_distance(
            acted_g_elements).float()
        if self.group.__class__.__name__ == "SE2":
            acted_g_elements = acted_g_elements.unsqueeze(-1)
        # (B, cur_no_elements, prev_no_elements, 1)

        # Act on Rd with the resulting elements
        if acted_g_elements.shape[0] > g_elems.shape[0]:
            _g_elems = g_elems.repeat_interleave(
                acted_g_elements.shape[0], dim=0)
        else:
            _g_elems = g_elems
        acted_rel_pos_Rd = self.group.left_action_on_Rd(
            self.group.inv(
                _g_elems.view(-1, self.group.dimension_stabilizer)), rel_pos
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
                acted_rel_pos_Rd.unsqueeze(2).expand(
                    *(-1,) * 2, input_g_no_elems, *(-1,) * 2),
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
            no_samples * output_g_no_elems,  # batch
            self.out_channels,
            self.in_channels,
            *acted_group_rel_pos.shape[2:],
        )

        # Filter values outside the sphere
        # circle mask
        mask = (torch.norm(acted_group_rel_pos[:, :2], dim=1) > 1.0).unsqueeze(
            1).unsqueeze(2)
        # expand mask to the same size as conv_kernel
        mask = mask.expand_as(conv_kernel)
        conv_kernel[mask] = 0

        self.conv_kernel = conv_kernel

        # Reshape conv_kernel for convolution
        conv_kernel = conv_kernel.contiguous().view(
            no_samples * output_g_no_elems * self.out_channels,
            self.in_channels * input_g_no_elems,
            *conv_kernel.shape[-2:],
        )  # (B*cur_no_elem*out_ch, in_ch*prev_no_elem, kernel_size, kernel_size)

        # x: (B, in_ch, prev_no_elem, h, w)
        # Convolution:
        if no_samples == 1:
            out = torch_F.conv2d(
                # (B, in_ch*prev_no_elem, h, w)
                input=x.contiguous().view(-1, self.in_channels * \
                                          input_g_no_elems, *x.shape[-2:]),
                weight=conv_kernel,
                padding=self.padding,
                groups=1,
            )  # (B, cur_no_elem*out_ch, H, W)
        else:
            # Compute convolution as a 2D convolution
            out = torch_F.conv2d(
                # (1, B*in_ch*prev_no_elem, h, w)
                input=x.contiguous().view(1, -1, *x.shape[3:]),
                weight=conv_kernel,
                padding=self.padding,
                groups=no_samples,
            )  # (1, B*cur_no_elem*out_ch, H, W)
        out = out.view(-1, output_g_no_elems, self.out_channels,
                       *out.shape[2:]).transpose(1, 2)
        # (B, cur_no_elem, out_ch, H, W) tr-> (B, out_ch, cur_no_elem, H, W)

        # Add bias:
        if self.bias is not None:
            out = out + self.bias.view(1, -1, *(1,) * (len(out.shape) - 2))

        alpha = self.get_alpha(x)  # (B, G)
        eps = torch.randn_like(alpha)  # (B, G)
        sigma = torch.exp(alpha)
        z = 1 + sigma*eps  # (B, G)
        B, G = z.shape
        out = z.view(B, 1, G, 1, 1)*out
        self.probs_all = alpha
        self.entropy = torch.log(self.probs_all+1e-9).mean()

        return out, g_elems

    def _v1_6(
        self,
        input_tuple: Tuple[torch.Tensor, torch.Tensor],
    ):
        """
        :param input_tuple: Consists of two elements:
            (1) The input function in the group [batch_size, in_channels, * ]
            (2) The grid of sampled group elements for which the feature representation is obtained.
        :return: Output: Function on the group of size [batch_size, out_channels, no_lifting_samples, * ]
        """
        # Unpack input tuple
        x, input_g_elems = input_tuple[0], input_tuple[1]
        # x: (B, in_ch, prev_no_elem, h, w)
        # input_g_elems (B, prev_no_elems)

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
        probs = self.encode_v1_6(x)
        assert probs.size(0) == x.size(0)
        assert probs.size(1) == 1
        if (
            self.group_sample_per_layer and self.group_sampling_method == SamplingMethods.RANDOM
        ) or self.partial_equiv:
            # Sample from the group
            g_elems = self.group.sample_from_posterior(
                no_samples=no_samples,
                no_elements=self.group_no_samples,
                method=self.group_sampling_method,
                device=x.device,
                partial_equivariance=True,
                probs=probs,
            )  # (B, cur_no_elements)
        else:
            g_elems = input_g_elems  # (B, cur_no_elements==prev_no_elements)

        self.probs_all = probs
        self.entropy = torch.log(probs+1e-9).mean()

        # Act on the grid of positions:

        # Act along the group dimension
        # g_distance
        acted_g_elements = self.group.left_action_on_H(
            self.group.inv(g_elems), input_g_elems)
        # (B, cur_no_elements, prev_no_elements)

        # Normalize elements to coordinates between -1 and 1
        # normalized g distance
        acted_g_elements = self.group.normalize_g_distance(
            acted_g_elements).float()
        if self.group.__class__.__name__ == "SE2":
            acted_g_elements = acted_g_elements.unsqueeze(-1)
        # (B, cur_no_elements, prev_no_elements, 1)

        # Act on Rd with the resulting elements
        if acted_g_elements.shape[0] > g_elems.shape[0]:
            _g_elems = g_elems.repeat_interleave(
                acted_g_elements.shape[0], dim=0)
        else:
            _g_elems = g_elems
        acted_rel_pos_Rd = self.group.left_action_on_Rd(
            self.group.inv(
                _g_elems.view(-1, self.group.dimension_stabilizer)), rel_pos
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
                acted_rel_pos_Rd.unsqueeze(2).expand(
                    *(-1,) * 2, input_g_no_elems, *(-1,) * 2),
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
            no_samples * output_g_no_elems,  # batch
            self.out_channels,
            self.in_channels,
            *acted_group_rel_pos.shape[2:],
        )

        # Filter values outside the sphere
        # circle mask
        mask = (torch.norm(acted_group_rel_pos[:, :2], dim=1) > 1.0).unsqueeze(
            1).unsqueeze(2)
        # expand mask to the same size as conv_kernel
        mask = mask.expand_as(conv_kernel)
        conv_kernel[mask] = 0

        self.conv_kernel = conv_kernel

        # Reshape conv_kernel for convolution
        conv_kernel = conv_kernel.contiguous().view(
            no_samples * output_g_no_elems * self.out_channels,
            self.in_channels * input_g_no_elems,
            *conv_kernel.shape[-2:],
        )  # (B*cur_no_elem*out_ch, in_ch*prev_no_elem, kernel_size, kernel_size)

        # x: (B, in_ch, prev_no_elem, h, w)
        # Convolution:
        if no_samples == 1:
            out = torch_F.conv2d(
                # (B, in_ch*prev_no_elem, h, w)
                input=x.contiguous().view(-1, self.in_channels * \
                                          input_g_no_elems, *x.shape[-2:]),
                weight=conv_kernel,
                padding=self.padding,
                groups=1,
            )  # (B, cur_no_elem*out_ch, H, W)
        else:
            # Compute convolution as a 2D convolution
            out = torch_F.conv2d(
                # (1, B*in_ch*prev_no_elem, h, w)
                input=x.contiguous().view(1, -1, *x.shape[3:]),
                weight=conv_kernel,
                padding=self.padding,
                groups=no_samples,
            )  # (1, B*cur_no_elem*out_ch, H, W)
        out = out.view(-1, output_g_no_elems, self.out_channels,
                       *out.shape[2:]).transpose(1, 2)
        # (B, cur_no_elem, out_ch, H, W) tr-> (B, out_ch, cur_no_elem, H, W)

        # Add bias:
        if self.bias is not None:
            out = out + self.bias.view(1, -1, *(1,) * (len(out.shape) - 2))

        return out, g_elems

    def _v1_2(
        self,
        input_tuple: Tuple[torch.Tensor, torch.Tensor],
    ):
        """
        :param input_tuple: Consists of two elements:
            (1) The input function in the group [batch_size, in_channels, * ]
            (2) The grid of sampled group elements for which the feature representation is obtained.
        :return: Output: Function on the group of size [batch_size, out_channels, no_lifting_samples, * ]
        """
        # Unpack input tuple
        x, input_g_elems = input_tuple[0], input_tuple[1]
        # x: (B, in_ch, prev_no_elem, h, w)
        # input_g_elems (B, prev_no_elems)

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
        probs = self.encode_prob(x)
        probs = probs.abs()
        if (
            self.group_sample_per_layer and self.group_sampling_method == SamplingMethods.RANDOM
        ) or self.partial_equiv:
            # Sample from the group
            g_elems, ood = self.group.sample_from_stabilizer_given_x(
                no_samples=no_samples,
                no_elements=self.group_no_samples,
                method=self.group_sampling_method,
                device=x.device,
                partial_equivariance=True,
                probs=probs,
            )  # (B, cur_no_elements)
        else:
            g_elems = input_g_elems  # (B, cur_no_elements==prev_no_elements)

        self.probs_all = probs
        self.entropy = torch.log(torch.minimum(
            probs+1e-9, torch.ones_like(probs))).mean()

        # Act on the grid of positions:

        # Act along the group dimension
        # g_distance
        acted_g_elements = self.group.left_action_on_H(
            self.group.inv(g_elems), input_g_elems)
        # (B, cur_no_elements, prev_no_elements)

        # Normalize elements to coordinates between -1 and 1
        # normalized g distance
        acted_g_elements = self.group.normalize_g_distance(
            acted_g_elements).float()
        if self.group.__class__.__name__ == "SE2":
            acted_g_elements = acted_g_elements.unsqueeze(-1)
        # (B, cur_no_elements, prev_no_elements, 1)

        # Act on Rd with the resulting elements
        if acted_g_elements.shape[0] > g_elems.shape[0]:
            _g_elems = g_elems.repeat_interleave(
                acted_g_elements.shape[0], dim=0)
        else:
            _g_elems = g_elems
        acted_rel_pos_Rd = self.group.left_action_on_Rd(
            self.group.inv(
                _g_elems.view(-1, self.group.dimension_stabilizer)), rel_pos
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
                acted_rel_pos_Rd.unsqueeze(2).expand(
                    *(-1,) * 2, input_g_no_elems, *(-1,) * 2),
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
            no_samples * output_g_no_elems,  # batch
            self.out_channels,
            self.in_channels,
            *acted_group_rel_pos.shape[2:],
        )

        # Filter values outside the sphere
        # circle mask
        mask = (torch.norm(acted_group_rel_pos[:, :2], dim=1) > 1.0).unsqueeze(
            1).unsqueeze(2)
        # expand mask to the same size as conv_kernel
        mask = mask.expand_as(conv_kernel)
        conv_kernel[mask] = 0

        self.conv_kernel = conv_kernel

        # Reshape conv_kernel for convolution
        conv_kernel = conv_kernel.contiguous().view(
            no_samples * output_g_no_elems * self.out_channels,
            self.in_channels * input_g_no_elems,
            *conv_kernel.shape[-2:],
        )  # (B*cur_no_elem*out_ch, in_ch*prev_no_elem, kernel_size, kernel_size)

        # x: (B, in_ch, prev_no_elem, h, w)
        # Convolution:
        if no_samples == 1:
            out = torch_F.conv2d(
                # (B, in_ch*prev_no_elem, h, w)
                input=x.contiguous().view(-1, self.in_channels * \
                                          input_g_no_elems, *x.shape[-2:]),
                weight=conv_kernel,
                padding=self.padding,
                groups=1,
            )  # (B, cur_no_elem*out_ch, H, W)
        else:
            # Compute convolution as a 2D convolution
            out = torch_F.conv2d(
                # (1, B*in_ch*prev_no_elem, h, w)
                input=x.contiguous().view(1, -1, *x.shape[3:]),
                weight=conv_kernel,
                padding=self.padding,
                groups=no_samples,
            )  # (1, B*cur_no_elem*out_ch, H, W)
        out = out.view(-1, output_g_no_elems, self.out_channels,
                       *out.shape[2:]).transpose(1, 2)
        # (B, cur_no_elem, out_ch, H, W) tr-> (B, out_ch, cur_no_elem, H, W)

        # Add bias:
        if self.bias is not None:
            out = out + self.bias.view(1, -1, *(1,) * (len(out.shape) - 2))

        out = out.transpose(1, 2)
        dims = out.shape
        out = out.view(-1, *dims[2:])
        out[ood.view(-1)] = 0
        out = out.view(*dims)
        out = out.transpose(1, 2)
        assert out.size(1) == self.out_channels
        assert out.size(2) == output_g_no_elems
        assert len(out.size()) == 5

        return out, g_elems

    def encode(self, x):  # v1.0
        x = x.mean(dim=(-1, -2))  # (B, C, G)
        x = x.transpose(1, 2)  # (B, G, C)
        q = self.filter_q(x)  # (B, G, C)
        k = self.filter_k(x)  # (B, G, C)
        k = k.transpose(1, 2)  # (B, C, G)
        d = q.size(-1)
        score = torch.bmm(q, k)/d**0.5  # (B, G, G)
        return score

    def encode_prob(self, x):  # v1.2
        x = x.mean(dim=(-1, -2, -3))  # (B, C)
        x = self.filter_layer1(x)
        x = self.filter_nonlinear(x)
        x = self.filter_layer2(x)  # (B, 1)
        return x

    def get_beta_params(self, x):
        x = x.mean(dim=(-1, -2, -3))  # (B, C)
        x = self.filter_layer1(x)
        x = self.filter_nonlinear(x)
        x = self.filter_layer2(x)  # (B, 2)
        return x

    def encode_v1_6(self, x):  # v1.2
        x = x.mean(dim=(-1, -2))  # (B, C, G)
        x = self.filter_conv1(x)
        x = self.filter_nonlinear(x)
        x = self.filter_conv2(x)
        x = self.filter_nonlinear(x)
        x = x.mean(dim=-1)  # (B, C)
        x = self.filter_linear1(x)  # (B, C/2)
        x = torch.sigmoid(x)
        return x

    def get_score(self, out_g_elems, x):  # v1.1
        out_g_elems = out_g_elems.float()
        if out_g_elems.shape[0] != x.shape[0]:
            out_g_elems = out_g_elems.repeat_interleave(x.shape[0], dim=0)
        out_g_elems = out_g_elems.unsqueeze(-1)  # (B, G, 1)
        x = x.mean(dim=(-1, -2))  # (B, C, G)
        x = x.transpose(1, 2)  # (B, G, C)
        q = self.filter_q(out_g_elems)  # (B, G, C)
        k = self.filter_k(x)  # (B, G, C)
        k = k.transpose(1, 2)  # (B, C, G)
        d = q.size(-1)
        score = torch.bmm(q, k)/d**0.5  # (B, G, G)
        return score

    def get_alpha(self, x):  # v1.4
        x = x.mean(dim=(-1, -2, -3))  # (B, C)
        x = self.filter_layer1(x)
        x = self.filter_nonlinear(x)
        x = self.filter_layer2(x)  # (B, 1)
        return x

    def sample_z(self, x):
        score = self.encode(x)
        logits = torch.softmax(score, dim=-1)
        dist = torch.distributions.RelaxedBernoulli(
            temperature=self.group.get_current_gumbel_temperature(),
            logits=logits
        )
        z = dist.rsample()
        assert z.size() == logits.size()
        return z, dist, logits


class VarLiftingConv(LiftingConvBase, VarConv):
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
        self.version = conv_config.version

        # Construct the probability variables given the group.
        probs = self.group.construct_probability_variables(
            self.group_sampling_method,
            base_group_config.no_samples,
        )
        if self.lift_partial_equiv:
            self.probs = torch.nn.Parameter(probs)
        else:
            self.register_buffer("probs", probs)

        if self.version == "v1.6":  # conventional
            hidden = 2*in_channels
            self.filter_conv1 = torch.nn.Conv2d(in_channels, hidden, 3, 2)
            self.filter_nonlinear = torch.nn.ReLU()
            self.filter_conv2 = torch.nn.Conv2d(hidden, hidden, 3, 2)
            self.filter_linear1 = torch.nn.Linear(hidden, 1)
            with torch.no_grad():
                self.filter_linear1.bias.fill_(5)

    def forward(self, *args, **kwargs):
        if self.version == "v1.6":
            return self._v1_6(*args, **kwargs)

    def _v1_6(self, x):
        """
        :param x: Input: Function on Rd of size [batch_size, in_channels, * ]
        :return: Output: Function on the group of size [batch_size, out_channels, no_lifting_samples, * ]
        """

        # Get input grid of conv kernel
        rel_pos = self.handle_rel_positions_on_Rd(x)

        # Define the number of independent samples to take from the group. If self.sample_per_batch_element == True, and
        # self.group_sampling_method == RANDOM, then batch_size independent self.group_no_samples samples from the group
        # will be taken for each batch element. Otherwise, the same rotations are used across batch elements.

        no_samples = x.shape[0]  # batch size

        probs = self.encode_v1_6(x)
        # Sample no_group_elements from the group, batch_size times.
        g_elems = self.group.sample_from_posterior(
            no_samples=no_samples,
            no_elements=self.group_no_samples,
            method=self.group_sampling_method,
            device=x.device,
            # We always lift first to the group.
            partial_equivariance=True,
            probs=probs,
        )  # [no_samples, output_g_no_elems]
        output_g_no_elems = g_elems.shape[-1]
        self.probs_all = probs
        self.entropy = torch.log(probs+1e-9).mean()

        # For R2, we don't need to go to the LieAlgebra. We can parameterize the kernel directly on the input space
        acted_rel_pos = self.group.left_action_on_Rd(
            self.group.inv(
                g_elems.view(-1, self.group.dimension_stabilizer)), rel_pos
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
        mask = (torch.norm(acted_rel_pos[:, :2], dim=1) > 1.0).unsqueeze(
            1).unsqueeze(2)
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

        out = out.view(-1, output_g_no_elems, self.out_channels,
                       *out.shape[-2:]).transpose(1, 2)

        # Add bias:
        if self.bias is not None:
            out = out + self.bias.view(1, -1, *(1,) * (len(out.shape) - 2))

        return out, g_elems

    def encode_v1_6(self, x):  # v1.2
        x = self.filter_conv1(x)
        x = self.filter_nonlinear(x)
        x = self.filter_conv2(x)
        x = self.filter_nonlinear(x)
        x = x.mean(dim=(-1, -2))  # (B, C)
        x = self.filter_linear1(x)
        x = torch.sigmoid(x)
        return x
