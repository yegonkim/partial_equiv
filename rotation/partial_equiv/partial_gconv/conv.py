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

class PartConv():
    # Convs containing probs
    pass


class ConvBase(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group: Group,
        conv_type: str,
        base_group_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
    ):
        super().__init__()

        # Unpack values from kernel_config
        kernel_type = kernel_config.type
        kernel_no_hidden = kernel_config.no_hidden
        kernel_no_layers = kernel_config.no_layers
        kernel_weight_norm = kernel_config.weight_norm
        kernel_omega0 = kernel_config.omega0
        kernel_learn_omega0 = kernel_config.learn_omega0
        kernel_size = kernel_config.size
        kernel_activation = kernel_config.activation
        kernel_norm = kernel_config.norm

        # Unpack values from conv_config
        bias = conv_config.bias
        padding = conv_config.padding
        learn_partial = conv_config.partial_equiv
        learn_lift_partial = conv_config.lift_partial_equiv

        # Unpack values from group_config and save them in self.
        self.group = group
        self.group_no_samples = base_group_config.no_samples
        self.group_sampling_method = base_group_config.sampling_method
        self.group_sample_per_batch_element = base_group_config.sample_per_batch_element
        self.lift_sample_per_batch_element = base_group_config.lift_sample_per_batch_element
        self.group_sample_per_layer = base_group_config.sample_per_layer

        # Save parameters in self
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.partial_equiv = learn_partial
        self.lift_partial_equiv = learn_lift_partial

        # Get the dim_linear as well as the dim_input_space from the type of convolution.
        if conv_type == "lifting":
            self.dim_linear = self.group.dimension_Rd
            self.dim_input_space = self.group.dimension_Rd
        elif conv_type == "group":
            self.dim_linear = self.group.dimension
            self.dim_input_space = self.group.dimension_Rd + self.group.dimension_stabilizer
        elif conv_type == "pointwise":
            self.dim_linear = self.group.dimension - self.group.dimension_Rd
            self.dim_input_space = self.group.dimension_stabilizer

        # Create kernel
        if kernel_type == "SIREN":
            self.kernelnet = ck.SIREN(
                dim_linear=self.dim_linear,
                dim_input_space=self.dim_input_space,
                out_channels=out_channels * in_channels,
                hidden_channels=kernel_no_hidden,
                no_layers=kernel_no_layers,
                weight_norm=kernel_weight_norm,
                bias=True,
                omega_0=kernel_omega0,
                learn_omega_0=kernel_learn_omega0,
            )
        elif kernel_type == "Gabor":
            self.kernelnet = ck.GaborNet(
                dim_linear=self.dim_linear,
                dim_input_space=self.dim_input_space,
                hidden_channels=kernel_no_hidden,
                out_channels=out_channels * in_channels,
                no_layers=kernel_no_layers,
                bias=True,
            )
        elif kernel_type == "Fourier":
            self.kernelnet = ck.FourierNet(
                dim_linear=self.dim_linear,
                dim_input_space=self.dim_input_space,
                hidden_channels=kernel_no_hidden,
                out_channels=out_channels * in_channels,
                no_layers=kernel_no_layers,
                bias=True,
            )
        elif kernel_type == "MAGNet":
            self.kernelnet = ck.MAGNet(
                dim_linear=self.dim_linear,
                dim_input_space=self.dim_input_space,
                hidden_channels=kernel_no_hidden,
                out_channels=out_channels * in_channels,
                no_layers=kernel_no_layers,
                steerable=False,  # TODO: Not implemented in 3D
                bias=True,
            )
        elif kernel_type == "MLP":
            self.kernelnet = ck.MLP(
                dim_linear=self.dim_linear,
                dim_input_space=self.dim_input_space,
                hidden_channels=kernel_no_hidden,
                out_channels=out_channels * in_channels,
                no_layers=kernel_no_layers,
                activation=kernel_activation,
                norm_type=kernel_norm,
                bias=True,
            )
        else:
            raise NotImplementedError(f"kernel_type {kernel_type} not implemented.")

        # Define position holder for relative positions
        self.rel_positions = None

        # Define bias:
        if bias:
            bias = torch.zeros((1, out_channels))  # [Batch, Ch, other_dimensions]
            self.bias = torch.nn.Parameter(bias)
        else:
            self.bias = None

    def handle_rel_positions_on_Rd(self, x):
        """
        Handles the vector of relative positions on Rd.
        In the case of lifting convolutions, the Rd vector is given to the KernelNet.
        In the case of group convolutions, this vector will **later** be acted upon, and concatenated
        with the additional group coordinates, in order to get the vector of positions given to the KernelNet.
        """
        if self.rel_positions is None:

            kernel_size = torch.zeros(1).int()

            # Decide the extent of the rel_positions vector
            if self.kernel_size == "full":
                kernel_size[0] = (2 * x.shape[-1]) - 1
            elif self.kernel_size == "same":
                kernel_size[0] = x.shape[-1]
            elif int(self.kernel_size) % 2 == 1:
                # Odd number
                kernel_size[0] = int(self.kernel_size)
            else:
                raise ValueError(
                    f'The horizon argument of the operation must be either "full", "same" or an odd number in string format. Current value: {self.kernel_size}'
                )

            # Creates the vector of relative positions.
            rel_positions = g_utils.rel_positions_grid(grid_sizes=kernel_size.repeat(self.group.dimension_Rd))
            self.rel_positions = rel_positions.to(x.device)
            # -> With form: [dim, x_dimension, y_dimension]

        return self.rel_positions

class LiftingConvBase(ConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group: Group,
        conv_type: str,
        base_group_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_type=conv_type,
            group=group,
            base_group_config=base_group_config,
            kernel_config=kernel_config,
            conv_config=conv_config,
        )


class LiftingConv(LiftingConvBase, PartConv):
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

        # Construct the probability variables given the group.
        probs = self.group.construct_probability_variables(
            self.group_sampling_method,
            base_group_config.no_samples,
        )
        if self.lift_partial_equiv:
            self.probs = torch.nn.Parameter(probs)
        else:
            self.register_buffer("probs", probs)

    def forward(self, x):
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
            partial_equivariance=self.lift_partial_equiv,  # We always lift first to the group.
            probs=self.probs,
        )  # [no_samples, output_g_no_elems]
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

        return out, g_elems


class GroupConvBase(ConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group: Group,
        conv_type: str,
        base_group_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_type=conv_type,
            group=group,
            base_group_config=base_group_config,
            kernel_config=kernel_config,
            conv_config=conv_config,
        )

class GroupConv(GroupConvBase, PartConv):
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

        # Construct the probability variables given the group.
        probs = self.group.construct_probability_variables(
            self.group_sampling_method,
            base_group_config.no_samples,
        )
        if self.partial_equiv:
            self.probs = torch.nn.Parameter(probs)
        else:
            self.register_buffer("probs", probs)

    def forward(
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
            ) # (B, cur_no_elements)
        else:
            g_elems = input_g_elems # (B, cur_no_elements==prev_no_elements) 

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

        return out, g_elems

class PointConvBase(ConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group: Group,
        conv_type: str,
        base_group_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_type=conv_type,
            group=group,
            base_group_config=base_group_config,
            kernel_config=kernel_config,
            conv_config=conv_config,
        )

class PointwiseGroupConv(PointConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group: Group,
        base_group_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
    ):
        kernel_config = copy.deepcopy(kernel_config)
        kernel_config.size = "1"

        conv_config = copy.deepcopy(conv_config)
        conv_config.bias = False
        conv_config.partial_equiv = False

        conv_type = "pointwise"
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
        input_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        """
        :param input_tuple: Consists of two elements:
            (1) The input function in the group [batch_size, in_channels, * ]
            (2) The grid of sampled group elements from the input.
            (3) The grid of sampled group elements from the output.
        :return: Output: Function on the group of size [batch_size, out_channels, no_lifting_samples, * ]
        """
        # Unpack input tuple
        x, input_g_elems, output_g_elems = input_tuple

        # Act along the group dimension
        acted_g_elements = self.group.left_action_on_H(self.group.inv(output_g_elems), input_g_elems)

        # Normalize elements to coordinates between -1 and 1
        acted_g_elements = self.group.normalize_g_distance(acted_g_elements).float()
        if self.group.__class__.__name__ == "SE2":
            acted_g_elements = acted_g_elements.unsqueeze(-1)

        input_g_no_elems = acted_g_elements.shape[2]
        output_g_no_elems = acted_g_elements.shape[1]
        no_samples = acted_g_elements.shape[0]

        acted_g_elements = (
            acted_g_elements.transpose(-1, -2)
            .contiguous()
            .view(
                no_samples * output_g_no_elems,
                self.group.dimension_stabilizer,
                input_g_no_elems,
            )
        )
        # Resulting grid: [no_samples * g_elems, group.dimension_stabilizer, self.input_g_elems]

        # Get the kernel
        conv_kernel = self.kernelnet(acted_g_elements).view(
            no_samples * output_g_no_elems,
            self.out_channels,
            self.in_channels,
            input_g_no_elems,
            1,
            1,
        )
        self.conv_kernel = conv_kernel

        # Reshape conv_kernel for convolution
        conv_kernel = conv_kernel.view(
            no_samples * output_g_no_elems * self.out_channels,
            self.in_channels * input_g_no_elems,
            *conv_kernel.shape[-2:],
        )

        # Convolution:
        if no_samples == 1:
            out = torch_F.conv2d(
                input=x.contiguous().view(-1, self.in_channels * input_g_no_elems, *x.shape[-2:]),
                weight=conv_kernel,
                padding=self.padding,
                groups=1,
            )
        else:
            # Compute convolution as a 2D convolution
            out = torch_F.conv2d(
                input=x.contiguous().view(1, -1, *x.shape[3:]),
                weight=conv_kernel,
                padding=self.padding,
                groups=no_samples,
            )
        out = out.view(-1, output_g_no_elems, self.out_channels, *out.shape[2:]).transpose(1, 2)

        # Add bias:
        if self.bias is not None:
            out = out + self.bias.view(1, -1, *(1,) * (len(out.shape) - 2))

        return out, output_g_elems


class LocalGroupConvDep(GroupConvBase):
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
        conv_config.partial_equiv = False
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_type=conv_type,
            group=group,
            base_group_config=base_group_config,
            kernel_config=kernel_config,
            conv_config=conv_config,
        )

        # Construct the probability variables given the group.
        probs = self.group.construct_probability_variables(
            self.group_sampling_method,
            base_group_config.no_samples,
        )
        if self.partial_equiv:
            self.probs = torch.nn.Parameter(probs)
        else:
            self.register_buffer("probs", probs)
        
        self.nbhd_window = conv_config.nbhd_window

    def forward(
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
            ) # (B, cur_no_elements)
        else:
            g_elems = input_g_elems # (B, cur_no_elements==prev_no_elements) 

        # Act on the grid of positions:

        # Act along the group dimension
        # g_distance
        k=self.nbhd_window
        acted_g_elements, g_distance = self.group.left_action_on_nbhdH(self.group.inv(g_elems), input_g_elems, k=k)
        # acted_g_elements = self.group.left_action_on_H(self.group.inv(g_elems), input_g_elems)
        # (B, cur_no_elements, k)

        # Normalize elements to coordinates between -1 and 1
        # normalized g distance
        acted_g_elements = self.group.normalize_g_distance(acted_g_elements).float()
        if self.group.__class__.__name__ == "SE2":
            acted_g_elements = acted_g_elements.unsqueeze(-1)
        # (B, cur_no_elements, k, 1)

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
        # Resulting grid: [no_samples * g_elems, group.dim, k, kernel_size, kernel_size]
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
        # (B*cur_no_elements, 2, kernel_size, kernel_size)
        # torch.cat([(uinv*Rd)_x, (uinv*Rd)_y, uinv*v])
        # (B*cur_no_elements, 2+dimension_stabilizer, k, kernel_size, kernel_size)

        self.acted_rel_pos = acted_group_rel_pos

        # Get the kernel: conv3d [2+dim_stb, k, kernel, kernel]
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

        mask = (
            g_distance.abs().view(-1, *g_distance.shape[2:]) > k*math.pi/8
        ).view(-1, 1, 1, k, 1, 1)
        mask = mask.expand_as(conv_kernel)
        conv_kernel[mask] = 0

        self.conv_kernel = conv_kernel

        # Reshape conv_kernel for convolution
        conv_kernel = conv_kernel.contiguous().view(
            no_samples * output_g_no_elems * self.out_channels,
            self.in_channels * input_g_no_elems,
            *conv_kernel.shape[-2:],
        ) # (no_samples*cur_no_elem*out_ch, in_ch*k, kernel_size, kernel_size)

        # x: (B, in_ch, prev_no_elem, h, w)
        bdr = k//2
        l = x.size(2) # prev_no_elem
        # x: (B, in_ch, prev_no_elem+2*(k//2), h, w)
        x = torch.cat([x[:,:,-bdr:,:,:],x,x[:,:,:bdr,:,:]], dim=2)
        # x: (B, in_ch, prev_no_elem, k, h, w)
        x = torch.stack([x[:,:,i:l+i,:,:] for i in range(k)], dim=3)
        # x: (B, prev_no_elem, in_ch, k, h, w)
        x = x.transpose(1,2)
        # Convolution:
        if no_samples == 1:
            out = torch_F.conv2d(
                # (B, cur_no_elem*in_ch*k, h, w)
                input=x.contiguous().view(-1, output_g_no_elems*self.in_channels*input_g_no_elems, *x.shape[-2:]),
                weight=conv_kernel,
                padding=self.padding,
                groups=1*output_g_no_elems,
            ) # (B, cur_no_elem*out_ch, H, W)
        else:
            # Compute convolution as a 2D convolution
            out = torch_F.conv2d(
                # (1, B*cur_no_elem*in_ch*k, h, w)
                input=x.contiguous().view(1, -1, *x.shape[-2:]),
                weight=conv_kernel,
                padding=self.padding,
                groups=no_samples*output_g_no_elems,
            ) # (1, B*cur_no_elem*out_ch, H, W)
        out = out.view(-1, output_g_no_elems, self.out_channels, *out.shape[2:]).transpose(1, 2)
        # (B, cur_no_elem, out_ch, H, W) tr-> (B, out_ch, cur_no_elem, H, W) 

        # Add bias:
        if self.bias is not None:
            out = out + self.bias.view(1, -1, *(1,) * (len(out.shape) - 2))

        return out, g_elems


class LocalGroupConv(GroupConvBase, PartConv):
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

        # Construct the probability variables given the group.
        probs = self.group.construct_probability_variables(
            self.group_sampling_method,
            base_group_config.no_samples,
        )
        if self.partial_equiv:
            self.probs = torch.nn.Parameter(probs)
        else:
            self.register_buffer("probs", probs)

        self.nbhd_window = conv_config.nbhd_window

    def forward(
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
            ) # (B, cur_no_elements)
        else:
            g_elems = input_g_elems # (B, cur_no_elements==prev_no_elements) 

        # Act on the grid of positions:

        # Act along the group dimension
        # g_distance
        g_distance = self.group.left_action_on_H(self.group.inv(g_elems), input_g_elems)
        # (B, cur_no_elements, prev_no_elements)

        # Normalize elements to coordinates between -1 and 1
        # normalized g distance
        acted_g_elements = self.group.normalize_g_distance(g_distance).float()
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

        # Enforcing locality
        k = self.nbhd_window
        n = self.group_no_samples
        g_distance = g_distance.view(-1, *g_distance.shape[2:])
        right = g_distance <= k*math.pi/n
        left = g_distance > 2*math.pi - k*math.pi/n
        mask = ~(right|left)
        mask = mask.view(-1, 1, 1, input_g_no_elems, 1, 1)
        mask = mask.expand_as(conv_kernel)
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

        return out, g_elems