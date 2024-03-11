# torch
import copy

# typing
from typing import Tuple

import torch
import torch.nn.functional as torch_F
from omegaconf import OmegaConf

import partial_equiv.ck as ck

# project
import partial_equiv.general.utils as g_utils
from partial_equiv.groups import Group, SamplingMethods
from partial_equiv.partial_gconv.module import Filter, circular_masking
from .conv import ConvBase, GroupConvBase, LiftingConvBase
import math
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image

class ProbConv():
    # Convs containing input-dependent probs (probs_all)
    pass

class ProbGroupConv(GroupConvBase, ProbConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group: Group,
        base_group_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf
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
            if conv_config.version == "v1.1":
                probs_out = 2
            else:
                probs_out = len(probs)
            self.probs = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, 2),
                # torch.nn.Conv2d(1, 16, 3, 2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, 3, 2),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Flatten(),
                torch.nn.Linear(288,128),
                torch.nn.ReLU(),
                torch.nn.Linear(128,probs_out),
            )
            self.probs_all = None
        else:
            self.register_buffer("probs", probs)

        self.register_buffer("probs_all", None)
        self.register_buffer("entropy", torch.zeros(1).mean())
        self.version = conv_config.version

    def forward(self, *args, **kwargs):
        if self.version == "v1.0":
            return self._v1_0(*args, **kwargs)
        elif self.version == "v1.1":
            return self._v1_1(*args, **kwargs)
        else:
            raise NotImplementedError

    def _v1_0( # Unif[-theta, theta]
        self,
        input_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        """
        :param input_tuple: Consists of two elements:
            (1) The input function in the group [batch_size, in_channels, * ]
            (2) The grid of sampled group elements for which the feature representation is obtained.
        :return: Output: Function on the group of size [batch_size, out_channels, no_lifting_samples, * ]
        """
        # Unpack input tuple
        x, input_g_elems, input_x = input_tuple

        # Get input grid of conv kernel
        rel_pos = self.handle_rel_positions_on_Rd(x)

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
            probs = torch.sigmoid(self.probs(input_x))
            g_elems = self.group.sample_from_posterior(
                no_samples=no_samples,
                no_elements=self.group_no_samples,
                method=self.group_sampling_method,
                device=x.device,
                partial_equivariance=self.partial_equiv,
                probs=probs,
            )
            self.probs_all = probs
            self.entropy = torch.log(self.probs_all).squeeze(-1).mean()
        else:
            g_elems = input_g_elems

        # Act on the grid of positions:

        # Act along the group dimension
        acted_g_elements = self.group.left_action_on_H(self.group.inv(g_elems), input_g_elems)

        # Normalize elements to coordinates between -1 and 1
        acted_g_elements = self.group.normalize_g_distance(acted_g_elements).float()
        if self.group.__class__.__name__ == "SE2":
            acted_g_elements = acted_g_elements.unsqueeze(-1)

        # Act on Rd with the resulting elements
        acted_rel_pos_Rd = self.group.left_action_on_Rd(
            self.group.inv(g_elems.view(-1, self.group.dimension_stabilizer)), rel_pos
        )

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

        self.acted_rel_pos = acted_group_rel_pos

        # Get the kernel
        conv_kernel = self.kernelnet(acted_group_rel_pos).view(
            no_samples * output_g_no_elems,
            self.out_channels,
            self.in_channels,
            *acted_group_rel_pos.shape[2:],
        )

        # Filter values outside the sphere
        mask = (torch.norm(acted_group_rel_pos[:, :2], dim=1) > 1.0).unsqueeze(1).unsqueeze(2)
        mask = mask.expand_as(conv_kernel)
        conv_kernel[mask] = 0

        self.conv_kernel = conv_kernel

        # Reshape conv_kernel for convolution
        conv_kernel = conv_kernel.contiguous().view(
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

        return out, g_elems, input_x


    def _v1_1( # Unif[theta_min, theta_max]
        self,
        input_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        """
        :param input_tuple: Consists of two elements:
            (1) The input function in the group [batch_size, in_channels, * ]
            (2) The grid of sampled group elements for which the feature representation is obtained.
        :return: Output: Function on the group of size [batch_size, out_channels, no_lifting_samples, * ]
        """
        # Unpack input tuple
        x, input_g_elems, input_x = input_tuple

        # Get input grid of conv kernel
        rel_pos = self.handle_rel_positions_on_Rd(x)

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
            probs = torch.sigmoid(self.probs(input_x))
            g_elems = self.group.sample_from_posterior(
                no_samples=no_samples,
                no_elements=self.group_no_samples,
                method=self.group_sampling_method,
                device=x.device,
                partial_equivariance=self.partial_equiv,
                probs=probs,
                version=self.version
            )
            self.probs_all = probs
            # self.entropy = (self.probs_all[:,0]-self.probs_all[:,1]).abs().log().mean()
            theta_min, theta_max = torch.split(self.probs_all, 1, dim=-1)
            self.entropy = torch.log(torch.where(theta_min>theta_max, theta_max+1, theta_max)-theta_min).mean()
        else:
            g_elems = input_g_elems

        # Act on the grid of positions:

        # Act along the group dimension
        acted_g_elements = self.group.left_action_on_H(self.group.inv(g_elems), input_g_elems)

        # Normalize elements to coordinates between -1 and 1
        acted_g_elements = self.group.normalize_g_distance(acted_g_elements).float()
        if self.group.__class__.__name__ == "SE2":
            acted_g_elements = acted_g_elements.unsqueeze(-1)

        # Act on Rd with the resulting elements
        acted_rel_pos_Rd = self.group.left_action_on_Rd(
            self.group.inv(g_elems.view(-1, self.group.dimension_stabilizer)), rel_pos
        )

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

        self.acted_rel_pos = acted_group_rel_pos

        # Get the kernel
        conv_kernel = self.kernelnet(acted_group_rel_pos).view(
            no_samples * output_g_no_elems,
            self.out_channels,
            self.in_channels,
            *acted_group_rel_pos.shape[2:],
        )

        # Filter values outside the sphere
        mask = (torch.norm(acted_group_rel_pos[:, :2], dim=1) > 1.0).unsqueeze(1).unsqueeze(2)
        mask = mask.expand_as(conv_kernel)
        conv_kernel[mask] = 0

        self.conv_kernel = conv_kernel

        # Reshape conv_kernel for convolution
        conv_kernel = conv_kernel.contiguous().view(
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

        return out, g_elems, input_x



class ProbLiftingConv(LiftingConvBase, ProbConv):
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

        probs = self.group.construct_probability_variables(
            self.group_sampling_method,
            base_group_config.no_samples,
        )
        if conv_config.version == "v1.1":
            probs_out = 2
            bias_init = torch.nn.Parameter(torch.tensor([-5.,5.]).to(probs))
        else:
            probs_out = len(probs)
            bias_init = torch.nn.Parameter(5*torch.ones(probs_out).to(probs))
        self.probs = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(288,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,probs_out),
        )
        with torch.no_grad():
            self.probs[-1].bias = bias_init
        self.register_buffer("entropy", torch.zeros(1).mean())
        self.version = conv_config.version

    def forward(self, *args, **kwargs):
        if self.version == "v1.0":
            return self._v1_0(*args, **kwargs)
        elif self.version == "v1.1":
            return self._v1_0(*args, **kwargs) # only sample_from_posterior version is different
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

        assert self.lift_sample_per_batch_element
        no_samples = x.shape[0]  # batch size

        # Sample no_group_elements from the group, batch_size times.
        probs = torch.sigmoid(self.probs(x))
        g_elems = self.group.sample_from_posterior(
            no_samples=no_samples,
            no_elements=self.group_no_samples,
            method=self.group_sampling_method,
            device=x.device,
            partial_equivariance=self.lift_partial_equiv,  # We always lift first to the group.
            probs=probs,
        )  # [no_samples, self.group_no_samples]
        self.probs_all = probs
        if self.version == "v1.1":
            # self.entropy = (self.probs_all[:,0]-self.probs_all[:,1]).abs().log().mean()
            theta_min, theta_max = torch.split(self.probs_all, 1, dim=-1)
            self.entropy = torch.log(torch.where(theta_min>theta_max, theta_max+1, theta_max)-theta_min).mean()
        else:
            self.entropy = torch.log(self.probs_all).squeeze(-1).mean()
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


class LocalLiftingConv(LiftingConvBase, ProbConv):
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
        if self.version in ["v1.0", "v1.4"]:
            invariant_dim = 2
        else:
            invariant_dim = 1
        self.probs = Filter(
            in_channels,
            out_dim=invariant_dim,
            hidden_dim=conv_config.filter_hidden,
            version=self.version)
        self.register_buffer("entropy", torch.zeros(1).mean())
        self.probs_all = None

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

        assert self.lift_sample_per_batch_element
        no_samples = x.shape[0]  # batch size

        # Sample no_group_elements from the group, batch_size times.
        # probs = self.probs(circular_masking(x))
        probs, filter, _ = self.probs(x)
        # print("probs", probs[1])
        probs = probs.unsqueeze(-1)
        g_elems = self.group.filter_from_stabilizer(
            no_samples=no_samples,
            no_elements=self.group_no_samples,
            method=self.group_sampling_method,
            device=x.device,
            probs=probs,
            filter=filter,
        )  # [no_samples, self.group_no_samples]
        self.probs_all = filter
        theta_min, theta_max = torch.split(filter, 1, dim=-1)
        self.entropy = torch.log(torch.where(theta_min>theta_max, theta_max+1, theta_max)-theta_min).mean()

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

        assert self.lift_sample_per_batch_element
        no_samples = x.shape[0]  # batch size

        # Sample no_group_elements from the group, batch_size times.
        # probs = self.probs(circular_masking(x))
        probs, filter, _ = self.probs(x)
        # print("probs", probs[1])
        probs = probs.unsqueeze(-1)
        g_elems = self.group.filter_from_stabilizer(
            no_samples=no_samples,
            no_elements=self.group_no_samples,
            method=self.group_sampling_method,
            device=x.device,
            probs=probs,
            filter=filter,
            version=self.version
        )  # [no_samples, self.group_no_samples]
        self.probs_all = filter
        theta_min, theta_diff = torch.split(filter, 1, dim=-1)
        self.entropy = torch.log(theta_diff).mean()

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


    def _v1_2(self, x):
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

        # Sample no_group_elements from the group, batch_size times.
        g_elems = self.group.sample_from_stabilizer(
            no_samples=no_samples,
            no_elements=self.group_no_samples,
            method=self.group_sampling_method,
            device=x.device,
            partial_equivariance=False,
            probs=None,
        )  # [no_samples, self.group_no_samples]
        eps = g_elems/(2*math.pi)
        if eps.size(0) == 1:
            eps = eps.repeat_interleave(x.size(0), dim=0)
        _, logits, _ = self.probs(x, eps) # (B, out_g_no_elems)
        logprobs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(logprobs)

        self.probs_all = probs
        self.entropy = -(probs*logprobs).sum(-1).mean()

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

        out = out.view(
            -1, output_g_no_elems, self.out_channels, *out.shape[-2:]
        ).transpose(1, 2)

        # Add bias:
        if self.bias is not None:
            out = out + self.bias.view(1, -1, *(1,) * (len(out.shape) - 2))

        out = probs.view(probs.shape[0], 1, probs.shape[1], 1, 1)*out

        return out, g_elems



    def _v1_3(self, x):
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

        # Sample no_group_elements from the group, batch_size times.
        g_elems = self.group.sample_from_stabilizer(
            no_samples=no_samples,
            no_elements=self.group_no_samples,
            method=self.group_sampling_method,
            device=x.device,
            partial_equivariance=False,
            probs=None,
        )  # [no_samples, self.group_no_samples]
        eps = g_elems/(2*math.pi)
        if eps.size(0) == 1:
            eps = eps.repeat_interleave(x.size(0), dim=0)
        _, logits, _ = self.probs(x, eps) # (B, out_g_no_elems)

        dist = torch.distributions.RelaxedBernoulli(
            temperature=self.group.get_current_gumbel_temperature(),
            logits=logits
        )
        _probs = dist.rsample([1]).squeeze(0)
        probs = (_probs > 0.5).float()
        probs = probs + _probs.detach() - _probs
        empty = probs.sum(-1) == 0
        if len(probs[empty]) > 0:
            probs[empty][0] = 1

        self.probs_all = probs
        logprob = dist.log_prob(_probs)
        prob = logprob.exp()
        self.entropy = -(prob*logprob).mean()

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

        out = out.view(
            -1, output_g_no_elems, self.out_channels, *out.shape[-2:]
        ).transpose(1, 2)

        # Add bias:
        if self.bias is not None:
            out = out + self.bias.view(1, -1, *(1,) * (len(out.shape) - 2))

        out = probs.view(probs.shape[0], 1, probs.shape[1], 1, 1)*out

        return out, g_elems



    def _v1_4(self, x):
        """
        :param x: Input: Function on Rd of size [batch_size, in_channels, * ]
        :return: Output: Function on the group of size [batch_size, out_channels, no_lifting_samples, * ]
        """

        # Get input grid of conv kernel
        rel_pos = self.handle_rel_positions_on_Rd(x)

        # Define the number of independent samples to take from the group. If self.sample_per_batch_element == True, and
        # self.group_sampling_method == RANDOM, then batch_size independent self.group_no_samples samples from the group
        # will be taken for each batch element. Otherwise, the same rotations are used across batch elements.

        assert self.lift_sample_per_batch_element
        no_samples = x.shape[0]  # batch size

        # Sample no_group_elements from the group, batch_size times.
        # probs = self.probs(circular_masking(x))
        probs, filter, _ = self.probs(x)
        # print("probs", probs[1])
        g_elems = self.group.filter_from_stabilizer(
            no_samples=no_samples,
            no_elements=self.group_no_samples,
            method=self.group_sampling_method,
            device=x.device,
            probs=probs,
            filter=filter,
            version=self.version
        )  # [no_samples, self.group_no_samples]
        self.probs_all = filter
        theta_left, theta_right = torch.split(filter, 1, dim=-1)
        self.entropy = torch.log(theta_right+theta_left + 0.01).mean()

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
