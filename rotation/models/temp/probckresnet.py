import copy
from functools import partial

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
from omegaconf import OmegaConf

import rotation.partial_equiv.general as gral
import rotation.partial_equiv.groups as groups

# project
import rotation.partial_equiv.partial_gconv as partial_gconv
from rotation.partial_equiv.general.nn.pass_module import ApplyFirstElem

# typing
from rotation.partial_equiv.groups import Group, SamplingMethods
from rotation.partial_equiv.partial_gconv.module import circular_masking

class ProbCKResBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group: Group,
        base_group_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
        NormType: torch.nn.Module,
        dropout: float,
        pool: bool,
    ):
        super().__init__()

        # Define convolutional layers
        ProbGroupConv = (partial_gconv.ProbGroupConv if conv_config.partial_equiv else partial_gconv.LocalGroupConv)
        Conv = partial(
            ProbGroupConv,
            base_group_config=base_group_config,
            kernel_config=kernel_config,
            conv_config=conv_config,
        )
        self.gconv1 = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            group=copy.deepcopy(group),
        )
        self.gconv2 = Conv(
            in_channels=out_channels,
            out_channels=out_channels,
            group=copy.deepcopy(group),
        )
        # Norm layers:
        self.norm_out = ApplyFirstElem(NormType(out_channels))
        # Dropout layer
        self.dp = ApplyFirstElem(torch.nn.Dropout(dropout))
        # Activation
        self.activ = ApplyFirstElem(torch.nn.ReLU())
        # Pool
        if pool:
            pool = partial_gconv.pool.MaxPoolRn(
                kernel_size=2,
                stride=2,
                padding=0,
            )
        else:
            pool = torch.nn.Identity()
        self.pool = ApplyFirstElem(pool)

        # Shortcut connection
        shortcut = []
        if (in_channels != out_channels) or base_group_config.sample_per_layer or conv_config.partial_equiv:
            # Make the width of the network smaller
            kernel_config_shortcut = copy.deepcopy(kernel_config)
            kernel_config_shortcut.no_hidden = kernel_config_shortcut.no_hidden // 2
            # Create the shortcut
            shortcut.append(
                partial_gconv.PointwiseGroupConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    group=group,
                    base_group_config=base_group_config,
                    kernel_config=kernel_config_shortcut,
                    conv_config=conv_config,
                )
            )
            self.shortcut_is_pointwise = True
        else:
            self.shortcut_is_pointwise = False
        self.shortcut = torch.nn.Sequential(*shortcut)
        

    def entropy(self):
        return self.gconv1.entropy + self.gconv2.entropy

    def forward(self, input_tuple):
        x, input_g_elems, input_x = input_tuple
        # Following Sosnovik et al. 2020, dropout placed after first ReLU.
        output = self.gconv1([x, input_g_elems, input_x])
        out = output[0]
        g_elems = output[1]
        out = torch.nn.functional.layer_norm(out, out.shape[-3:])  # InstanceNorm
        out, g_elems = self.dp(self.activ([out, g_elems]))
        # self.entropy += self.gconv1.entropy

        output = self.gconv2([out, g_elems, input_x])
        out = output[0]
        g_elems = output[1]
        out = torch.nn.functional.layer_norm(out, out.shape[-3:])  # InstanceNorm
        out, g_elems = self.activ([out, g_elems])
        # self.entropy += self.gconv2.entropy

        # Shortcut
        if self.shortcut_is_pointwise:
            shortcut, g_elems = self.shortcut([x, input_g_elems, g_elems])
        else:
            shortcut, g_elems = self.shortcut([x, input_g_elems])
        out = out + shortcut

        out, g_elems = self.activ(self.pool(self.norm_out([out, g_elems])))
        return out, g_elems, input_x

class ProbCKResNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_group: Group,
        net_config: OmegaConf,
        base_group_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
        **kwargs,
    ):
        super().__init__()

        # Unpack arguments from net_config
        hidden_channels = net_config.no_hidden
        norm = net_config.norm
        no_blocks = net_config.no_blocks
        dropout = net_config.dropout
        pool_blocks = net_config.pool_blocks
        block_width_factors = net_config.block_width_factors
        last_conv_is_T2 = net_config.last_conv_T2
        learnable_final_pooling = net_config.learnable_final_pooling
        final_spatial_dim = net_config.final_spatial_dim

        # Params in self
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.last_conv_is_T2 = last_conv_is_T2
        self.partial_equiv = conv_config.partial_equiv
        self.lift_partial_equiv = conv_config.lift_partial_equiv
        self.learnable_final_pooling = learnable_final_pooling
        self.final_spatial_dim = torch.tensor(final_spatial_dim)

        # Define type of normalization layer to use
        if norm == "BatchNorm":
            NormType = getattr(torch.nn, f"BatchNorm{base_group.dimension}d")
        elif norm == "LayerNorm":
            NormType = gral.nn.LayerNorm
        else:
            raise NotImplementedError(f"No norm type {norm} found.")

        # Activation layer
        self.activ = ApplyFirstElem(torch.nn.ReLU())

        # Lifting
        LiftingConv = (
            partial_gconv.PartialLiftingConv if self.lift_partial_equiv else partial_gconv.LiftingConv
        )
        self.lift_conv = LiftingConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            group=copy.deepcopy(base_group),
            base_group_config=base_group_config,
            kernel_config=kernel_config,
            conv_config=conv_config,
        )
        # Lifting normalization layer
        self.lift_norm = ApplyFirstElem(NormType(hidden_channels))

        # Pooling after lifting
        if 0 in pool_blocks:
            pool = partial_gconv.pool.MaxPoolRn(
                kernel_size=2,
                stride=2,
                padding=0,
            )
        else:
            pool = torch.nn.Identity()
        self.lift_pool = ApplyFirstElem(pool)

        # Define blocks
        # Create vector of width_factors:
        # If value is zero, then all values are one
        if block_width_factors[0] == 0.0:
            width_factors = (1,) * no_blocks
        else:
            width_factors = [
                (factor,) * n_blcks for factor, n_blcks in gral.utils.pairwise_iterable(block_width_factors)
            ]
            width_factors = [factor for factor_tuple in width_factors for factor in factor_tuple]

        if len(width_factors) != no_blocks:
            raise ValueError("The size of the width_factors does not matched the number of blocks in the network.")

        blocks = []
        for i in range(no_blocks):
            print(f"Block {i+1}/{no_blocks}")

            if i == 0:
                input_ch = hidden_channels
                hidden_ch = int(hidden_channels * width_factors[i])
            else:
                input_ch = int(hidden_channels * width_factors[i - 1])
                hidden_ch = int(hidden_channels * width_factors[i])


            blocks.append(
                ProbCKResBlock(
                    in_channels=input_ch,
                    out_channels=hidden_ch,
                    group=base_group,
                    base_group_config=base_group_config,
                    kernel_config=kernel_config,
                    conv_config=conv_config,
                    dropout=dropout,
                    NormType=NormType,
                    pool=(i + 1) in pool_blocks,
                )
            )
        self.blocks = torch.nn.Sequential(*blocks)

        # Last layer
        # calculate output channels of blocks
        if block_width_factors[0] == 0.0:
            final_no_hidden = hidden_channels
        else:
            final_no_hidden = int(hidden_channels * block_width_factors[-2])
        # Construct layer
        last_lyr_conv_config = copy.deepcopy(conv_config)
        last_lyr_base_group_config = copy.deepcopy(base_group_config)

        last_lyr_conv_config.padding = "valid"

        if self.last_conv_is_T2:
            # Set partial equivariance to False
            last_lyr_conv_config.partial_equiv = False

            # Change the configs of the last layer
            last_lyr_base_group_config.no_samples = 1
            last_lyr_base_group_config.sample_per_layer = False
            last_lyr_base_group_config.sampling_method = SamplingMethods.DETERMINISTIC

        ProbGroupConv = (partial_gconv.ProbGroupConv if conv_config.partial_equiv else partial_gconv.LocalGroupConv)
        self.last_gconv = ProbGroupConv(
            in_channels=final_no_hidden,
            out_channels=final_no_hidden,
            group=base_group,
            base_group_config=last_lyr_base_group_config,
            kernel_config=kernel_config,
            conv_config=last_lyr_conv_config,
        )
        # Last g_conv normalization layer
        self.last_gconv_norm = ApplyFirstElem(NormType(final_no_hidden))

        # Create learnable pooling layer, if required:
        if self.learnable_final_pooling:
            if conv_config.partial_equiv:
                raise ValueError(f"learnable final pooling can only be used without partial equivariance.")

            self.learnable_pooling = torch.nn.Linear(
                in_features=last_lyr_base_group_config.no_samples * torch.prod(self.final_spatial_dim).item(),
                out_features=1,
                bias=True,
            )

        # Last Layer
        LastLinearType = getattr(gral.nn, f"Linear{base_group.dimension_Rd}d")
        # create
        self.out_layer = LastLinearType(in_channels=final_no_hidden, out_channels=out_channels)

    def entropy(self):
        e = 0
        for b in self.blocks:
            e += b.entropy()
        e += self.last_gconv.entropy
        return e

    def forward(self, x):
        # Lifting
        out, g_samples = self.activ(self.lift_pool(self.lift_norm(self.lift_conv(x))))
        # Rx = TF.rotate(x, 45, InterpolationMode.BILINEAR)
        # Rout, Rg_samples = self.activ(self.lift_pool(self.lift_norm(self.lift_conv(Rx))))
        # _out = torch.cat(list(out[1,0]),dim=-1)
        # _Rout = torch.cat(list(Rout[1,0]),dim=-1)
        # show = torch.cat([_out,_Rout], dim=-2)
        # save_image(show, "/mnt/home/hyunsu/partial_gcnn/equivariance_of_localliftingconv.png")
        # assert False

        # Group blocks
        out, g_samples, _ = self.blocks([out, g_samples, x])

        # Last g_conv
        if self.last_conv_is_T2:
            out = torch.mean(out, dim=-3, keepdim=True)
            g_samples = torch.zeros_like(g_samples[:, :1], device=g_samples.device)

        output = self.activ(self.last_gconv_norm(self.last_gconv([out, g_samples, x])))
        out = output[0]
        g_sammples = output[1]

        # global pooling
        if self.learnable_final_pooling:
            out_shape = out.shape
            out = self.learnable_pooling(out.view(*out_shape[:-3], -1))
            out = out.view(*out_shape[:-3], *((1,) * (len(out_shape) - 2)))
        else:
            out = torch.amax(out, dim=(-3, -2, -1), keepdim=True)

        # Final layer
        out = self.out_layer(out.squeeze(-3))
        return out.view(-1, self.out_channels)



