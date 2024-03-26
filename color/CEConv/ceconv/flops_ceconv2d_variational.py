"""Color Equivariant Convolutional Layer."""

import math
import typing
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


def _get_hue_rotation_matrix(rotations: int) -> torch.Tensor:
    """Returns a 3x3 hue rotation matrix.

    Rotates a 3D point by 360/rotations degrees along the diagonal.

    Args:
      rotations: int, number of rotations
    """

    assert rotations > 0, "Number of rotations must be positive."

    # Constants in rotation matrix
    cos = math.cos(2 * math.pi / rotations)
    sin = math.sin(2 * math.pi / rotations)
    const_a = 1 / 3 * (1.0 - cos)
    const_b = math.sqrt(1 / 3) * sin

    # Rotation matrix
    return torch.tensor(
        [
            [cos + const_a, const_a - const_b, const_a + const_b],
            [const_a + const_b, cos + const_a, const_a - const_b],
            [const_a - const_b, const_a + const_b, cos + const_a],
        ],
        dtype=torch.float32,
    )


def _trans_input_filter(weights, rotations, rotation_matrix) -> torch.Tensor:
    """Apply linear transformation to filter.

    Args:
      weights: float32, input filter of size [c_out, 3 (c_in), 1, k, k]
      rotations: int, number of rotations applied to filter
      rotation_matrix: float32, rotation matrix of size [3, 3]
    """

    # Flatten weights tensor.
    weights_flat = weights.permute(2, 1, 0, 3, 4)  # [1, 3, c_out, k, k]
    weights_shape = weights_flat.shape
    weights_flat = weights_flat.reshape((1, 3, -1))  # [1, 3, c_out*k*k]

    # Construct full transformation matrix.
    rotation_matrix = torch.stack(
        [torch.matrix_power(rotation_matrix, i) for i in range(rotations)], dim=0
    )

    # Apply transformation to weights.
    # [rotations, 3, 3] * [1, 3, c_out*k*k] --> [rotations, 3, c_out*k*k]
    transformed_weights = torch.matmul(rotation_matrix, weights_flat)
    # [rotations, 1, c_in (3), c_out, k, k]
    transformed_weights = transformed_weights.view((rotations,) + weights_shape)
    # [c_out, rotations, c_in (3), 1, k, k]
    tw = transformed_weights.permute(3, 0, 2, 1, 4, 5)

    return tw.contiguous()


def _trans_hidden_filter(weights: torch.Tensor, rotations: int) -> torch.Tensor:
    """Perform cyclic permutation on hidden layer filter parameters."""

    # Create placeholder for output tensor
    w_shape = weights.shape
    transformed_weights = torch.zeros(
        ((w_shape[0],) + (rotations,) + w_shape[1:]), device=weights.device
    )

    # Apply cyclic permutation on output tensor
    for i in range(rotations):
        transformed_weights[:, i, :, :, :, :] = torch.roll(weights, shifts=i, dims=2)

    return transformed_weights


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(log_a, temperature):
    y = log_a + sample_gumbel(log_a.size()).to(log_a.device)
    soft = F.softmax(y / temperature, dim=-1)
    hard = F.one_hot(torch.argmax(y, dim=-1), num_classes=log_a.shape[-1]).float()
    return hard + soft - soft.detach()

def gumbel_softmax(f, temperature):
    return gumbel_softmax_sample(f, temperature)

class CEConv2d(nn.Conv2d):
    """
    Applies a Color Equivariant convolution over an input signal composed of several
    input planes.


    Args:
        in_rotations (int): Number of input rotations: 1 for input layer, >1 for
            hidden layers.
        out_rotations (int): Number of output rotations.
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        learnable (bool): If True, the transformation matrix is learnable.
        separable (bool): If True, the convolution is separable.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    """

    def __init__(
        self,
        in_rotations: int,
        out_rotations: int,
        in_channels: int,
        out_channels: int,
        kernel_size: typing.Union[int, typing.Tuple[int, int]],
        learnable: bool = False,
        separable: bool = True,
        gumbel_no_iterations: int = 0,
        version: str = "v1.0",
        **kwargs
    ) -> None:
        self.in_rotations = in_rotations
        self.out_rotations = out_rotations
        self.separable = separable
        self.version = version
        self._filter_conv1 = 0
        self._filter_conv2 = 0

        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        # Initialize transformation matrix and weights.
        if in_rotations == 1:
            init = (
                torch.rand((3, 3)) * 2.0 / 3 - (1.0 / 3)
                if learnable
                else _get_hue_rotation_matrix(out_rotations)
            )
            self.transformation_matrix = Parameter(init, requires_grad=learnable)
            self.weight = Parameter(
                torch.Tensor(out_channels, in_channels, 1, *self.kernel_size)
            )
        else:
            if separable:
                if in_rotations > 1:
                    self.weight = Parameter(
                        # torch.Tensor(out_channels, 1, 1, *self.kernel_size)
                        torch.Tensor(out_channels, in_channels, 1, *self.kernel_size)
                    )
                    self.pointwise_weight = Parameter(
                        torch.Tensor(out_channels, in_channels, self.in_rotations, 1, 1)
                    )
            else:
                self.weight = Parameter(
                    torch.Tensor(
                        out_channels, in_channels, self.in_rotations, *self.kernel_size
                    )
                )
        
        self.gumbel_init_temperature = 0.5
        self.gumbel_end_temperature = 0.0001
        self.gumbel_no_iterations = gumbel_no_iterations
        self.register_buffer("gumbel_iter_counter", torch.zeros(1))

        if self.version == "v1.0":
            probs_dim = out_rotations-1
        elif self.version in ["v1.1", "v1.2"]:
            probs_dim = 1
        else:
            raise Exception(f"Version is invalid {self.version}")

        if self.in_rotations == 1:
            hidden = 2*in_channels
            self.filter_conv1 = torch.nn.Conv2d(in_channels, hidden, 3, 2)
            self._filter_conv1 = in_channels*hidden*3*3
            self.filter_nonlinear = torch.nn.ReLU()
            self.filter_conv2 = torch.nn.Conv2d(hidden, hidden, 3, 2)
            self._filter_conv2 = hidden*hidden*3*3
            self.filter_linear1 = torch.nn.Linear(hidden, probs_dim)
        else:
            self.filter_conv1 = torch.nn.Conv1d(in_channels, in_channels, 2, 1)
            self._filter_conv1 = in_channels*in_channels*2
            self.filter_nonlinear = torch.nn.ReLU()
            self.filter_conv2 = torch.nn.Conv1d(in_channels, in_channels, 2, 1)
            self._filter_conv2 = in_channels*in_channels*2
            self.filter_linear1 = torch.nn.Linear(in_channels, probs_dim)

        if self.version == "v1.0":
            with torch.no_grad():
                self.filter_linear1.bias.fill_(5)
        elif self.version == "v1.1":
            with torch.no_grad():
                self.filter_linear1.bias.fill_(0)
        elif self.version == "v1.2":
            with torch.no_grad():
                self.filter_linear1.bias.fill_(-3)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""

        # Compute standard deviation for weight initialization.
        n = self.in_channels * self.in_rotations * np.prod(self.kernel_size)
        stdv = 1.0 / math.sqrt(n)

        # Initialize weights.
        self.weight.data.uniform_(-stdv, stdv)
        if hasattr(self, "pointwise_weight"):
            self.pointwise_weight.data.uniform_(-stdv, stdv)

        # Initialize bias.
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def encode(self, x):
        assert len(x.size()) == 5
        if self.in_rotations == 1:
            x = x.squeeze(-3)
            x = self.filter_conv1(x)
            print(self._filter_conv1*x.size(-1)*x.size(-2))
            x = self.filter_nonlinear(x)
            x = self.filter_conv2(x)
            print(self._filter_conv2*x.size(-1)*x.size(-2))
            x = self.filter_nonlinear(x)
            x = x.mean(dim=(-1, -2))  # (B, C)
            x = self.filter_linear1(x)
        else:
            x = x.mean(dim=(-1, -2))  # (B, C, G)
            x = self.filter_conv1(x)
            print(self._filter_conv1*x.size(-1))
            x = self.filter_nonlinear(x)
            x = self.filter_conv2(x)
            print(self._filter_conv2*x.size(-1))
            x = self.filter_nonlinear(x)
            x = x.mean(dim=-1)  # (B, C)
            x = self.filter_linear1(x)  # (B, C/2)

        return x

    def get_current_gumbel_temperature(self):
        current_temperature = self.gumbel_init_temperature - self.gumbel_iter_counter / float(
            self.gumbel_no_iterations
        ) * (self.gumbel_init_temperature - self.gumbel_end_temperature)
        if self.training:
            self.gumbel_iter_counter += 1
        return current_temperature

    def forward(self, *args, **kwargs):
        if self.version == "v1.0":
            return self._v1_0(*args, **kwargs)
        elif self.version == "v1.1":
            return self._v1_1(*args, **kwargs)
        elif self.version == "v1.2":
            return self._v1_2(*args, **kwargs)
        else:
            raise NotImplementedError
    def _v1_0(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

        # Compute full filter weights.
        if self.in_rotations == 1:
            # Apply rotation to input layer filter.
            tw = _trans_input_filter(
                self.weight, self.out_rotations, self.transformation_matrix
            )
        else:
            # Apply cyclic permutation to hidden layer filter.
            if self.separable:
                weight = torch.mul(self.pointwise_weight, self.weight)
            else:
                weight = self.weight
            tw = _trans_hidden_filter(weight, self.out_rotations)

        tw_shape = (
            self.out_channels * self.out_rotations,
            self.in_channels * self.in_rotations,
            *self.kernel_size,
        )
        tw = tw.view(tw_shape)

        # Apply convolution.
        input_shape = input.size()
        input = input.view(
            input_shape[0],
            self.in_channels * self.in_rotations,
            input_shape[-2],
            input_shape[-1],
        )

        y = F.conv2d(
            input, weight=tw, bias=None, stride=self.stride, padding=self.padding
        )

        batch_size, _, ny_out, nx_out = y.size()
        y = y.view(batch_size, self.out_channels, self.out_rotations, ny_out, nx_out)

        # Apply bias.
        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1)
            y = y + bias
        
        input = input.view(
            input_shape[0],
            self.in_channels,
            self.in_rotations,
            input_shape[-2],
            input_shape[-1]
        )
        probs = self.encode(input)
        probs = torch.sigmoid(2 * probs)
        prob_rotations = (
            torch.distributions.RelaxedBernoulli(
                temperature=self.get_current_gumbel_temperature(),
                logits=probs,
            )
            .rsample([])
        )
        sample_rotation = (prob_rotations > 0.5).float()
        sample_rotation = sample_rotation - prob_rotations.detach() + prob_rotations

        self.probs_all = probs
        entropy = sample_rotation*torch.log(torch.sigmoid(probs)+1e-9) + (1-sample_rotation)*torch.log(1-torch.sigmoid(probs)+1e-9)
        self.entropy = entropy.mean()
        self.variance = sample_rotation.sum(-1).var(dim=0)
        self.samples = sample_rotation

        num_ones = sample_rotation.sum(-1, keepdims=True) + 1
        B, d = sample_rotation.size()
        sample_rotation = torch.cat([torch.ones(B, 1).to(sample_rotation), sample_rotation], dim=-1)
        sample_rotation /= num_ones # normalization

        y = y * sample_rotation.view(B, 1, self.out_rotations, 1, 1)
        return y

    def _v1_1(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

        # Compute full filter weights.
        if self.in_rotations == 1:
            # Apply rotation to input layer filter.
            tw = _trans_input_filter(
                self.weight, self.out_rotations, self.transformation_matrix
            )
        else:
            # Apply cyclic permutation to hidden layer filter.
            if self.separable:
                weight = torch.mul(self.pointwise_weight, self.weight)
            else:
                weight = self.weight
            tw = _trans_hidden_filter(weight, self.out_rotations)

        tw_shape = (
            self.out_channels * self.out_rotations,
            self.in_channels * self.in_rotations,
            *self.kernel_size,
        )
        tw = tw.view(tw_shape)

        # Apply convolution.
        input_shape = input.size()
        input = input.view(
            input_shape[0],
            self.in_channels * self.in_rotations,
            input_shape[-2],
            input_shape[-1],
        )

        y = F.conv2d(
            input, weight=tw, bias=None, stride=self.stride, padding=self.padding
        )
        IN = self.in_channels*self.in_rotations
        OUT = self.out_channels*self.out_rotations
        K2 = np.prod(self.kernel_size)
        H, W = y.size(-2), y.size(-1)
        print(IN*OUT*K2*H*W)

        batch_size, _, ny_out, nx_out = y.size()
        y = y.view(batch_size, self.out_channels, self.out_rotations, ny_out, nx_out)

        # Apply bias.
        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1)
            y = y + bias
        
        input = input.view(
            input_shape[0],
            self.in_channels,
            self.in_rotations,
            input_shape[-2],
            input_shape[-1]
        )
        probs = self.encode(input)
        probs = torch.exp(probs)
        max_factor = 6*2**(self.out_rotations//3-1)
        min_factor = 0.1
        probs = torch.minimum(probs+min_factor, max_factor*torch.ones_like(probs))
        prob_rotations = torch.arange(0,self.out_rotations).to(probs).view(1,-1)
        prob_rotations = torch.softmax(prob_rotations/probs,dim=-1)
        sample_rotation = (prob_rotations > (1/(self.out_rotations+1))).float()
        sample_rotation = sample_rotation - prob_rotations.detach() + prob_rotations

        self.probs_all = probs.clone()
        entropy = - (prob_rotations-(1/(self.out_rotations+0.5))).pow(2).sum(-1)
        self.entropy = entropy.mean()
        self.variance = sample_rotation.sum(-1).var(dim=0)
        self.samples = sample_rotation.clone()

        num_ones = sample_rotation.sum(-1, keepdims=True)
        B, d = sample_rotation.size()
        sample_rotation /= num_ones # normalization

        y = y * sample_rotation.view(B, 1, self.out_rotations, 1, 1)
        return y


    def _v1_2(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

        # Compute full filter weights.
        if self.in_rotations == 1:
            # Apply rotation to input layer filter.
            tw = _trans_input_filter(
                self.weight, self.out_rotations, self.transformation_matrix
            )
        else:
            # Apply cyclic permutation to hidden layer filter.
            if self.separable:
                weight = torch.mul(self.pointwise_weight, self.weight)
            else:
                weight = self.weight
            tw = _trans_hidden_filter(weight, self.out_rotations)

        tw_shape = (
            self.out_channels * self.out_rotations,
            self.in_channels * self.in_rotations,
            *self.kernel_size,
        )
        tw = tw.view(tw_shape)

        # Apply convolution.
        input_shape = input.size()
        input = input.view(
            input_shape[0],
            self.in_channels * self.in_rotations,
            input_shape[-2],
            input_shape[-1],
        )

        y = F.conv2d(
            input, weight=tw, bias=None, stride=self.stride, padding=self.padding
        )
        IN = self.in_channels*self.in_rotations
        OUT = self.out_channels*self.out_rotations
        K2 = np.prod(self.kernel_size)
        H, W = y.size(-2), y.size(-1)
        print(IN*OUT*K2*H*W)

        batch_size, _, ny_out, nx_out = y.size()
        y = y.view(batch_size, self.out_channels, self.out_rotations, ny_out, nx_out)

        # Apply bias.
        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1)
            y = y + bias
        
        input = input.view(
            input_shape[0],
            self.in_channels,
            self.in_rotations,
            input_shape[-2],
            input_shape[-1]
        )
        probs = self.encode(input)
        max_factor = 6*2**(self.out_rotations//3-1)
        min_factor = 0.1
        probs = min_factor + (max_factor-min_factor)*torch.sigmoid(probs)
        prob_rotations = torch.arange(0,self.out_rotations).to(probs).view(1,-1)
        prob_rotations = torch.softmax(prob_rotations/probs,dim=-1)
        sample_rotation = (prob_rotations > (1/(self.out_rotations+1))).float()
        sample_rotation = sample_rotation - prob_rotations.detach() + prob_rotations

        self.probs_all = probs
        entropy = - (prob_rotations-(1/(self.out_rotations+0.5))).pow(2).sum(-1)
        self.entropy = entropy.mean()
        self.variance = sample_rotation.sum(-1).var(dim=0)
        self.samples = sample_rotation

        num_ones = sample_rotation.sum(-1, keepdims=True)
        B, d = sample_rotation.size()
        sample_rotation /= num_ones # normalization

        y = y * sample_rotation.view(B, 1, self.out_rotations, 1, 1)
        return y