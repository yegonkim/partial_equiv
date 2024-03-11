from e2cnn import gspaces
from e2cnn import nn
import torch
import torch.nn.functional as F
import math


def get_rotation_matrix(v, eps=10e-5):
    print("v shape", v.size())
    v = v / (torch.norm(v, dim=-1, keepdim=True) + eps)
    rot = torch.stack((
        torch.stack((v[:, 0], v[:, 1]), dim=-1),
        torch.stack((-v[:, 1], v[:, 0]), dim=-1),
        torch.zeros(v.size(0), 2).type_as(v)
    ), dim=-1)
    return rot

def rot_img(x, rot):
    grid = F.affine_grid(rot, x.size(), align_corners=False).type_as(x)
    x = F.grid_sample(x, grid, align_corners=False)
    return x

def get_rotation_matrix_from_radian(radian):
    cos = torch.cos(radian)
    sin = torch.sin(radian)
    R1 = torch.cat([cos,-sin])
    R2 = torch.cat([sin,cos])
    R = torch.stack([R1,R2])
    return R

def get_non_linearity(scalar_fields, vector_fields):
    out_type = scalar_fields + vector_fields
    relu = nn.ReLU(scalar_fields)
    norm_relu = nn.NormNonLinearity(vector_fields)
    nonlinearity = nn.MultipleModule(
        out_type,
        ['relu'] * len(scalar_fields) + ['norm'] * len(vector_fields),
        [(relu, 'relu'), (norm_relu, 'norm')]
    )
    return nonlinearity

def get_batch_norm(scalar_fields, vector_fields):
    out_type = scalar_fields + vector_fields
    batch_norm = nn.InnerBatchNorm(scalar_fields)
    norm_batch_norm = nn.NormBatchNorm(vector_fields)
    batch_norm = nn.MultipleModule(
        out_type,
        ['bn'] * len(scalar_fields) + ['nbn'] * len(vector_fields),
        [(batch_norm, 'bn'), (norm_batch_norm, 'nbn')]
    )
    return batch_norm

def circular_masking(image_tensor):
    # img_tensor (B, C, H, W)
    height, width = image_tensor.shape[2:]
    radius = min(height, width) // 2

    # Create a grid of coordinates
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
    y, x = y.to(image_tensor), x.to(image_tensor)

    # Create a circular mask using the distance from the center
    mask = ((x - width // 2) ** 2 + (y - height // 2) ** 2 <= radius ** 2).float()

    # Expand the mask dimensions to match the image
    mask = mask.unsqueeze(0).unsqueeze(0)

    # Apply the circular mask to the image
    masked_image = image_tensor * mask
    return masked_image

# class FilterLarge(torch.nn.Module):
#     def __init__(self, in_ch, out_dim, hidden_dim=32):
#         super().__init__()
#         self.out_dim=out_dim
#         self.r2_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=8)
#         in_type = nn.FieldType(self.r2_act, in_ch*[self.r2_act.trivial_repr])
#         self.input_type = in_type

#         # convolution 1
#         out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
#         out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
#         out_type = out_scalar_fields + out_vector_field
#         batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
#         nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

#         self.block1 = nn.SequentialModule(
#             #nn.MaskModule(in_type, 29, margin=1),
#             nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
#             batch_norm,
#             nonlinearity
#         )

#         # convolution 2
#         in_type = out_type
#         out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
#         out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
#         out_type = out_scalar_fields + out_vector_field
#         batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
#         nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

#         self.block2 = nn.SequentialModule(
#             nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
#             batch_norm,
#             nonlinearity
#         )
#         self.pool1 = nn.SequentialModule(
#             nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
#         )

#         # convolution 3
#         in_type = out_type
#         out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
#         out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
#         out_type = out_scalar_fields + out_vector_field
#         batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
#         nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

#         self.block3 = nn.SequentialModule(
#             nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
#             batch_norm,
#             nonlinearity
#         )

#         # convolution 4
#         in_type = out_type
#         out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
#         out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
#         out_type = out_scalar_fields + out_vector_field
#         batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
#         nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

#         self.block4 = nn.SequentialModule(
#             nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
#             batch_norm,
#             nonlinearity
#         )
#         self.pool2 = nn.SequentialModule(
#             nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
#         )

#         # convolution 5
#         in_type = out_type
#         out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
#         out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
#         out_type = out_scalar_fields + out_vector_field
#         batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
#         nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

#         self.block5 = nn.SequentialModule(
#             nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
#             batch_norm,
#             nonlinearity
#         )

#         # convolution 6
#         in_type = out_type
#         out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
#         out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
#         out_type = out_scalar_fields + out_vector_field
#         batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
#         nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

#         self.block6 = nn.SequentialModule(
#             nn.R2Conv(in_type, out_type, kernel_size=3, padding=2, bias=False),
#             batch_norm,
#             nonlinearity
#         )
#         self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)

#         # convolution 7 --> out
#         # the old output type is the input type to the next layer
#         in_type = out_type
#         out_scalar_fields = nn.FieldType(self.r2_act, out_dim * [self.r2_act.trivial_repr])
#         out_vector_field = nn.FieldType(self.r2_act, 1 * [self.r2_act.irrep(1)])
#         out_type = out_scalar_fields + out_vector_field

#         self.block7 = nn.SequentialModule(
#             nn.R2Conv(in_type, out_type, kernel_size=1, padding=0, bias=False),
#         )

#     def forward(self, x: torch.Tensor):
#         # x (128,3,32,32)
#         # x = x.unsqueeze(1)
#         #x = torch.nn.functional.pad(x, (0, 1, 0, 1), value=0).unsqueeze(1)
#         x = nn.GeometricTensor(x, self.input_type)

#         x = self.block1(x)
#         x = self.block2(x)
#         #x = self.pool1(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         #x = self.pool2(x)
#         x = self.block5(x)
#         x = self.block6(x)
#         #x = self.pool3(x)
#         x = self.block7(x)

#         #x = x.tensor.squeeze(-1).squeeze(-1)
#         x = x.tensor.mean(dim=(2, 3))

#         x_0, x_1 = x[:, :self.out_dim], x[:, self.out_dim:]
#         # x_0 (128,32) invariant value
#         # x_1 (128,2) equivariant value
#         a, b = x_1[:, 0], x_1[:, 1]
#         numer = a
#         norm = torch.minimum(a**2+b**2, 1e-9)
#         denom = torch.minimum(norm.sqrt(),1e-4)
#         theta = torch.arccos(numer/denom)
#         theta = torch.where(
#             x_1[:, 1] >= 0,
#             theta,
#             2*math.pi - theta
#         )
#         return (1-theta/2/math.pi)

class Filter(torch.nn.Module):
    def __init__(self, in_ch, out_dim, hidden_dim=5, version="v1.0", eps=False):
        super().__init__()
        self.version = version
        self.out_dim=out_dim
        self.r2_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=8)
        in_type = nn.FieldType(self.r2_act, in_ch*[self.r2_act.trivial_repr])
        self.input_type = in_type

        # convolution 1
        out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)
        if self.version in ["v1.2", "v1.3"]:
            self.noise_type = out_type
            self.hidden_dim = hidden_dim

        self.conv1 = nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=True)
        self.norm = batch_norm
        self.act = nonlinearity
        self.block1 = nn.SequentialModule(
            #nn.MaskModule(in_type, 29, margin=1),
            self.conv1,
            self.norm,
            self.act
        )
        # the old output type is the input type to the next layer
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, out_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, 1 * [self.r2_act.irrep(1)])
        if version in ["v1.5"]:
            out_type = out_scalar_fields 
        else:
            out_type = out_scalar_fields + out_vector_field

        lastlayer = nn.R2Conv(in_type, out_type, kernel_size=3, padding=0, bias=True)
        if version in ["v1.1"]:
            lastlayer.bias = torch.nn.Parameter(lastlayer.bias.data+5, requires_grad=True)
        elif version in ["v1.3"]:
            lastlayer.bias = torch.nn.Parameter(lastlayer.bias.data+3, requires_grad=True)
        self.conv2 = lastlayer
        self.block7 = nn.SequentialModule(
            self.conv2
        )
        if version in ["v1.2", "v1.3"]:
            self.noise_embed = torch.nn.Sequential(
                torch.nn.Linear(1, 8),
                torch.nn.ReLU(),
                torch.nn.Linear(8, hidden_dim),
                torch.nn.Sigmoid()
            )
        elif version in ["v1.5"]:
            in_dim = out_dim+1 if eps else out_dim
            self.head = torch.nn.Sequential(
                torch.nn.Linear(in_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1)
            )
            with torch.no_grad():
                # self.head[-1].bias = torch.nn.Parameter(10*torch.ones(1).to(self.head[-1].bias.data))
                torch.nn.init.zeros_(self.head[-1].weight)
                torch.nn.init.ones_(self.head[-1].bias)

    def forward(self, *args, **kwargs):
        if self.version == "v1.0":
            return self._v1_0(*args, **kwargs)
        elif self.version == "v1.1":
            return self._v1_1(*args, **kwargs)
        elif self.version == "v1.2":
            return self._v1_2(*args, **kwargs)
        elif self.version == "v1.3":
            return self._v1_2(*args, **kwargs) #shared
        elif self.version == "v1.4":
            return self._v1_4(*args, **kwargs)
        elif self.version == "v1.5":
            return self._v1_5(*args, **kwargs)
        else:
            raise NotImplementedError

    def _v1_0(self, x: torch.Tensor):
        # x (128,3,32,32)
        # x = x.unsqueeze(1)
        #x = torch.nn.functional.pad(x, (0, 1, 0, 1), value=0).unsqueeze(1)
        x = nn.GeometricTensor(x, self.input_type)

        x = self.block1(x)
        x = self.block7(x)

        #x = x.tensor.squeeze(-1).squeeze(-1)
        x = x.tensor.mean(dim=(2, 3))

        x_0, x_1 = x[:, :self.out_dim], x[:, self.out_dim:]
        # x_0 (128,2) invariant value
        # x_1 (128,2) equivariant value

        # frame
        a, b = x_1[:, 0], x_1[:, 1]
        numer = a
        # threshold = 1e-9*torch.ones_like(a)
        # norm = torch.minimum(a**2+b**2, threshold)
        # denom = torch.minimum(norm.sqrt(), threshold)
        norm = a**2+b**2 + 1e-9
        denom = norm.sqrt() + 1e-9
        theta = torch.arccos(numer/denom)
        theta = torch.where(
            b >= 0,
            theta,
            2*math.pi - theta
        )
        theta = (1-theta/2/math.pi)

        # filter
        x_0 = x_0 + torch.tensor([[-5,5]]).to(x_0)
        x_0 = torch.sigmoid(x_0)
        # theta_min, theta_max = x_0[:,0], x_0[:,1]
        # equiv = torch.where(
        #     theta_min<theta_max,
        #     (theta_min<=theta)&(theta<=theta_max),
        #     (theta<=theta_max)|(theta_min<=theta)
        # )
        # theta = torch.where(
        #     equiv,
        #     theta,
        #     0.5*torch.ones_like(theta)
        #     # 1-theta
        # )
        return theta, x_0, x_1


    def _v1_1(self, x: torch.Tensor):
        # x (128,3,32,32)
        # x = x.unsqueeze(1)
        #x = torch.nn.functional.pad(x, (0, 1, 0, 1), value=0).unsqueeze(1)
        x = nn.GeometricTensor(x, self.input_type)

        x = self.block1(x)
        x = self.block7(x)

        #x = x.tensor.squeeze(-1).squeeze(-1)
        x = x.tensor.mean(dim=(2, 3))

        x_0, x_1 = x[:, :self.out_dim], x[:, self.out_dim:]
        # x_0 (128,1) invariant value
        # x_1 (128,2) equivariant value

        # frame
        a, b = x_1[:, 0], x_1[:, 1]
        numer = a
        # threshold = 1e-9*torch.ones_like(a)
        # norm = torch.minimum(a**2+b**2, threshold)
        # denom = torch.minimum(norm.sqrt(), threshold)
        norm = a**2+b**2 + 1e-9
        denom = norm.sqrt() + 1e-9
        cos = numer/denom
        ones = torch.ones_like(cos) - 1e-9
        cos = torch.maximum(cos, -ones)
        cos = torch.minimum(cos, ones)
        theta = torch.arccos(cos)
        theta = torch.where(
            b >= 0,
            theta,
            2*math.pi - theta
        )
        theta_min = (1-theta/2/math.pi)
        theta_diff = torch.sigmoid(x_0.squeeze(-1))
        return theta_min, torch.stack([theta_min, theta_diff], dim=-1), None

    def _v1_2(self, x: torch.Tensor, eps:torch.Tensor):
        # x (B,3,32,32)
        # eps (B,k)  0<=eps<1
        B, k = eps.size()
        eps = eps.float()
        eps = eps.unsqueeze(-1) # (B,k,1)
        eps = self.noise_embed(eps) # (B,k,hidden_dim)
        # x = (
        #     x.unsqueeze(1) + eps.view(*eps.shape, 1, 1)
        # ).view(-1, *x.shape[1:]) # (B*k, 3, 32, 32)
        x = x.repeat_interleave(k, dim=0)
        x = nn.GeometricTensor(x, self.input_type)

        x = self.conv1(x)

        _eps = torch.zeros_like(x.tensor)
        _eps[:, :self.hidden_dim] = eps.view(B*k, self.hidden_dim, 1, 1)
        eps = nn.GeometricTensor(_eps, self.noise_type)
        x = x+eps

        x = self.norm(x)
        x = self.act(x)
        x = self.conv2(x)

        #x = x.tensor.squeeze(-1).squeeze(-1)
        x = x.tensor.mean(dim=(2, 3))

        x_0, x_1 = x[:, :self.out_dim], x[:, self.out_dim:]
        # x_0 (B*k,1) invariant value -> logit of eps
        # x_1 (B*k,2) equivariant value

        # 
        return None, x_0.view(B, k), None


    def _v1_4(self, x: torch.Tensor):
        # x (B,3,32,32)
        # eps (B,k)  0<=eps<1
        x = nn.GeometricTensor(x, self.input_type)

        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv2(x)

        #x = x.tensor.squeeze(-1).squeeze(-1)
        x = x.tensor.mean(dim=(2, 3))

        x_0, x_1 = x[:, :self.out_dim], x[:, self.out_dim:]
        # x_0 (B,2) invariant value
        # x_1 (B,2) equivariant value

        x_0 = torch.sigmoid(x_0)
        return None, x_0, None

    def _v1_5(self, x: torch.Tensor, eps: torch.Tensor = None):
        # x (B,3,32,32)
        # eps (B,k)  0<=eps<1
        x = nn.GeometricTensor(x, self.input_type)

        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv2(x)

        x = x.tensor.mean(dim=(2, 3))
        # x_0 (B,d) invariant value

        if eps is not None:
            eps = eps.float()
            B,d = x.size()
            assert len(eps.size()) == 2
            k = eps.size(1)
            x = x.view(B, 1, d).repeat_interleave(k, dim=1).view(-1, d)
            # B*k,d
            eps = eps.view(-1, 1)
            # B*k,1
            x = torch.cat([x, eps], dim=-1)
            # B*k,d+1
            x = self.head(x)
            # B*k,1
            return x.view(B, k, 1)

        x = self.head(x)
        return x

class MLPFilter(torch.nn.Module):
    def __init__(self, in_ch, out_dim, hidden_dim=5, version="v1.0", eps=False):
        super().__init__()
        self.version = version
        self.out_dim=out_dim
        self.probs = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, hidden_dim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_dim, hidden_dim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Flatten(),
        )
        if eps:
            in_dim = 9*hidden_dim + 1
        else:
            in_dim = 9*hidden_dim
        self.eps = eps
        self.head = torch.nn.Sequential(
            torch.nn.Linear(in_dim,(9*hidden_dim)//2),
            torch.nn.ReLU(),
            torch.nn.Linear((9*hidden_dim)//2,1)
        )
        # with torch.no_grad():
        #     torch.nn.init.zeros_(self.head[-1].weight)
        #     torch.nn.init.ones_(self.head[-1].bias)
    def forward(self, x, eps: torch.Tensor=None):
        if self.eps:
            eps = eps.float()
            x = self.probs(x)
            B,d = x.size()
            assert len(x.size()) == 2
            assert len(eps.size()) == 2
            k = eps.size(1)
            x = x.view(B, 1, d).repeat_interleave(k, dim=1).view(-1, d)
            # B*k,d
            eps = eps.view(-1, 1)
            # B*k,1
            x = torch.cat([x, eps], dim=-1)
            # B*k,d+1
            x = self.head(x)
            # B*k,1
            return x.view(B, k, 1)


        return self.head(self.probs(x))