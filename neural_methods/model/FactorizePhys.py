"""
FactorizePhys: Effective Spatial-Temporal Attention in Remote Photo-plethysmography through Factorization of Voxel Embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

import numpy as np

# num_filters
nf = [4, 8, 8, 8]

model_config = {
    "MD_FSAM": True,
    "MD_TYPE": "NMF",
    "MD_R": 1,
    "MD_S": 1,
    "MD_STEPS": 4,
    "INV_T": 1,
    "ETA": 0.9,
    "RAND_INIT": True,
    "in_channels": 3,
    "data_channels": 4,
    "align_channels": nf[3]//2,
    "height": 72,
    "weight": 72,
    "batch_size": 2,
    "frames": 160,
    "debug": False,
    "assess_latency": True,
    "num_trials": 20,
    "visualize": False,
    "ckpt_path": "",
    "data_path": "data/1003_input13.npy",
    "label_path": "data/1003_label3.npy"
}

class _MatrixDecompositionBase(nn.Module):
    def __init__(self, device, md_config, debug=False, dim="3D"):
        super().__init__()

        self.dim = dim
        self.md_type = md_config["MD_TYPE"]
        self.S = md_config["MD_S"]
        self.R = md_config["MD_R"]
        self.debug = debug

        self.train_steps = md_config["MD_STEPS"]
        self.eval_steps = md_config["MD_STEPS"]

        self.inv_t = model_config["INV_T"]
        self.eta = model_config["ETA"]

        self.rand_init = model_config["RAND_INIT"]
        self.device = device

        # print('Dimension:', self.dim)
        # print('S', self.S)
        # print('D', self.D)
        # print('R', self.R)
        # print('train_steps', self.train_steps)
        # print('eval_steps', self.eval_steps)
        # print('inv_t', self.inv_t)
        # print('eta', self.eta)
        # print('rand_init', self.rand_init)

    def _build_bases(self, B, S, D, R):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):

        if self.debug:
            print("Org x.shape", x.shape)

        if self.dim == "3D":        # (B, C, T, H, W) -> (B * S, D, N)
            B, C, T, H, W = x.shape

            # # dimension of vector of our interest is T (rPPG signal as T dimension), so forming this as vector
            # # From spatial and channel dimension, which are features, only 2-4 shall be enough to generate the approximated attention matrix
            D = T // self.S
            N = C * H * W 

            # # smoothening the temporal dimension
            # x = x.view(B * self.S, N, D)
            # # print("Intermediate-1 x", x.shape)

            # sample_1 = x[:, :, 0].unsqueeze(2)
            # sample_2 = x[:, :, -1].unsqueeze(2)
            # x = torch.cat([sample_1, x, sample_2], dim=2)
            # gaussian_kernel = [1.0, 1.0, 1.0]
            # kernels = torch.FloatTensor([[gaussian_kernel]]).repeat(N, N, 1).to(self.device)
            # bias = torch.FloatTensor(torch.zeros(N)).to(self.device)
            # x = F.conv1d(x, kernels, bias=bias, padding="valid")
            # x = (x - x.min()) / (x.max() - x.min())

            # x = x.permute(0, 2, 1)
            # # print("Intermediate-2 x", x.shape)

            x = x.view(B * self.S, D, N)

        elif self.dim == "2D":      # (B, C, H, W) -> (B * S, D, N)
            B, C, H, W = x.shape
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)

        elif self.dim == "1D":                       # (B, C, L) -> (B * S, D, N)
            B, C, L = x.shape
            D = L // self.S
            N = C
            x = x.view(B * self.S, D, N)

        else:
            print("Dimension not supported")
            exit()

        if self.debug:
            print("MD_Type", self.md_type)
            print("MD_S", self.S)
            print("MD_D", D)
            print("MD_N", N)
            print("MD_R", self.R)
            print("MD_TRAIN_STEPS", self.train_steps)
            print("MD_EVAL_STEPS", self.eval_steps)
            print("x.view(B * self.S, D, N)", x.shape)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R)
        else:
            bases = self.bases.repeat(B, 1, 1).to(self.device)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))


        if self.dim == "3D":

            apply_smoothening = False
            if apply_smoothening:
                # smoothening the temporal dimension
                x = x.view(B, D * self.S, N)    #Joining temporal dimension for contiguous smoothening
                # print("Intermediate-0 x", x.shape)            
                x = x.permute(0, 2, 1)
                # print("Intermediate-1 x", x.shape)

                sample_1 = x[:, :, 0].unsqueeze(2)
                # sample_2 = x[:, :, 0].unsqueeze(2)
                sample_3 = x[:, :, -1].unsqueeze(2)
                # sample_4 = x[:, :, -1].unsqueeze(2)
                x = torch.cat([sample_1, x, sample_3], dim=2)
                # x = torch.cat([sample_1, sample_2, x, sample_3, sample_4], dim=2)
                # gaussian_kernel = [0.25, 0.50, 0.75, 0.50, 0.25]
                # gaussian_kernel = [0.33, 0.66, 1.00, 0.66, 0.33]
                # gaussian_kernel = [0.3, 0.7, 1.0, 0.7, 0.3]
                # gaussian_kernel = [0.3, 1.0, 1.0, 1.0, 0.3]
                # gaussian_kernel = [0.20, 0.80, 1.00, 0.80, 0.20]
                # gaussian_kernel = [1.0, 1.0, 1.0]
                gaussian_kernel = [0.8, 1.0, 0.8]
                kernels = torch.FloatTensor([[gaussian_kernel]]).repeat(N, N, 1).to(self.device)
                bias = torch.FloatTensor(torch.zeros(N)).to(self.device)
                x = F.conv1d(x, kernels, bias=bias, padding="valid")
                # x = (x - x.min()) / (x.max() - x.min())
                # x = (x - x.mean()) / (x.std())
                # x = x - x.min()
                x = (x - x.min())/(x.std())

                # print("Intermediate-2 x", x.shape)

            # (B * S, D, N) -> (B, C, T, H, W)
            x = x.view(B, C, T, H, W)
        elif self.dim == "2D":
            # (B * S, D, N) -> (B, C, H, W)
            x = x.view(B, C, H, W)
        else:
            # (B * S, D, N) -> (B, C, L)
            x = x.view(B, C, L)

        # (B * L, D, R) -> (B, L, N, D)
        bases = bases.view(B, self.S, D, self.R)

        if not self.rand_init and not self.training and not return_bases:
            self.online_update(bases)

        # if not self.rand_init or return_bases:
        #     return x, bases
        # else:
        return x

    @torch.no_grad()
    def online_update(self, bases):
        # (B, S, D, R) -> (S, D, R)
        update = bases.mean(dim=0)
        self.bases += self.eta * (update - self.bases)
        self.bases = F.normalize(self.bases, dim=1)


class NMF(_MatrixDecompositionBase):
    def __init__(self, device, md_config, debug=False, dim="3D"):
        super().__init__(device, md_config, debug=debug, dim=dim)
        self.device = device
        self.inv_t = 1

    def _build_bases(self, B, S, D, R):
        # bases = torch.rand((B * S, D, R)).to(self.device)
        bases = torch.ones((B * S, D, R)).to(self.device)
        bases = F.normalize(bases, dim=1)

        return bases

    @torch.no_grad()
    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class _SmoothMatrixDecompositionBase(nn.Module):
    def __init__(self, device, md_config, debug=False, dim="3D"):
        super().__init__()

        self.dim = dim
        self.md_type = md_config["MD_TYPE"]
        self.S = md_config["MD_S"]
        self.R = md_config["MD_R"]
        self.debug = debug

        self.train_steps = md_config["MD_STEPS"]
        self.eval_steps = md_config["MD_STEPS"]

        self.inv_t = model_config["INV_T"]
        self.eta = model_config["ETA"]

        self.rand_init = model_config["RAND_INIT"]
        self.device = device

        # print('Dimension:', self.dim)
        # print('S', self.S)
        # print('D', self.D)
        # print('R', self.R)
        # print('train_steps', self.train_steps)
        # print('eval_steps', self.eval_steps)
        # print('inv_t', self.inv_t)
        # print('eta', self.eta)
        # print('rand_init', self.rand_init)

    def _build_bases(self, B, S, D, R):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    @torch.no_grad()
    def local_inference(self, x, rbfs, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), torch.bmm(rbfs, bases))
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, rbfs, bases, coef)

        return bases, coef

    def compute_coef(self, x, rbfs, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):

        if self.debug:
            print("Org x.shape", x.shape)

        if self.dim == "3D":        # (B, C, T, H, W) -> (B * S, D, N)
            B, C, T, H, W = x.shape

            # # dimension of vector of our interest is T (rPPG signal as T dimension), so forming this as vector
            # # From spatial and channel dimension, which are features, only 2-4 shall be enough to generate the approximated attention matrix
            D = T // self.S
            N = C * H * W
            x = x.view(B * self.S, D, N)

        elif self.dim == "2D":      # (B, C, H, W) -> (B * S, D, N)
            B, C, H, W = x.shape
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)

        elif self.dim == "1D":                       # (B, C, L) -> (B * S, D, N)
            B, C, L = x.shape
            D = L // self.S
            N = C
            x = x.view(B * self.S, D, N)

        else:
            print("Dimension not supported")
            exit()

        P = D
        sig0 = torch.tensor(5.0)
        sig1 = sig0 * 2
        sig2 = sig0 * 4
        sig3 = sig0 * 6
        sig4 = sig0 * 8
        sig5 = sig0 * 10

        # dt = torch.tensor((P - 1) / (D - 1))
        tt = torch.arange(0, D).unsqueeze(1)
        nn = torch.arange(0, P).unsqueeze(0)
        # print(tt.shape, nn.shape)
        dd = torch.pow(tt - nn, 2)
        rbf0 = torch.exp((-0.5 * dd)/(2 * torch.pow(sig0, 2)))
        rbf1 = torch.exp((-0.5 * dd)/(2 * torch.pow(sig1, 2)))
        rbf2 = torch.exp((-0.5 * dd)/(2 * torch.pow(sig2, 2)))
        rbf3 = torch.exp((-0.5 * dd)/(2 * torch.pow(sig3, 2)))
        rbf4 = torch.exp((-0.5 * dd)/(2 * torch.pow(sig4, 2)))
        rbf5 = torch.exp((-0.5 * dd)/(2 * torch.pow(sig5, 2)))
        rbfN = torch.ones(P, 1)

        rbfs = torch.cat([
            rbf0[:, torch.arange(0, P, 1)],
            rbf1[:, torch.arange(0, P, 2)],
            rbf2[:, torch.arange(0, P, 4)],
            rbf3[:, torch.arange(0, P, 6)],
            rbf4[:, torch.arange(0, P, 8)],
            rbf5[:, torch.arange(0, P, 10)],
            rbfN,
            ], dim=1)

        rbfs = rbfs.repeat(B * self.S, 1, 1).to(self.device)
        rbf_shape2 = rbfs.shape[2]

        if self.debug:
            print("MD_Type", self.md_type)
            print("MD_S", self.S)
            print("MD_D", D)
            print("MD_N", N)
            print("MD_R", self.R)
            print("MD_TRAIN_STEPS", self.train_steps)
            print("MD_EVAL_STEPS", self.eval_steps)
            print("x.view(B * self.S, D, N)", x.shape)
            print("rbfs.shape", rbfs.shape)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, rbf_shape2, self.R)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, rbf_shape2, self.R)
        else:
            bases = self.bases.repeat(B, 1, 1).to(self.device)

        bases, coef = self.local_inference(x, rbfs, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, rbfs, bases, coef)

        bases_prod = torch.bmm(rbfs, bases)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases_prod, coef.transpose(1, 2))


        if self.dim == "3D":
            # (B * S, D, N) -> (B, C, T, H, W)
            x = x.view(B, C, T, H, W)
        elif self.dim == "2D":
            # (B * S, D, N) -> (B, C, H, W)
            x = x.view(B, C, H, W)
        else:
            # (B * S, D, N) -> (B, C, L)
            x = x.view(B, C, L)

        # (B * L, D, R) -> (B, L, N, D)
        # bases_prod_updated = torch.bmm(rbfs, bases)
        bases = torch.bmm(rbfs.transpose(1, 2), bases_prod)
        bases = bases.view(B, self.S, rbf_shape2, self.R)

        if not self.rand_init and not self.training and not return_bases:
            print("it comes here")
            self.online_update(bases)

        # if not self.rand_init or return_bases:
        #     return x, bases
        # else:
        return x

    @torch.no_grad()
    def online_update(self, bases):
        # (B, S, D, R) -> (S, D, R)
        update = bases.mean(dim=0)
        self.bases += self.eta * (update - self.bases)
        self.bases = F.normalize(self.bases, dim=1)


class Smooth_NMF(_SmoothMatrixDecompositionBase):
    def __init__(self, device, md_config, debug=False, dim="3D"):
        super().__init__(device, md_config, debug=debug, dim=dim)
        self.device = device
        self.inv_t = 1

    def _build_bases(self, B, S, D, R):
        bases = torch.rand((B * S, D, R)).to(self.device)
        # bases = torch.ones((B * S, D, R)).to(self.device)
        bases = F.normalize(bases, dim=1)

        return bases

    @torch.no_grad()
    def local_step(self, x, rbfs, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)

        bases_prod = torch.bmm(rbfs, bases)

        numerator = torch.bmm(x.transpose(1, 2), bases_prod)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases_prod.transpose(1, 2).bmm(bases_prod))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases_prod.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update

        # bases = (bases_prod * numerator / (denominator + 1e-6))
        bases = torch.bmm(rbfs.transpose(1, 2), (bases_prod * numerator /
                          (denominator + 1e-6)))
        # print(bases.shape)
        # exit()

        return bases, coef

    def compute_coef(self, x, rbfs, bases, coef):
        bases_prod = torch.bmm(rbfs, bases)

        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases_prod)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases_prod.transpose(1, 2).bmm(bases_prod))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class VQ(_MatrixDecompositionBase):
    def __init__(self, device, md_config, debug=False, dim="3D"):
        super().__init__(device, md_config, debug=debug, dim=dim)
        self.device = device

    def _build_bases(self, B, S, D, R):
        # bases = torch.randn((B * S, D, R)).to(self.device)
        bases = torch.ones((B * S, D, R)).to(self.device)
        bases = F.normalize(bases, dim=1)
        return bases

    @torch.no_grad()
    def local_step(self, x, bases, _):
        # (B * S, D, N), normalize x along D (for cosine similarity)
        std_x = F.normalize(x, dim=1)

        # (B * S, D, R), normalize bases along D (for cosine similarity)
        std_bases = F.normalize(bases, dim=1, eps=1e-6)

        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(std_x.transpose(1, 2), std_bases)

        # softmax along R
        coef = F.softmax(self.inv_t * coef, dim=-1)

        # normalize along N
        coef = coef / (1e-6 + coef.sum(dim=1, keepdim=True))

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        bases = torch.bmm(x, coef)

        return bases, coef


    def compute_coef(self, x, bases, _):
        with torch.no_grad():
            # (B * S, D, N) -> (B * S, 1, N)
            x_norm = x.norm(dim=1, keepdim=True)

        # (B * S, D, N) / (B * S, 1, N) -> (B * S, D, N)
        std_x = x / (1e-6 + x_norm)

        # (B * S, D, R), normalize bases along D (for cosine similarity)
        std_bases = F.normalize(bases, dim=1, eps=1e-6)

        # (B * S, N, D)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(std_x.transpose(1, 2), std_bases)

        # softmax along R
        coef = F.softmax(self.inv_t * coef, dim=-1)

        return coef


class ConvBNReLU(nn.Module):
    @classmethod
    def _same_paddings(cls, kernel_size, dim):
        if dim == "3D":
            if kernel_size == (1, 1, 1):
                return (0, 0, 0)
            elif kernel_size == (3, 3, 3):
                return (1, 1, 1)
        elif dim == "2D":
            if kernel_size == (1, 1):
                return (0, 0)
            elif kernel_size == (3, 3):
                return (1, 1)
        else:
            if kernel_size == 1:
                return 0
            elif kernel_size == 3:
                return 1

    def __init__(self, in_c, out_c, dim,
                 kernel_size=1, stride=1, padding='same',
                 dilation=1, groups=1, act='relu', apply_bn=False, apply_act=True):
        super().__init__()

        self.apply_bn = apply_bn
        self.apply_act = apply_act
        self.dim = dim
        if dilation == 1:
            if self.dim == "3D":
                dilation = (1, 1, 1)
            elif self.dim == "2D":
                dilation = (1, 1)
            else:
                dilation = 1

        if kernel_size == 1:
            if self.dim == "3D":
                kernel_size = (1, 1, 1)
            elif self.dim == "2D":
                kernel_size = (1, 1)
            else:
                kernel_size = 1

        if stride == 1:
            if self.dim == "3D":
                stride = (1, 1, 1)
            elif self.dim == "2D":
                stride = (1, 1)
            else:
                stride = 1

        if padding == 'same':
            padding = self._same_paddings(kernel_size, dim)

        if self.dim == "3D":
            self.conv = nn.Conv3d(in_c, out_c,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation,
                                  groups=groups,
                                  bias=False)
        elif self.dim == "2D":
            self.conv = nn.Conv2d(in_c, out_c,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation,
                                  groups=groups,
                                  bias=False)
        else:
            self.conv = nn.Conv1d(in_c, out_c,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation,
                                  groups=groups,
                                  bias=False)

        if act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = nn.ReLU(inplace=True)

        if self.apply_bn:
            if self.dim == "3D":
                self.bn = nn.InstanceNorm3d(out_c)
            elif self.dim == "2D":
                self.bn = nn.InstanceNorm2d(out_c)
            else:
                self.bn = nn.InstanceNorm1d(out_c)

    def forward(self, x):
        x = self.conv(x)
        if self.apply_act:
            x = self.act(x)
        if self.apply_bn:
            x = self.bn(x)
        return x


class FeaturesFactorizationModule(nn.Module):
    def __init__(self, inC, device, md_config, dim="3D", debug=False):
        super().__init__()

        self.device = device
        self.dim = dim
        md_type = md_config["MD_TYPE"]
        align_C = model_config["align_channels"]    #inC // 2  # // 2 #// 8

        if self.dim == "3D":
            if "nmf" in md_type.lower():
                self.pre_conv_block = nn.Sequential(
                    nn.Conv3d(inC, align_C, (1, 1, 1)), 
                    nn.ReLU(inplace=True))
            else:
                self.pre_conv_block = nn.Conv3d(inC, align_C, (1, 1, 1))
        elif self.dim == "2D":
            if "nmf" in md_type.lower():
                self.pre_conv_block = nn.Sequential(
                    nn.Conv2d(inC, align_C, (1, 1)),
                    nn.ReLU(inplace=True)
                    )
            else:
                self.pre_conv_block = nn.Conv2d(inC, align_C, (1, 1))
        elif self.dim == "1D":
            if "nmf" in md_type.lower():
                self.pre_conv_block = nn.Sequential(
                    nn.Conv1d(inC, align_C, 1),
                    nn.ReLU(inplace=True)
                    )
            else:
                self.pre_conv_block = nn.Conv1d(inC, align_C, 1)
        else:
            print("Dimension not supported")

        if "smooth_nmf" in md_type.lower():
            self.md_block = Smooth_NMF(self.device, md_config, dim=self.dim, debug=debug)
        elif "nmf" in md_type.lower():
            self.md_block = NMF(self.device, md_config, dim=self.dim, debug=debug)
        elif "vq" in md_type.lower():
            self.md_block = VQ(self.device, md_config, dim=self.dim, debug=debug)
        else:
            print("Unknown type specified for MD_TYPE:", md_type)
            exit()

        if self.dim == "3D":
            if "nmf" in md_type.lower():
                self.post_conv_block = nn.Sequential(
                    ConvBNReLU(align_C, align_C, dim=self.dim, kernel_size=1),
                    nn.Conv3d(align_C, inC, 1, bias=False)
                    )
            else:
                self.post_conv_block = nn.Sequential(
                    ConvBNReLU(align_C, align_C, dim=self.dim, kernel_size=1, apply_act=False), 
                    nn.Conv3d(align_C, inC, 1, bias=False)
                    )
        elif self.dim == "2D":
            if "nmf" in md_type.lower():
                self.post_conv_block = nn.Sequential(
                    ConvBNReLU(align_C, align_C, dim=self.dim, kernel_size=1), 
                    nn.Conv2d(align_C, inC, 1, bias=False)
                    )
            else:
                self.post_conv_block = nn.Sequential(
                    ConvBNReLU(align_C, align_C, dim=self.dim, kernel_size=1, apply_act=False),
                    nn.Conv2d(align_C, inC, 1, bias=False)
                    )
        else:
            if "nmf" in md_type.lower():
                self.post_conv_block = nn.Sequential(
                    ConvBNReLU(align_C, align_C, dim=self.dim, kernel_size=1),
                    nn.Conv1d(align_C, inC, 1, bias=False)
                    )
            else:
                self.post_conv_block = nn.Sequential(
                    ConvBNReLU(align_C, align_C, dim=self.dim, kernel_size=1, apply_act=False),
                    nn.Conv1d(align_C, inC, 1, bias=False)
                    )

        self._init_weight()


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                N = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / N))
            elif isinstance(m, nn.Conv2d):
                N = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / N))
            elif isinstance(m, nn.Conv1d):
                N = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / N))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.pre_conv_block(x)
        att = self.md_block(x)
        dist = torch.dist(x, att)
        att = self.post_conv_block(att)

        return att, dist

    def online_update(self, bases):
        if hasattr(self.md_block, 'online_update'):
            self.md_block.online_update(bases)


class ConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.conv_block_3d(x)


class encoder_block(nn.Module):
    def __init__(self, inCh, dropout_rate=0.1, debug=False):
        super(encoder_block, self).__init__()
        # inCh, out_channel, kernel_size, stride, padding

        self.debug = debug

        self.encoder = nn.Sequential(
            ConvBlock3D(inCh, nf[0], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            ConvBlock3D(nf[0], nf[1], [3, 3, 3], [1, 2, 2], [1, 1, 1]),
            ConvBlock3D(nf[1], nf[1], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf[1], nf[1], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            ConvBlock3D(nf[1], nf[2], [3, 3, 3], [1, 2, 2], [1, 1, 1]),
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            ConvBlock3D(nf[2], nf[3], [3, 3, 3], [1, 2, 2], [1, 1, 1]),
            ConvBlock3D(nf[3], nf[3], [3, 3, 3], [1, 1, 1], [1, 0, 0]),
            nn.Dropout3d(p=dropout_rate)
        )

    def forward(self, x):
        x = self.encoder(x)
        if self.debug:
            print("Encoder")
            print("     x.shape", x.shape)
        return x


class BVP_Head(nn.Module):
    def __init__(self, md_config, device, dropout_rate=0.1, debug=False):
        super(BVP_Head, self).__init__()
        self.debug = debug

        self.use_fsam = md_config["MD_FSAM"]
        self.md_type = md_config["MD_TYPE"]

        if self.use_fsam:
            inC = nf[3]
            self.fsam = FeaturesFactorizationModule(inC, device, md_config, dim="3D", debug=debug)
            self.fsam_norm = nn.InstanceNorm3d(inC)
            self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)
        else:
            inC = nf[3]

        self.conv_decoder = nn.Sequential(
            nn.Conv3d(inC, nf[0], (3, 4, 4), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.Tanh(),
            nn.InstanceNorm3d(nf[0]),

            nn.Dropout3d(p=dropout_rate),

            nn.Conv3d(nf[0], 1, (3, 4, 4), stride=(1, 1, 1), padding=(1, 0, 0)),
        )


    def forward(self, voxel_embeddings, batch, length):

        if self.debug:
            print("Decoder")
            print("     voxel_embeddings.shape", voxel_embeddings.shape)

        if (self.training or self.debug) and self.use_fsam:
            if "NMF" in self.md_type:
                att_mask, appx_error = self.fsam(voxel_embeddings - voxel_embeddings.min()) # to make it positive (>= 0)
            else:
                att_mask, appx_error = self.fsam(voxel_embeddings)

            if self.debug:
                print("att_mask.shape", att_mask.shape)

            # # directly use att_mask   ---> difficult to converge without Residual connection. Needs high rank
            # factorized_embeddings = self.fsam_norm(att_mask)

            # # Residual connection: 
            # factorized_embeddings = voxel_embeddings + self.fsam_norm(att_mask)

            # Multiplication
            x = torch.mul(voxel_embeddings - voxel_embeddings.min() + self.bias1, att_mask - att_mask.min() + self.bias1)
            factorized_embeddings = self.fsam_norm(x)

            # # Multiplication with Residual connection
            # x = torch.mul(voxel_embeddings - voxel_embeddings.min() + self.bias1, att_mask - att_mask.min() + self.bias1)
            # factorized_embeddings = self.fsam_norm(x)
            # factorized_embeddings = voxel_embeddings + factorized_embeddings
            
            # # Concatenate
            # factorized_embeddings = torch.cat([voxel_embeddings, self.fsam_norm(x)], dim=1)

            x = self.conv_decoder(factorized_embeddings)
        
        else:
            x = self.conv_decoder(voxel_embeddings)

        # x_1d = x.view(batch, -1, length)

        # if self.debug:
        #     print("     x.shape", x.shape)
        #     print("     x_1d.shape", x_1d.shape)

        # xa = self.final_conv1a(x_1d)
        # xb = self.final_conv1b(x_1d)
        # xc = self.final_conv1c(x_1d)

        # rPPG = self.final_conv(torch.cat([xa, xb, xc], dim=1))
        # rPPG = rPPG.view(-1, length)
        
        rPPG = x.view(-1, length)

        if self.debug:
            print("     rPPG.shape", rPPG.shape)
        
        if (self.training or self.debug) and self.use_fsam:
            return rPPG, factorized_embeddings, att_mask, appx_error
        else:
            return rPPG



class FactorizePhys(nn.Module):
    def __init__(self, frames, md_config, in_channels=3, dropout=0.2, device=torch.device("cpu"), debug=False):
        super(FactorizePhys, self).__init__()
        self.debug = debug

        self.in_channels = in_channels
        if self.in_channels == 1 or self.in_channels == 3:
            self.norm = nn.InstanceNorm3d(self.in_channels)
        elif self.in_channels == 4:
            self.rgb_norm = nn.InstanceNorm3d(3)
            self.thermal_norm = nn.InstanceNorm3d(1)
        else:
            print("Unsupported input channels")
        
        self.use_fsam = md_config["MD_FSAM"]

        if self.debug:
            print("nf:", nf)

        self.encoder = encoder_block(self.in_channels, dropout_rate=dropout, debug=debug)

        self.rppg_head = BVP_Head(md_config, device=device, dropout_rate=dropout, debug=debug)


        
    def forward(self, x): # [batch, Features=3, Temp=frames, Width=32, Height=32]
        
        [batch, channel, length, width, height] = x.shape
        
        # if self.in_channels == 1:
        #     x = x[:, :, :-1, :, :]
        # else:
        #     x = torch.diff(x, dim=2)
        
        x = torch.diff(x, dim=2)

        if self.debug:
            print("Input.shape", x.shape)

        if self.in_channels == 1:
            x = self.norm(x[:, -1:, :, :, :])
        elif self.in_channels == 3:
            x = self.norm(x[:, :3, :, :, :])
        elif self.in_channels == 4:
            rgb_x = self.rgb_norm(x[:, :3, :, :, :])
            thermal_x = self.thermal_norm(x[:, -1:, :, :, :])
            x = torch.concat([rgb_x, thermal_x], dim = 1)
        else:
            try:
                print("Specified input channels:", self.in_channels)
                print("Data channels", channel)
                assert self.in_channels <= channel
            except:
                print("Incorrectly preprocessed data provided as input. Number of channels exceed the specified or default channels")
                print("Default or specified channels:", self.in_channels)
                print("Data channels [B, C, N, W, H]", x.shape)
                print("Exiting")
                exit()

        if self.debug:
            print("Diff Normalized shape", x.shape)

        voxel_embeddings = self.encoder(x)
        if self.debug:
            print("voxel_embeddings.shape", voxel_embeddings.shape)
        
        if (self.training or self.debug) and self.use_fsam:
            rPPG, factorized_embeddings, att_mask, appx_error = self.rppg_head(voxel_embeddings, batch, length-1)
        else:
            rPPG = self.rppg_head(voxel_embeddings, batch, length-1)

        # if self.debug:
        #     print("rppg_feats.shape", rppg_feats.shape)

        # rPPG = rppg_feats.view(-1, length-1)

        if self.debug:
            print("rPPG.shape", rPPG.shape)

        if (self.training or self.debug) and self.use_fsam:
            return rPPG, voxel_embeddings, factorized_embeddings, att_mask, appx_error
        else:
            return rPPG, voxel_embeddings
    

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import resample
    # from torch.utils.tensorboard import SummaryWriter

    # default `log_dir` is "runs" - we'll be more specific here
    # writer = SummaryWriter('runs/FactorizePhys')

    # ckpt_path = "/Users/jiteshjoshi/Downloads/rPPG_Testing/models/PURE_PURE_iBVP_FactorizePhys_FactorizePhys_5_Epoch6.pth"
    # ckpt_path = "/Users/jiteshjoshi/Downloads/rPPG_Testing/models/PURE_PURE_iBVP_FactorizePhys_FactorizePhys_5_Epoch12.pth"
    # ckpt_path = "/Users/jiteshjoshi/Downloads/rPPG_Testing/models/PURE_PURE_iBVP_FactorizePhys_FactorizePhys_5_Epoch19.pth"
    ckpt_path = model_config["ckpt_path"]

    # data_path = "/Users/jiteshjoshi/Downloads/rPPG_Testing/data/subject36_input9.npy"
    data_path = model_config["data_path"]

    label_path = model_config["label_path"]

    use_fsam = model_config["MD_FSAM"]
    batch_size = model_config["batch_size"]
    frames = model_config["frames"]
    in_channels = model_config["in_channels"]
    data_channels = model_config["data_channels"]
    height = model_config["height"]
    width = model_config["weight"]
    debug = bool(model_config["debug"])
    assess_latency = bool(model_config["assess_latency"])
    num_trials = model_config["num_trials"]
    visualize = model_config["visualize"]

    # if torch.cuda.is_available():
    #     device = torch.device(0)
    # else:
    device = torch.device("cpu")

    if visualize:
        np_data = np.load(data_path)
        np_label = np.load(label_path)
        np_label = np.expand_dims(np_label, 0)

        print("Chunk data shape", np_data.shape)
        print("Chunk label shape", np_label.shape)
        print("Min Max of input data:", np.min(np_data), np.max(np_data))
        # exit()

        test_data = np.transpose(np_data, (3, 0, 1, 2))
        test_data = torch.from_numpy(test_data)
        test_data = test_data.unsqueeze(0)

        last_frame = torch.unsqueeze(test_data[:, :, -1, :, :], 2).repeat(1, 1, 1, 1, 1)
        test_data = torch.cat((test_data, last_frame), 2)
    else:
        # test_data = torch.rand(batch_size, in_channels, frames, height, width).to(device)
        test_data = torch.rand(batch_size, data_channels, frames + 1, height, width)

    test_data = test_data.to(torch.float32).to(device)
    # print(test_data.shape)
    # exit()
    md_config = {}
    md_config["MD_S"] = model_config["MD_S"]
    md_config["MD_R"] = model_config["MD_R"]
    md_config["MD_STEPS"] = model_config["MD_STEPS"]
    md_config["MD_FSAM"] = model_config["MD_FSAM"]
    md_config["MD_TYPE"] = model_config["MD_TYPE"]
    net = nn.DataParallel(FactorizePhys(frames=frames, md_config=md_config, device=device, in_channels=in_channels, debug=debug)).to(device)
    # net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()

    if assess_latency:
        time_vec = []
        if debug:
            appx_error_list = []
        for passes in range(num_trials):
            t0 = time.time()
            if (net.training or debug) and use_fsam:
                pred, vox_embed, factorized_embed, att_mask, appx_error = net(test_data)
            else:
                pred, vox_embed = net(test_data)
            t1 = time.time()
            time_vec.append(t1-t0)
            if debug:
                appx_error_list.append(appx_error.item())

        print("Median time: ", np.median(time_vec))
        if debug:
            print("Median error:", np.median(appx_error_list))
        plt.plot(time_vec)
        plt.show()
    else:
        if (net.training or debug) and use_fsam:
            pred, vox_embed, factorized_embed, att_mask, appx_error = net(test_data)
            print("Appx error: ", appx_error.item())  # .detach().numpy())
        else:
            pred, vox_embed = net(test_data)

    # print("-"*100)
    # print(net)
    # print("-"*100)

    if visualize:
        test_data = test_data.detach().numpy()
        vox_embed = vox_embed.detach().numpy()
        if use_fsam:
            factorized_embed = factorized_embed.detach().numpy()
            att_mask = att_mask.detach().numpy()

        # print(test_data.shape, vox_embed.shape, factorized_embed.shape)
        b, ch, enc_frames, enc_height, enc_width = vox_embed.shape
        # exit()
        for ch in range(vox_embed.shape[1]):
            if use_fsam:
                fig, ax = plt.subplots(9, 4, layout="tight")
            else:
                fig, ax = plt.subplots(9, 2, layout="tight")

            frame = 0
            ax[0, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[0, 0].axis('off')
            ax[0, 1].imshow(vox_embed[0, ch, frame, :, :])
            ax[0, 1].axis('off')
            if use_fsam:
                ax[0, 2].imshow(factorized_embed[0, ch, frame, :, :])
                ax[0, 2].axis('off')
                ax[0, 3].imshow(att_mask[0, ch, frame, :, :])
                ax[0, 3].axis('off')

            frame = 20
            ax[1, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[1, 0].axis('off')
            ax[1, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[1, 1].axis('off')
            if use_fsam:
                ax[1, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
                ax[1, 2].axis('off')
                ax[1, 3].imshow(att_mask[0, ch, frame//4, :, :])
                ax[1, 3].axis('off')

            frame = 40
            ax[2, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[2, 0].axis('off')
            ax[2, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[2, 1].axis('off')
            if use_fsam:
                ax[2, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
                ax[2, 2].axis('off')
                ax[2, 3].imshow(att_mask[0, ch, frame//4, :, :])
                ax[2, 3].axis('off')

            frame = 60
            ax[3, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[3, 0].axis('off')
            ax[3, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[3, 1].axis('off')
            if use_fsam:
                ax[3, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
                ax[3, 2].axis('off')
                ax[3, 3].imshow(att_mask[0, ch, frame//4, :, :])
                ax[3, 3].axis('off')

            frame = 80
            ax[4, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[4, 0].axis('off')
            ax[4, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[4, 1].axis('off')
            if use_fsam:
                ax[4, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
                ax[4, 2].axis('off')
                ax[4, 3].imshow(att_mask[0, ch, frame//4, :, :])
                ax[4, 3].axis('off')

            frame = 100
            ax[5, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[5, 0].axis('off')
            ax[5, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[5, 1].axis('off')
            if use_fsam:
                ax[5, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
                ax[5, 2].axis('off')
                ax[5, 3].imshow(att_mask[0, ch, frame//4, :, :])
                ax[5, 3].axis('off')

            frame = 120
            ax[6, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[6, 0].axis('off')
            ax[6, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[6, 1].axis('off')
            if use_fsam:
                ax[6, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
                ax[6, 2].axis('off')
                ax[6, 3].imshow(att_mask[0, ch, frame//4, :, :])
                ax[6, 3].axis('off')

            frame = 140
            ax[7, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[7, 0].axis('off')
            ax[7, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[7, 1].axis('off')
            if use_fsam:
                ax[7, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
                ax[7, 2].axis('off')
                ax[7, 3].imshow(att_mask[0, ch, frame//4, :, :])
                ax[7, 3].axis('off')

            frame = 159
            ax[8, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[8, 0].axis('off')
            ax[8, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[8, 1].axis('off')
            if use_fsam:
                ax[8, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
                ax[8, 2].axis('off')
                ax[8, 3].imshow(att_mask[0, ch, frame//4, :, :])
                ax[8, 3].axis('off')

            plt.show()
            plt.close(fig)
    print("pred.shape", pred.shape)

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print("Total parameters = ", pytorch_total_params)

    pytorch_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Trainable parameters = ", pytorch_trainable_params)

    # writer.add_graph(net, test_data)
    # writer.close()