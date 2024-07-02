"""EfficientPhysFM: Enabling Simple, Fast and Accurate Camera-Based Vitals Measurement
Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2023)
Xin Liu, Brial Hill, Ziheng Jiang, Shwetak Patel, Daniel McDuff
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

model_config = {
    "INPUT_CHANNELS": 1,
    "MD_S": 1,
    "MD_R": 12,
    "TRAIN_STEPS": 6,
    "EVAL_STEPS": 6,
    "INV_T": 1,
    "ETA": 0.9,
    "RAND_INIT": True
}

class _MatrixDecompositionBase(nn.Module):
    def __init__(self, device, frame_depth, batch_size, MD_R, dim="2D"):
        super().__init__()

        self.frame_depth = frame_depth
        self.dim = dim
        self.S = model_config["MD_S"]
        # self.D = model_config["MD_D"]
        BN = batch_size * frame_depth
        # BN = frame_depth
        factor = 8
        # self.R = (BN // factor) if (BN // factor) % 2 == 0 else (BN // factor) + 1
        # # self.R = 2 * frame_depth
        # self.R = 4 * batch_size
        self.R = MD_R

        self.train_steps = model_config["TRAIN_STEPS"]
        self.eval_steps = model_config["EVAL_STEPS"]

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

        if self.dim == "3D":        # (B, C, T, H, W) -> (B * S, D, N)
            B, C, T, H, W = x.shape

            D = C * H * W // self.S
            N = T

            # D = C // self.S
            # N = T * H * W

            # D = T // self.S
            # N = C * H * W

            # D = T * H * W // self.S
            # N = C

            # D = C * H // self.S
            # N = T * W

            # D = T * C // self.S
            # N = H * W

            x = x.view(B * self.S, D, N)

            # print("C, T, H, W", C, T, H, W)
            # print("D", D)
            # print("R", self.R)
            # print("N", N)
            # print("x.shape", x.shape)

        elif self.dim == "2D":      # (B, C, H, W) -> (B * S, D, N)
            BN, C, H, W = x.shape
            # print("BN, C, H, W", BN, C, H, W)
            B = BN // self.frame_depth
            # D = C * H * W // self.S  # self.frame_depth  # C * H * W // self.S
            # N = self.frame_depth  # C * H * W // self.S  # self.frame_depth
            # D = C * self.frame_depth
            # N = H * W
            D = self.frame_depth
            N = C * H * W
            # B = 1
            # self.R = min(D, N) // max(C, self.frame_depth)   #since we need to have a rank lower than frame-depth and C
            # self.R = min(D, N) // 8   #since we need to have a rank lower than frame-depth and C
            x = x.view(B * self.S, D, N)

            # print("---")
            # print("D", D)
            # print("R", self.R)
            # print("N", N)
            # print("x.shape", x.shape)

        else:                       # (B, C, L) -> (B * S, D, N)
            B, C, L = x.shape
            D = C // self.S
            N = L
            x = x.view(B * self.S, D, N)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        if self.dim == "3D":
            # (B * S, D, N) -> (B, C, H, W)
            x = x.view(B, C, T, H, W)
        elif self.dim == "2D":
            # (B * S, D, N) -> (B, C, H, W)
            # x = x.view(B*self.frame_depth, C, H, W)
            x = x.view(BN, C, H, W)

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
    def __init__(self, device, frame_depth, batch_size, MD_R, dim="2D"):
        super().__init__(device, frame_depth, batch_size, MD_R, dim=dim)
        self.device = device
        self.inv_t = 1

    def _build_bases(self, B, S, D, R):
        bases = torch.rand((B * S, D, R)).to(self.device)
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


class ConvBNReLU(nn.Module):
    @classmethod
    def _same_paddings(cls, kernel_size):
        if kernel_size == (1, 1):
            return (0, 0)
        elif kernel_size == (3, 3):
            return (1, 1)

    def __init__(self, in_c, out_c,
                 kernel_size=(1, 1), stride=(1, 1), padding='same',
                 dilation=(1, 1), groups=1, act='relu', apply_bn=False):
        super().__init__()

        self.apply_bn = apply_bn
        if padding == 'same':
            padding = self._same_paddings(kernel_size)

        self.conv = nn.Conv2d(in_c, out_c,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              groups=groups,
                              bias=False)
        if self.apply_bn:
            self.bn = nn.BatchNorm2d(out_c)
        if act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.apply_bn:
            x = self.bn(x)
        x = self.act(x)

        return x


class FeaturesFactorizationModule(nn.Module):
    def __init__(self, device, frame_depth, batch_size, in_c, MD_R):
        super().__init__()

        self.device = device
        mid_c = in_c // 8

        self.pre_conv_block = nn.Sequential(
            nn.Conv2d(in_c, mid_c, (1, 1)),
            nn.ReLU(inplace=True)
        )

        self.md_block = NMF(self.device, frame_depth, batch_size, MD_R)

        self.post_conv_block = nn.Sequential(
            ConvBNReLU(mid_c, mid_c, kernel_size=(1, 1)),
            nn.Conv2d(mid_c, in_c, (1, 1), bias=False)
        )
        self.shortcut = nn.Sequential()
        self._init_weight()


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                N = m.kernel_size[0] * m.kernel_size[1] * m.out_channels 
                m.weight.data.normal_(0, np.sqrt(2. / N))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.pre_conv_block(x)
        x = self.md_block(x)
        x = self.post_conv_block(x)

        x = F.tanh(shortcut + torch.multiply(shortcut, x))
        # x = F.tanh(torch.multiply(shortcut, x))

        return x

    def online_update(self, bases):
        if hasattr(self.md_block, 'online_update'):
            self.md_block.online_update(bases)



class TSM(nn.Module):
    def __init__(self, n_segment=10, fold_div=3):
        super(TSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out.view(nt, c, h, w)


class EfficientPhysFM(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, frame_depth=20, batch_size=1, img_size=36, channel='raw', device=None):
        super(EfficientPhysFM, self).__init__()
        if device != None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device(0)
            else:
                self.device = torch.device("cpu")
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        # TSM layers
        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)
        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1),
                                  bias=True)
        self.motion_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1),
                                  bias=True)
        self.motion_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)

        # self.feature_factorizer_1 = FeaturesFactorizationModule(
        #     self.device, frame_depth, batch_size, self.nb_filters1, MD_R=20)
        self.feature_factorizer_2 = FeaturesFactorizationModule(
            self.device, frame_depth, batch_size, self.nb_filters2, MD_R=1)

        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_4 = nn.AvgPool2d(self.pool_size)
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)
        # Dense layers
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif img_size == 72:
            # self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)
            # self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
            self.final_dense_1 = nn.Linear(576, self.nb_dense, bias=True)
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
        else:
            raise Exception('Unsupported image size')
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)

        if self.in_channels == 1 or self.in_channels == 3:
            self.batch_norm = nn.BatchNorm2d(self.in_channels)
        elif self.in_channels == 4:
            self.rgb_norm = nn.BatchNorm2d(3)
            self.thermal_norm = nn.BatchNorm2d(1)
        else:
            print("Unsupported input channels")

        self.channel = channel

    def forward(self, inputs, params=None):
        [batch, channel, width, height] = inputs.shape
        inputs = torch.diff(inputs, dim=0)

        if self.in_channels == 1:
            inputs = self.batch_norm(inputs[:, -1:, :, :])
        elif self.in_channels == 3:
            inputs = self.batch_norm(inputs[:, :3, :, :])
        elif self.in_channels == 4:
            rgb_inputs = self.rgb_norm(inputs[:, :3, :, :])
            thermal_inputs = self.thermal_norm(inputs[:, -1:, :, :])
            inputs = torch.concat([rgb_inputs, thermal_inputs], dim=1)
        else:
            try:
                print("Specified input channels:", self.in_channels)
                print("Data channels", channel)
                assert self.in_channels <= channel
            except:
                print(
                    "Incorrectly preprocessed data provided as input. Number of channels exceed the specified or default channels")
                print("Default or specified channels:", self.in_channels)
                print("Data channels [B, C, W, H]", inputs.shape)
                print("Exiting")
                exit()

        network_input = self.TSM_1(inputs)
        d1 = torch.tanh(self.motion_conv1(network_input))
        d1 = self.TSM_2(d1)
        d2 = torch.tanh(self.motion_conv2(d1))
        # print("d2.shape", d2.shape)

        # d2 = self.feature_factorizer_1(d2)

        d3 = self.avg_pooling_1(d2)
        d4 = self.dropout_1(d3)

        d4 = self.TSM_3(d4)
        d5 = torch.tanh(self.motion_conv3(d4))

        d5 = self.avg_pooling_2(d5)
        d5 = self.dropout_2(d5)

        d5 = self.TSM_4(d5)
        d6 = torch.tanh(self.motion_conv4(d5))

        d6 = self.avg_pooling_3(d6)
        d6 = self.dropout_3(d6)

        d6 = self.feature_factorizer_2(d6)

        d7 = self.avg_pooling_4(d6)
        d8 = self.dropout_3(d7)

        d9 = d8.view(d8.size(0), -1)

        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        out = self.final_dense_2(d11)

        return out


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import numpy as np
    # from torch.utils.tensorboard import SummaryWriter

    # default `log_dir` is "runs" - we'll be more specific here
    # writer = SummaryWriter('runs/EfficientPhysFM')

    batch_size = 4
    frames = 40    #duration*fs
    in_channels = 3
    height = 72
    width = 72
    num_of_gpu = 1
    base_len = num_of_gpu * frames
    assess_latency = False
    # assess_latency = True

    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device("cpu")

    # test_data = torch.rand(batch_size, frames, in_channels, height, width).to(device)
    test_data = torch.rand(batch_size, in_channels, frames, height, width).to(device)
    print("test_data.shape", test_data.shape)
    labels = torch.rand(batch_size, frames)
    # print("org labels.shape", labels.shape)
    labels = labels.view(-1, 1)
    # print("view labels.shape", labels.shape)

    N, C, D, H, W = test_data.shape
    print(test_data.shape)

    test_data = test_data.view(N * D, C, H, W)

    test_data = test_data[:(N * D) // base_len * base_len]
    # Add one more frame for EfficientPhysFM since it does torch.diff for the input
    last_frame = torch.unsqueeze(test_data[-1, :, :, :], 0).repeat(num_of_gpu, 1, 1, 1)
    test_data = torch.cat((test_data, last_frame), 0)

    labels = labels[:(N * D) // base_len * base_len]
    # print("s1 labels.shape", labels.shape)
    last_sample = torch.unsqueeze(labels[-1, :], 0).repeat(num_of_gpu, 1)
    # print("s2 labels.shape", labels.shape)

    labels = torch.cat((labels, last_sample), 0)
    # print("s3 labels.shape", labels.shape)
    labels = torch.diff(labels, dim=0)
    # print("s4 labels.shape", labels.shape)
    labels = labels / torch.std(labels)  # normalize
    labels[torch.isnan(labels)] = 0
    # print("s5 labels.shape", labels.shape)

    # print("After: test_data.shape", test_data.shape)
    # exit()
    net = EfficientPhysFM(frame_depth=frames, img_size=height, batch_size=batch_size).to(device)
    net.eval()
    
    if assess_latency:
        num_trials = 10
        time_vec = []
        for passes in range(num_trials):
            t0 = time.time()
            pred = net(test_data)
            t1 = time.time()
            time_vec.append(t1-t0)
        
        print("Average time: ", np.median(time_vec))
        plt.plot(time_vec)
        plt.show()
    else:
        pred = net(test_data)

    # print("-"*100)
    # print(net)
    # print("-"*100)

    print("pred.shape", pred.shape)

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print("Total parameters = ", pytorch_total_params)

    pytorch_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Trainable parameters = ", pytorch_trainable_params)

    # writer.add_graph(net, test_data)
    # writer.close()