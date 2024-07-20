"""EfficientPhys_FSAM: Enabling Simple, Fast and Accurate Camera-Based Vitals Measurement
Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2023)
Xin Liu, Brial Hill, Ziheng Jiang, Shwetak Patel, Daniel McDuff
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from neural_methods.model.FactorizePhys.FSAM import FeaturesFactorizationModule



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


class EfficientPhys_FSAM(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, frame_depth=20, batch_size=1, img_size=36, channel='raw', device=None, debug=False):
        super(EfficientPhys_FSAM, self).__init__()
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

        self.debug = debug
        self.model_config = {
            "MD_FSAM": True,
            "MD_TYPE": "NMF",
            "MD_R": 1,
            "MD_S": frame_depth,
            "MD_STEPS": 4,
            "MD_INFERENCE": False,
            "MD_RESIDUAL": False,
            "INV_T": 1,
            "ETA": 0.9,
            "RAND_INIT": True,
            "in_channels": 3,
            "data_channels": 4,
            "MODE": "BVP",
            "align_channels": self.nb_filters2 // 8,
            "height": 72,
            "weight": 72,
            "batch_size": 4,
            "frames": 160,
            "debug": debug,
            "assess_latency": False,
            "num_trials": 20,
            "visualize": False,
            "ckpt_path": "",
            "data_path": "",
            "label_path": ""
        }

        self.feature_factorizer = FeaturesFactorizationModule(self.nb_filters2, self.device, self.model_config, dim="2D_TSM", debug=debug)
        self.fsam_norm = nn.InstanceNorm2d(self.nb_filters2)
        self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)


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

        if self.model_config["MD_INFERENCE"] or self.training or self.debug:
            if "NMF" in self.model_config["MD_TYPE"]:
                att_mask, appx_error = self.feature_factorizer(d6 - d6.min()) # to make it positive (>= 0)
            else:
                att_mask, appx_error = self.feature_factorizer(d6)

            if self.model_config["MD_RESIDUAL"]:
                # Multiplication with Residual connection
                x = torch.mul(d6 - d6.min() + self.bias1, att_mask - att_mask.min() + self.bias1)
                factorized_embeddings = self.fsam_norm(x)
                factorized_embeddings = d6 + factorized_embeddings
            else:
                # Multiplication
                x = torch.mul(d6 - d6.min() + self.bias1, att_mask - att_mask.min() + self.bias1)
                factorized_embeddings = self.fsam_norm(x)            

            d7 = self.avg_pooling_4(factorized_embeddings)

        else:
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
    # writer = SummaryWriter('runs/EfficientPhys_FSAM')

    batch_size = 4
    frames = 20    #duration*fs
    in_channels = 3
    height = 72
    width = 72
    num_of_gpu = 1
    base_len = num_of_gpu * frames
    assess_latency = False
    debug = True
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
    # Add one more frame for EfficientPhys_FSAM since it does torch.diff for the input
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
    net = EfficientPhys_FSAM(frame_depth=frames, img_size=height, batch_size=batch_size, debug=debug).to(device)
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