"""
FactorizePhys: Effective Spatial-Temporal Attention in Remote Photo-plethysmography through Factorization of Voxel Embeddings
"""

import torch
import torch.nn as nn
from FSAM import FeaturesFactorizationModule


class ConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding=padding, bias=False),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.conv_block_3d(x)


class encoder_block(nn.Module):
    def __init__(self, inCh, nf, dropout_rate=0.1, debug=False):
        super(encoder_block, self).__init__()
        # inCh, out_channel, kernel_size, stride, padding

        self.debug = debug
        #                                                        Input: #B, inCh, 160, 72, 72
        self.encoder1 = nn.Sequential(
            ConvBlock3D(inCh, nf[0], [3, 3, 3], [1, 1, 1], [1, 0, 0]),  #B, nf[0], 160, 70, 70
            ConvBlock3D(nf[0], nf[1], [3, 3, 3], [1, 2, 2], [1, 0, 0]), #B, nf[1], 160, 34, 34
            ConvBlock3D(nf[1], nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf[1], 160, 32, 32
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf[1], nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf[1], 160, 30, 30
            ConvBlock3D(nf[1], nf[2], [3, 3, 3], [1, 2, 2], [1, 0, 0]), #B, nf[2], 160, 14, 14
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf[2], 160, 12, 12
            nn.Dropout3d(p=dropout_rate),
        )

        self.encoder2 = nn.Sequential(
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf[2], 160, 10, 10
            ConvBlock3D(nf[2], nf[3], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf[3], 160, 8, 8
            ConvBlock3D(nf[3], nf[3], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf[3], 160, 6, 6
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x):
        hidden_embeddings = self.encoder1(x)
        voxel_embeddings = self.encoder2(hidden_embeddings)
        if self.debug:
            print("Encoder")
            print("     hidden_embeddings.shape", hidden_embeddings.shape)
            print("     voxel_embeddings.shape", voxel_embeddings.shape)
        return hidden_embeddings, voxel_embeddings


class PhysHead(nn.Module):
    def __init__(self, md_config, device, dropout_rate=0.1, debug=False):
        super(PhysHead, self).__init__()
        self.debug = debug

        self.use_fsam = md_config["MD_FSAM"]
        self.md_type = md_config["MD_TYPE"]
        self.md_infer = md_config["MD_INFERENCE"]
        self.md_res = md_config["MD_RESIDUAL"]
        self.nf = md_config["num_filters"]

        if self.use_fsam:
            inC = self.nf[3]
            self.fsam = FeaturesFactorizationModule(inC, device, md_config, dim="3D", debug=debug)
            self.fsam_norm = nn.InstanceNorm3d(inC)
            self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)
        else:
            inC = self.nf[3]

        self.conv_decoder = nn.Sequential(
            nn.Conv3d(inC, self.nf[0], (3, 4, 4), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),  #B, nf[0], 160, 3, 3
            nn.Tanh(),
            nn.InstanceNorm3d(self.nf[0]),

            nn.Conv3d(self.nf[0], 1, (5, 3, 3), stride=(1, 1, 1), padding=(2, 0, 0), bias=False),    #B, 1, 160, 1, 1
        )

    def forward(self, voxel_embeddings, batch, length):

        if self.debug:
            print("Decoder")
            print("     voxel_embeddings.shape", voxel_embeddings.shape)

        if (self.md_infer or self.training or self.debug) and self.use_fsam:
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

            if self.md_res:
                # Multiplication with Residual connection
                x = torch.mul(voxel_embeddings - voxel_embeddings.min() + self.bias1, att_mask - att_mask.min() + self.bias1)
                factorized_embeddings = self.fsam_norm(x)
                factorized_embeddings = voxel_embeddings + factorized_embeddings
            else:
                # Multiplication
                x = torch.mul(voxel_embeddings - voxel_embeddings.min() + self.bias1, att_mask - att_mask.min() + self.bias1)
                factorized_embeddings = self.fsam_norm(x)            

            # # Concatenate
            # factorized_embeddings = torch.cat([voxel_embeddings, self.fsam_norm(x)], dim=1)

            x = self.conv_decoder(factorized_embeddings)
        
        else:
            x = self.conv_decoder(voxel_embeddings)

        rPPG = x.view(-1, length)

        if self.debug:
            print("     rPPG.shape", rPPG.shape)
        
        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            return rPPG, factorized_embeddings, att_mask, appx_error
        else:
            return rPPG



class FactorizePhysMT(nn.Module):
    def __init__(self, frames, md_config, in_channels=3, dropout=0.2, device=torch.device("cpu"), debug=False):
        super(FactorizePhysMT, self).__init__()
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
        self.md_infer = md_config["MD_INFERENCE"]

        nf = md_config["num_filters"]
        if self.debug:
            print("nf:", nf)

        self.encoder = encoder_block(self.in_channels, nf, dropout_rate=dropout, debug=debug)

        self.rppg_head = PhysHead(md_config, device=device, dropout_rate=dropout, debug=debug)
        self.rBr_head = PhysHead(md_config, device=device, dropout_rate=dropout, debug=debug)


        
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

        hidden_embeddings, voxel_embeddings = self.encoder(x)
        # if self.debug:
        #     print("hidden_embeddings.shape", hidden_embeddings.shape)
        #     print("voxel_embeddings.shape", voxel_embeddings.shape)
        
        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            rPPG, factorized_embeddings, att_mask, appx_error = self.rppg_head(voxel_embeddings, batch, length-1)
            rBr, factorized_embeddings_br, att_mask_br, appx_error_br = self.rBr_head(voxel_embeddings, batch, length-1)
        else:
            rPPG = self.rppg_head(voxel_embeddings, batch, length-1)
            rBr = self.rBr_head(voxel_embeddings, batch, length-1)

        # if self.debug:
        #     print("rppg_feats.shape", rppg_feats.shape)

        # rPPG = rppg_feats.view(-1, length-1)

        if self.debug:
            print("rPPG.shape", rPPG.shape)
            print("rBr.shape", rBr.shape)

        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            return rPPG, rBr, voxel_embeddings, factorized_embeddings, att_mask, appx_error, factorized_embeddings_br, att_mask_br, appx_error_br
        else:
            return rPPG, rBr, voxel_embeddings