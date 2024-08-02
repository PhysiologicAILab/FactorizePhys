"""
FactorizePhys: Effective Spatial-Temporal Attention in Remote Photo-plethysmography through Factorization of Voxel Embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from neural_methods.model.FactorizePhys.FactorizePhys import FactorizePhys

model_config = {
    "MD_FSAM": True,
    "MD_TYPE": "NMF",
    "MD_TRANSFORM": "T_KAB",
    "MD_R": 1,
    "MD_S": 1,
    "MD_STEPS": 4,
    "MD_INFERENCE": True,
    "MD_RESIDUAL": True,
    "in_channels": 3,
    "data_channels": 4,
    "height": 72,
    "weight": 72,
    "batch_size": 2,
    "frames": 160,
    "debug": True,
    "assess_latency": False,
    "num_trials": 20,
    "visualize": True,
    "ckpt_path": "./final_model_release/iBVP_FactorizePhys_FSAM_Res.pth",
    "data_path": "/home/jitesh/data/iBVP_Dataset/iBVP_RGB_160_72x72/p07a_input1.npy",
    "label_path": "/home/jitesh/data/iBVP_Dataset/iBVP_RGB_160_72x72/p07a_label1.npy"
}

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    from scipy.signal import resample
    # from torch.utils.tensorboard import SummaryWriter

    # default `log_dir` is "runs" - we'll be more specific here
    # writer = SummaryWriter('runs/FactorizePhys')

    ckpt_path = model_config["ckpt_path"]
    data_path = model_config["data_path"]

    label_path = model_config["label_path"]

    use_fsam = model_config["MD_FSAM"]
    md_infer = model_config["MD_INFERENCE"]

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

    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device("cpu")

    if visualize:
        np_data = np.load(data_path)
        np_label = np.load(label_path)
        np_label = np.expand_dims(np_label, 0)
        np_label = torch.tensor(np_label)

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
    md_config["FRAME_NUM"] = frames
    md_config["MD_S"] = model_config["MD_S"]
    md_config["MD_R"] = model_config["MD_R"]
    md_config["MD_STEPS"] = model_config["MD_STEPS"]
    md_config["MD_FSAM"] = model_config["MD_FSAM"]
    md_config["MD_TYPE"] = model_config["MD_TYPE"]
    md_config["MD_TRANSFORM"] = model_config["MD_TRANSFORM"]
    md_config["MD_INFERENCE"] = model_config["MD_INFERENCE"]
    md_config["MD_RESIDUAL"] = model_config["MD_RESIDUAL"]

    if visualize:
        net = nn.DataParallel(FactorizePhys(frames=frames, md_config=md_config, device=device, in_channels=in_channels, debug=debug)).to(device)
        net.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        net = FactorizePhys(frames=frames, md_config=md_config, device=device, in_channels=in_channels, debug=debug).to(device)
    net.eval()

    if assess_latency:
        time_vec = []
        if debug:
            appx_error_list = []
        for passes in range(num_trials):
            t0 = time.time()
            if (md_infer or net.training or debug) and use_fsam:
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
        if (md_infer or net.training or debug) and use_fsam:
            pred, vox_embed, factorized_embed, att_mask, appx_error = net(test_data)
            print("Appx error: ", appx_error.item())  # .detach().numpy())
        else:
            pred, vox_embed = net(test_data)

    # print("-"*100)
    # print(net)
    # print("-"*100)

    if visualize:
        
        # b, channels, enc_frames, enc_height, enc_width = vox_embed.shape
        # label_matrix = np_label.unsqueeze(0).repeat(1, channels, 1).unsqueeze(
        #     2).unsqueeze(2).permute(0, 1, 4, 3, 2).repeat(1, 1, 1, enc_height, enc_width)
        # label_matrix = label_matrix.to(device=device)

        # corr_matrix = F.cosine_similarity(vox_embed, label_matrix, dim=2)

        avg_emb = torch.mean(vox_embed, dim=1)
        b, enc_frames, enc_height, enc_width = avg_emb.shape

        label_matrix = np_label.unsqueeze(0).unsqueeze(2).permute(0, 3, 2, 1).repeat(1, 1, enc_height, enc_width)
        label_matrix = label_matrix.to(device=device)
        corr_matrix = F.cosine_similarity(avg_emb, label_matrix, dim=1)

        print("corr_matrix.shape", corr_matrix.shape)
        # exit()

        test_data = test_data.detach().cpu().numpy()
        vox_embed = vox_embed.detach().cpu().numpy()
        corr_matrix = corr_matrix.detach().cpu().numpy()
        
        if (md_infer or net.training or debug) and use_fsam:
            att_mask = torch.mean(att_mask, dim=1)
            factorized_embed = factorized_embed.detach().cpu().numpy()
            # att_mask = att_mask.detach().cpu().numpy()

            b, enc_frames, att_height, att_width = att_mask.shape

            label_matrix_att = np_label.unsqueeze(0).unsqueeze(2).permute(0, 3, 2, 1).repeat(1, 1, att_height, att_width)
            label_matrix_att = label_matrix_att.to(device=device)
            corr_matrix_att = F.cosine_similarity(att_mask, label_matrix_att, dim=1)

            corr_matrix_att = corr_matrix_att.detach().cpu().numpy()
            att_mask = att_mask.detach().cpu().numpy()

        print("corr_matrix_att.shape", corr_matrix_att.shape)

        print("test_data.shape:", test_data.shape)
        print("vox_embed.shape:", vox_embed.shape)
        print("factorized_embed.shape:", factorized_embed.shape)
        
        # exit()
        # for ch in range(vox_embed.shape[1]):

        # if (md_infer or net.training or debug) and use_fsam:
        #     fig, ax = plt.subplots(3, 3, layout="tight")
        # else:

        fig, ax = plt.subplots(1, 3, layout="tight")

        ch = 0
        ax[0].imshow(np_data[ch, ...].astype(np.uint8))
        ax[0].axis('off')
        ax[1].imshow(corr_matrix[ch, :, :], cmap='nipy_spectral', vmin=-1, vmax=1)
        ax[1].axis('off')
        if (md_infer or net.training or debug) and use_fsam:
            ax[2].imshow(corr_matrix_att[ch, :, :], cmap='nipy_spectral', vmin=-1, vmax=1)
            ax[2].axis('off')

        # ch = 5
        # ax[1, 0].imshow(np_data[ch, ...].astype(np.uint8))
        # ax[1, 0].axis('off')
        # ax[1, 1].imshow(corr_matrix[0, ch, :, :])
        # ax[1, 1].axis('off')
        # # if (md_infer or net.training or debug) and use_fsam:
        # #     ax[1, 2].imshow(factorized_embed[0, ch, :, :])
        # #     ax[1, 2].axis('off')

        # ch = 12
        # ax[2, 0].imshow(np_data[ch, ...].astype(np.uint8))
        # ax[2, 0].axis('off')
        # ax[2, 1].imshow(corr_matrix[0, ch, :, :])
        # ax[2, 1].axis('off')
        # # if (md_infer or net.training or debug) and use_fsam:
        # #     ax[2, 2].imshow(factorized_embed[0, ch, :, :])
        # #     ax[2, 2].axis('off')

        fig.colorbar(mappable=cm.ScalarMappable(cmap='nipy_spectral'), ax=ax[0])

        # plt.show()
        plt.savefig("AttentionMap.jpg")
        plt.close(fig)
    print("pred.shape", pred.shape)

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print("Total parameters = ", pytorch_total_params)

    pytorch_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Trainable parameters = ", pytorch_trainable_params)

    # writer.add_graph(net, test_data)
    # writer.close()