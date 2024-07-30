"""
FactorizePhys: Effective Spatial-Temporal Attention in Remote Photo-plethysmography through Factorization of Voxel Embeddings
"""

import torch
import torch.nn as nn
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
    "visualize": False,
    "ckpt_path": "",
    "data_path": "",
    "label_path": ""
}

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
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

    # net = nn.DataParallel(FactorizePhys(frames=frames, md_config=md_config, device=device, in_channels=in_channels, debug=debug)).to(device)
    net = FactorizePhys(frames=frames, md_config=md_config, device=device, in_channels=in_channels, debug=debug).to(device)
    # net.load_state_dict(torch.load(ckpt_path, map_location=device))
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
        test_data = test_data.detach().numpy()
        vox_embed = vox_embed.detach().numpy()
        if (md_infer or net.training or debug) and use_fsam:
            factorized_embed = factorized_embed.detach().numpy()
            att_mask = att_mask.detach().numpy()

        # print(test_data.shape, vox_embed.shape, factorized_embed.shape)
        b, ch, enc_frames, enc_height, enc_width = vox_embed.shape
        # exit()
        for ch in range(vox_embed.shape[1]):
            if (md_infer or net.training or debug) and use_fsam:
                fig, ax = plt.subplots(9, 4, layout="tight")
            else:
                fig, ax = plt.subplots(9, 2, layout="tight")

            frame = 0
            ax[0, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[0, 0].axis('off')
            ax[0, 1].imshow(vox_embed[0, ch, frame, :, :])
            ax[0, 1].axis('off')
            if (md_infer or net.training or debug) and use_fsam:
                ax[0, 2].imshow(factorized_embed[0, ch, frame, :, :])
                ax[0, 2].axis('off')
                ax[0, 3].imshow(att_mask[0, ch, frame, :, :])
                ax[0, 3].axis('off')

            frame = 20
            ax[1, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[1, 0].axis('off')
            ax[1, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[1, 1].axis('off')
            if (md_infer or net.training or debug) and use_fsam:
                ax[1, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
                ax[1, 2].axis('off')
                ax[1, 3].imshow(att_mask[0, ch, frame//4, :, :])
                ax[1, 3].axis('off')

            frame = 40
            ax[2, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[2, 0].axis('off')
            ax[2, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[2, 1].axis('off')
            if (md_infer or net.training or debug) and use_fsam:
                ax[2, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
                ax[2, 2].axis('off')
                ax[2, 3].imshow(att_mask[0, ch, frame//4, :, :])
                ax[2, 3].axis('off')

            frame = 60
            ax[3, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[3, 0].axis('off')
            ax[3, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[3, 1].axis('off')
            if (md_infer or net.training or debug) and use_fsam:
                ax[3, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
                ax[3, 2].axis('off')
                ax[3, 3].imshow(att_mask[0, ch, frame//4, :, :])
                ax[3, 3].axis('off')

            frame = 80
            ax[4, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[4, 0].axis('off')
            ax[4, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[4, 1].axis('off')
            if (md_infer or net.training or debug) and use_fsam:
                ax[4, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
                ax[4, 2].axis('off')
                ax[4, 3].imshow(att_mask[0, ch, frame//4, :, :])
                ax[4, 3].axis('off')

            frame = 100
            ax[5, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[5, 0].axis('off')
            ax[5, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[5, 1].axis('off')
            if (md_infer or net.training or debug) and use_fsam:
                ax[5, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
                ax[5, 2].axis('off')
                ax[5, 3].imshow(att_mask[0, ch, frame//4, :, :])
                ax[5, 3].axis('off')

            frame = 120
            ax[6, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[6, 0].axis('off')
            ax[6, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[6, 1].axis('off')
            if (md_infer or net.training or debug) and use_fsam:
                ax[6, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
                ax[6, 2].axis('off')
                ax[6, 3].imshow(att_mask[0, ch, frame//4, :, :])
                ax[6, 3].axis('off')

            frame = 140
            ax[7, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[7, 0].axis('off')
            ax[7, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[7, 1].axis('off')
            if (md_infer or net.training or debug) and use_fsam:
                ax[7, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
                ax[7, 2].axis('off')
                ax[7, 3].imshow(att_mask[0, ch, frame//4, :, :])
                ax[7, 3].axis('off')

            frame = 159
            ax[8, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[8, 0].axis('off')
            ax[8, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[8, 1].axis('off')
            if (md_infer or net.training or debug) and use_fsam:
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