import numpy as np
import torch
import pickle
from scipy.signal import filtfilt, butter
from scipy.sparse import spdiags
import argparse
from pathlib import Path
import io
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=device)
        else:
            return super().find_class(module, name)


# path_dict = {
#     "root": "/Users/jiteshjoshi/Downloads/rPPG_Testing/FactorizePhys/",
#     "test_datasets": {
#         "PURE": {
#             "iBVP_EfficientPhys_SASN": "iBVP_iBVP_EfficientPhys_outputs.pickle",
#             "iBVP_EfficientPhys_FSAM": "iBVP_iBVP_EfficientPhys_FSAM_V12_FD40_outputs.pickle",
#             "iBVP_FactorizePhys": "iBVP_iBVP_FactorizePhys_FactorizePhys_v63_No_FSAM_outputs.pickle",
#             "iBVP_FactorizePhys_FSAM": "iBVP_iBVP_FactorizePhys_FactorizePhys_v63_R8_S8_ST6_LR1e-3_NoRes_outputs.pickle",
#             "iBVP_PhysFormer": "iBVP_iBVP_PHYSFORMER_outputs.pickle",
#             "iBVP_PhysNet": "iBVP_iBVP_PHYSNET_outputs.pickle",
#             "UBFC_EfficientPhys_FSAM": "UBFC_UBFC_PURE_EfficientPhys_FM_v12_outputs.pickle",
#             "UBFC_EfficientPhys_SASN": "UBFC_UBFC_PURE_EfficientPhys_outputs.pickle",
#             "UBFC_FactorizePhys": "UBFC_UBFC_PURE_FactorizePhys_v63_No_FSAM_outputs.pickle",
#             "UBFC_FactorizePhys_FSAM": "UBFC_UBFC_PURE_FactorizePhys_v63_R8_S8_ST6_LR1e-3_NoRes_outputs.pickle",
#             "UBFC_PhysNet": "UBFC_UBFC_PURE_PhysNet_outputs.pickle",
#             "UBFC_PhysFormer": "UBFC-rPPG_UBFC-rPPG_PURE_PhysFormer_outputs.pickle"
#         }
#     }
# }

path_dict = {
    "root": "/Users/jiteshjoshi/Downloads/rPPG_Testing/FactorizePhys/",
    "test_datasets": {
        "PURE": {
            "iBVP_EfficientPhys_SASN": "iBVP_iBVP_EfficientPhys_outputs.pickle",
            "iBVP_FactorizePhys_FSAM": "iBVP_iBVP_FactorizePhys_FactorizePhys_v63_R8_S8_ST6_LR1e-3_NoRes_outputs.pickle",
            "UBFC_EfficientPhys_SASN": "UBFC_UBFC_PURE_EfficientPhys_outputs.pickle",
            "UBFC_FactorizePhys_FSAM": "UBFC_UBFC_PURE_FactorizePhys_v63_R8_S8_ST6_LR1e-3_NoRes_outputs.pickle",
        }
    }
}



# HELPER FUNCTIONS

def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)

    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())

    return sort_data

def _process_signal(signal, fs=30, diff_flag=True):
    # Detrend and filter
    use_bandpass = True
    if diff_flag:  # if the predictions and labels are 1st derivative of PPG signal.
        gt_bvp = _detrend(np.cumsum(signal), 100)
    else:
        gt_bvp = _detrend(signal, 100)
    if use_bandpass:
        # bandpass filter between [0.75, 2.5] Hz
        # equals [45, 150] beats per min
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        signal = filtfilt(b, a, np.double(signal))
    return signal

def _detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal



def compare_estimated_bvps():

    root_dir = Path(path_dict["root"])
    if not root_dir.exists():
        print("Data path does not exists:", str(root_dir))
        exit()
    
    plot_dir = root_dir.joinpath("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    for test_dataset in path_dict["test_datasets"]:
        print("*"*50)
        print("Test Data:", test_dataset)
        print("*"*50)
        data_dict = {}

        plot_test_dir = plot_dir.joinpath(test_dataset)
        plot_test_dir.mkdir(parents=True, exist_ok=True)

        for train_model in path_dict["test_datasets"][test_dataset]:
            train_data = train_model.split("_")[0]
            model_name = "_".join(train_model.split("_")[1:])
            print("Train Data, Model:", [train_data, model_name])
            
            if train_data not in data_dict:
                data_dict[train_data] = {}
            
            fn = root_dir.joinpath(path_dict["test_datasets"][test_dataset][train_model])
            data_dict[train_data][model_name] = CPU_Unpickler(open(fn, "rb")).load()
        
        print("-"*50)
    
        # print(data_dict.keys())
        # print(data_dict["iBVP"].keys())
        # print(data_dict["UBFC"].keys())

        total_train_datasets = len(data_dict)
        train_datasets = list(data_dict.keys())
        model_names = list(data_dict[train_datasets[0]].keys())
        print("Total training datasets:", total_train_datasets)
        print("Training datasets:", train_datasets)
        print("Model Names:", model_names)

        # List of all video trials
        trial_list = list(data_dict[train_datasets[0]][model_names[0]]['predictions'].keys())
        print('Num Trials', len(trial_list))

        gt_bvp = np.array(_reform_data_from_dict(
            data_dict[train_datasets[0]][model_names[0]]['predictions'][trial_list[0]]))

        total_samples = len(gt_bvp)
        chunk_size = 160  # size of chunk to visualize: -1 will plot the entire signal
        total_chunks = total_samples // chunk_size
        print('Chunk size', chunk_size)
        print('Total chunks', total_chunks)


        for trial_ind in range(len(trial_list)):
            
            # Read in meta-data from pickle file
            fs = data_dict[train_datasets[0]][model_names[0]]['fs'] # Video Frame Rate
            label_type = data_dict[train_datasets[0]][model_names[0]]['label_type'] # PPG Signal Transformation: `DiffNormalized` or `Standardized`
            diff_flag = (label_type == 'DiffNormalized')

            trial_dict = {}

            gt_bvp = np.array(_reform_data_from_dict(
                data_dict[train_datasets[0]][model_names[0]]['labels'][trial_list[trial_ind]]))
            gt_bvp = _process_signal(gt_bvp, fs, diff_flag=diff_flag)

            for c_ind in range(total_chunks):

                fig, ax = plt.subplots(total_train_datasets, 1, figsize=(25, 9))
                plt.suptitle('Testing on PURE Dataset; Trial: ' + trial_list[trial_ind] + "; Chunk: " + str(c_ind))

                start = (c_ind)*chunk_size
                stop = (c_ind+1)*chunk_size
                samples = stop - start
                x_time = np.linspace(0, samples/fs, num=samples)

                for d_ind in range(total_train_datasets):
                    if train_datasets[d_ind] not in trial_dict:
                        trial_dict[train_datasets[d_ind]] = {}

                    for m_ind in range(len(model_names)):
                            
                        if model_names[m_ind] not in trial_dict[train_datasets[d_ind]]:
                            trial_dict[train_datasets[d_ind]][model_names[m_ind]] = {}

                            # Reform label and prediction vectors from multiple trial chunks
                            trial_dict[train_datasets[d_ind]][model_names[m_ind]]["prediction"] = np.array(_reform_data_from_dict(
                                data_dict[train_datasets[d_ind]][model_names[m_ind]]['predictions'][trial_list[trial_ind]]))

                            # Process label and prediction signals
                            trial_dict[train_datasets[d_ind]][model_names[m_ind]]["prediction"] = _process_signal(
                                trial_dict[train_datasets[d_ind]][model_names[m_ind]]["prediction"], fs, diff_flag=diff_flag)

                        ax[d_ind].plot(x_time, trial_dict[train_datasets[d_ind]][model_names[m_ind]]
                                       ["prediction"][start: stop], label=model_names[m_ind])

                    ax[d_ind].plot(x_time, gt_bvp[start: stop], label="GT", color='black')
                    ax[d_ind].legend(loc = "upper right")
                    ax[d_ind].set_xlabel('Time (s)')
                    ax[d_ind].set_title("Training Dataset: " + train_datasets[d_ind])

                # plt.show()
                save_fn = plot_test_dir.joinpath(str(trial_list[trial_ind]) + "_" + str(c_ind) + ".jpg")
                plt.savefig(save_fn)
                plt.close(fig)
        
if __name__ == "__main__":
    compare_estimated_bvps()