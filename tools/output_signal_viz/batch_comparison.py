import numpy as np
import torch
import pickle
import scipy
from scipy.signal import filtfilt, butter
from scipy.sparse import spdiags
import argparse
from pathlib import Path
import io
import matplotlib.pyplot as plt

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=device)
        else:
            return super().find_class(module, name)

path_dict_cross_dataset = {
    "test_datasets": {
        "PURE": {
            "root": "runs/exp/PURE_Raw_160_72x72/saved_test_outputs/",
            "exp": {
                "iBVP_EfficientPhys_SASN": "iBVP_EfficientPhys_outputs.pickle",
                "SCAMPS_EfficientPhys_SASN": "SCAMPS_EfficientPhys_PURE_outputs.pickle",
                "UBFC_EfficientPhys_SASN": "UBFC-rPPG_EfficientPhys_outputs.pickle",
                "iBVP_FactorizePhys_FSAM": "iBVP_FactorizePhys_FSAM_Res_PURE_outputs.pickle",
                "SCAMPS_FactorizePhys_FSAM": "SCAMPS_FactorizePhys_FSAM_Res_PURE_outputs.pickle",
                "UBFC_FactorizePhys_FSAM": "UBFC-rPPG_FactorizePhys_FSAM_Res_outputs.pickle"
            }
        },
        "iBVP": {
            "root": "runs/exp/iBVP_RGBT_160_72x72/saved_test_outputs/",
            "exp": {
                "PURE_EfficientPhys_SASN": "PURE_PURE_EfficientPhys_outputs.pickle",
                "PURE_FactorizePhys_FSAM": "PURE_FactorizePhys_FSAM_Res_outputs.pickle",
                "SCAMPS_EfficientPhys_SASN": "SCAMPS_EfficientPhys_outputs.pickle",
                "SCAMPS_FactorizePhys_FSAM": "SCAMPS_FactorizePhys_FSAM_Res_outputs.pickle",
                "UBFC_EfficientPhys_SASN": "UBFC-rPPG_EfficientPhys_iBVP_outputs.pickle",
                "UBFC_FactorizePhys_FSAM": "UBFC-rPPG_FactorizePhys_FSAM_Res_iBVP_outputs.pickle"
            }
        },
        "UBFC-rPPG": {
            "root": "runs/exp/UBFC-rPPG_Raw_160_72x72/saved_test_outputs",
            "exp": {
                "iBVP_EfficientPhys_SASN": "iBVP_EfficientPhys_UBFC-rPPG_outputs.pickle",
                "iBVP_FactorizePhys_FSAM": "iBVP_FactorizePhys_FSAM_Res_UBFC-rPPG_outputs.pickle",
                "PURE_EfficientPhys_SASN": "PURE_EfficientPhys_UBFC-rPPG_outputs.pickle",
                "PURE_FactorizePhys_FSAM": "PURE_FactorizePhys_FSAM_Res_UBFC-rPPG_outputs.pickle",
                "SCAMPS_EfficientPhys_SASN": "SCAMPS_EfficientPhys_UBFC-rPPG_outputs.pickle",
                "SCAMPS_FactorizePhys_FSAM": "SCAMPS_FactorizePhys_FSAM_Res_UBFC-rPPG_outputs.pickle"
            }
        }
    }
}

path_dict_within_dataset = {
    "test_datasets": {
        "PURE": {
            "root": "runs/exp/PURE_Raw_160_72x72/saved_test_outputs/",
            "exp": {
                "EfficientPhys_SASN": "PURE_Intra_EfficientPhys_outputs.pickle",
                "FactorizePhys_FSAM": "PURE_Intra_FactorizePhys_FSAM_Res_outputs.pickle",
                "PhysNet": "PURE_Intra_PhysNet_outputs.pickle",
            }
        },
        "UBFC-rPPG": {
            "root": "runs/exp/UBFC-rPPG_Raw_160_72x72/saved_test_outputs",
            "exp": {
                "EfficientPhys_SASN": "UBFC-rPPG_Intra_EfficientPhys_Epoch9_UBFC-rPPG_outputs.pickle",
                "FactorizePhys_FSAM": "UBFC-rPPG_Intra_FactorizePhys_FSAM_Res_Epoch9_UBFC-rPPG_outputs.pickle",
                "PhysNet": "UBFC-rPPG_Intra_PhysNet_Epoch9_UBFC-rPPG_outputs.pickle",
            }
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
        signal = _detrend(np.cumsum(signal), 100)
    else:
        signal = _detrend(signal, 100)
    if use_bandpass:
        # bandpass filter between [0.75, 2.5] Hz
        # equals [45, 150] beats per min
        [b, a] = butter(1, [0.5 / fs * 2, 2.5 / fs * 2], btype='bandpass')
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


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.75, high_pass=2.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(
        ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr


# Main functions

def compare_estimated_bvps_cross_dataset():

    plot_dir = Path.cwd().joinpath("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    for test_dataset in path_dict_cross_dataset["test_datasets"]:
        print("*"*50)
        print("Test Data:", test_dataset)
        print("*"*50)
        data_dict = {}

        root_dir = Path(path_dict_cross_dataset["test_datasets"][test_dataset]["root"])
        if not root_dir.exists():
            print("Data path does not exists:", str(root_dir))
            exit()

        plot_test_dir = plot_dir.joinpath(test_dataset)
        plot_test_dir.mkdir(parents=True, exist_ok=True)

        for train_model in path_dict_cross_dataset["test_datasets"][test_dataset]["exp"]:
            train_data = train_model.split("_")[0]
            model_name = "_".join(train_model.split("_")[1:])
            print("Train Data, Model:", [train_data, model_name])
            
            if train_data not in data_dict:
                data_dict[train_data] = {}
            
            fn = root_dir.joinpath(path_dict_cross_dataset["test_datasets"][test_dataset]["exp"][train_model])
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

            bvp_label = np.array(_reform_data_from_dict(
                data_dict[train_datasets[0]][model_names[0]]['labels'][trial_list[trial_ind]]))
            bvp_label = _process_signal(bvp_label, fs, diff_flag=diff_flag)

            hr_label = _calculate_fft_hr(bvp_label, fs=fs)
            hr_label = int(np.round(hr_label))
            hr_pred = {}

            for c_ind in range(total_chunks):
                try:
                    fig, ax = plt.subplots(total_train_datasets, 1, figsize=(20, 12), sharex=True)
                    # fig.tight_layout()
                    plt.suptitle('Testing on ' + test_dataset + ' Dataset; Trial: ' +
                                trial_list[trial_ind] + '; Chunk: ' + str(c_ind), fontsize=14)

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

                                hr_pred[model_names[m_ind]] = _calculate_fft_hr(trial_dict[train_datasets[d_ind]][model_names[m_ind]]["prediction"], fs=fs)
                                hr_pred[model_names[m_ind]] = int(np.round(hr_pred[model_names[m_ind]]))

                            ax[d_ind].plot(x_time, trial_dict[train_datasets[d_ind]][model_names[m_ind]]
                                           ["prediction"][start: stop], label=model_names[m_ind] + "; HR = " + str(hr_pred[model_names[m_ind]]))

                        ax[d_ind].plot(x_time, bvp_label[start: stop], label="GT ; HR = " + str(hr_label), color='black')
                        ax[d_ind].legend(loc = "upper right")
                        ax[d_ind].set_title("Training Dataset: " + train_datasets[d_ind])

                    # plt.show()
                    save_fn = plot_test_dir.joinpath(str(trial_list[trial_ind]) + "_" + str(c_ind) + ".jpg")
                    plt.xlabel('Time (s)')
                    plt.savefig(save_fn)
                    plt.close(fig)
                except Exception as e:
                    print("Encoutered error:", e)



def compare_estimated_bvps_within_dataset():

    plot_dir = Path.cwd().joinpath("plots_within_0p5_2p5")
    plot_dir.mkdir(parents=True, exist_ok=True)

    for test_dataset in path_dict_within_dataset["test_datasets"]:
        print("*"*50)
        print("Test Data:", test_dataset)
        print("*"*50)
        data_dict = {}

        root_dir = Path(path_dict_within_dataset["test_datasets"][test_dataset]["root"])
        if not root_dir.exists():
            print("Data path does not exists:", str(root_dir))
            exit()

        plot_test_dir = plot_dir.joinpath(test_dataset)
        plot_test_dir.mkdir(parents=True, exist_ok=True)

        for train_model in path_dict_within_dataset["test_datasets"][test_dataset]["exp"]:
            print("Model:", train_model)

            fn = root_dir.joinpath(path_dict_within_dataset["test_datasets"][test_dataset]["exp"][train_model])
            data_dict[train_model] = CPU_Unpickler(open(fn, "rb")).load()
        
        print("-"*50)

        model_names = list(data_dict.keys())
        print("Model Names:", model_names)

        # print(data_dict[model_names[0]].keys())
        # exit()
        # print(data_dict["iBVP"].keys())
        # print(data_dict["UBFC"].keys())

        # List of all video trials
        trial_list = list(data_dict[model_names[0]]['predictions'].keys())
        print('Num Trials', len(trial_list))

        gt_bvp = np.array(_reform_data_from_dict(
            data_dict[model_names[0]]['predictions'][trial_list[0]]))

        total_samples = len(gt_bvp)
        chunk_size = 160  # size of chunk to visualize: -1 will plot the entire signal
        total_chunks = total_samples // chunk_size
        print('Chunk size', chunk_size)
        print('Total chunks', total_chunks)

        for trial_ind in range(len(trial_list)):
            
            # Read in meta-data from pickle file
            fs = data_dict[model_names[0]]['fs'] # Video Frame Rate
            label_type = data_dict[model_names[0]]['label_type'] # PPG Signal Transformation: `DiffNormalized` or `Standardized`
            diff_flag = (label_type == 'DiffNormalized')

            trial_dict = {}

            bvp_label = np.array(_reform_data_from_dict(
                data_dict[model_names[0]]['labels'][trial_list[trial_ind]]))
            bvp_label = _process_signal(bvp_label, fs, diff_flag=diff_flag)

            hr_label = _calculate_fft_hr(bvp_label, fs=fs)
            hr_label = int(np.round(hr_label))
            hr_pred = {}

            for c_ind in range(total_chunks):
                try:
                    fig = plt.figure(figsize=(15, 5))
                    # fig.tight_layout()

                    start = (c_ind)*chunk_size
                    stop = (c_ind+1)*chunk_size
                    samples = stop - start
                    x_time = np.linspace(0, samples/fs, num=samples)

                    for m_ind in range(len(model_names)):

                        if model_names[m_ind] not in trial_dict:
                            trial_dict[model_names[m_ind]] = {}

                            # Reform label and prediction vectors from multiple trial chunks
                            trial_dict[model_names[m_ind]]["prediction"] = np.array(_reform_data_from_dict(
                                data_dict[model_names[m_ind]]['predictions'][trial_list[trial_ind]]))

                            # Process label and prediction signals
                            trial_dict[model_names[m_ind]]["prediction"] = _process_signal(
                                trial_dict[model_names[m_ind]]["prediction"], fs, diff_flag=diff_flag)

                            hr_pred[model_names[m_ind]] = _calculate_fft_hr(trial_dict[model_names[m_ind]]["prediction"], fs=fs)
                            hr_pred[model_names[m_ind]] = int(np.round([model_names[m_ind]]))
        
                        plt.plot(x_time, trial_dict[model_names[m_ind]]
                                 ["prediction"][start: stop], label=model_names[m_ind] + "; HR = " + str(hr_pred[model_names[m_ind]]))

                    plt.plot(x_time, bvp_label[start: stop], label="GT ; HR = " + str(hr_label), color='black')
                    plt.legend(loc = "upper right")
                    plt.title("Dataset: " + test_dataset + '; Trial: ' +
                                  trial_list[trial_ind] + '; Chunk: ' + str(c_ind), fontsize=14)

                    # plt.show()
                    save_fn = plot_test_dir.joinpath(str(trial_list[trial_ind]) + "_" + str(c_ind) + ".jpg")
                    plt.xlabel('Time (s)')
                    plt.savefig(save_fn)
                    plt.close(fig)
                except Exception as e:
                    print("Encoutered error:", e)



if __name__ == "__main__":
    compare_estimated_bvps_cross_dataset()
    compare_estimated_bvps_within_dataset()