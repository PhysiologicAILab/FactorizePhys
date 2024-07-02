import numpy as np
from scipy import fft as sp_fft
from scipy.signal import butter, filtfilt, lfilter, find_peaks
from scipy.signal import windows as sp_windows
import statsmodels.api as sm
import os
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt

# Filter requirements.
T = 60        # Sample Period
fs = 30.0       # sample rate, Hz
cutoff = 0.1      # desired cutoff frequency of the filter, Hz ,
order = 4

nyq = 0.5 * fs


def butter_highpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order/2, normal_cutoff, btype='highpass', analog=False)
    y = filtfilt(b, a, data)
    # b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    # y = lfilter(b, a, data)
    return y


def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order/2, normal_cutoff, btype='lowpass', analog=False)
    y = filtfilt(b, a, data)
    # b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    # y = lfilter(b, a, data)
    return y


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def filt_by_paper(data):
    # plt.figure()
    # plt.plot(data)
    # data_highpass = butter_highpass_filter(data, 0.3, 30, 4)
    data_lowpass = butter_lowpass_filter(data, 1, 30, 4)
    return butter_highpass_filter(data_lowpass, 0.3, 30, 4)
    # return butter_lowpass_filter(data_highpass,1,30,4)


def peak_amp(freqs, amps, fl, fh):
    peak = 0
    peakf = 0
    for i, f in enumerate(freqs):
        if f >= fl and f <= fh:
            if np.mean(amps[i]) > peak:
                peak = np.mean(amps[i])
                peakf = f
    return peak, peakf


def peak_amps(freqs, amps, n):
    ind = np.argpartition(amps, -n)[-n:]
    sorted_ind = ind[np.argsort(amps[ind])]
    return zip(freqs[sorted_ind][::-1], amps[sorted_ind][::-1])


def mean_fft_2d(data, fps):
    L = np.array(data, dtype="float64")

    L *= sp_windows.hann(len(L))

    L -= np.mean(L)

    fft = sp_fft.rfft(L)

    N = len(data)

    xf = sp_fft.rfftfreq(N, 1 / fps)

    return xf, np.abs(fft)


def get_eRR_zc(data):
    zc = ((data[:-1] * data[1:]) < 0).sum()
    fzc = (0.5*(zc-1)) * (30.0/len(data))
    eRR_zc = fzc*(len(data)/30.0)
    return eRR_zc


def get_eRR_pk(data, height=0, prominence=0):
    peaks, desc = find_peaks(data, height=height, prominence=prominence)
    # peaks, desc = find_peaks(data, prominence=0.1)
    eRR_pk = len(peaks)
    return eRR_pk

def get_eRR_sp(data_orig):
    data = data_orig.copy()
    xf, fft = mean_fft_2d(data, 30)

    candidates = peak_amps(xf, fft, 7)

    return list(candidates)[0][0] * (len(data)/30.0)


def get_eRR_ap(data, height=None, prominence=None):
    ac = sm.tsa.acf(data, nlags=len(data))
    peaks, desc = find_peaks(ac, height=height, prominence=prominence)
    avg_period = np.median(peaks[1:] - peaks[:-1])
    # peaks, desc = find_peaks(ac, height=0)
    # peaks, desc = find_peaks(ac, prominence=0.1)
    if len(data) > 0:
        return len(data) / avg_period
    else:
        return 0


def get_eRR_af(data):
    ac = sm.tsa.acf(data, nlags=len(data))
    xf, fft = mean_fft_2d(ac, 30)

    candidates = peak_amps(xf, fft, 7)

    return list(candidates)[0][0] * (len(data)/30.0)


# def get_thermal_voxel(num_frames, roi_matrix):
#     vox_vec = np.zeros((num_frames))
#     mean_frame_buffer = np.zeros(2) #for computing mean value from frames RoI

#     for i in range(num_frames):
#         frame_data = roi_matrix[i, :, :]

#         mean_frame_val = np.mean(frame_data)
#         mean_frame_buffer = np.roll(mean_frame_buffer, -1)
#         mean_frame_buffer[-1] = mean_frame_val

#         if np.sum(mean_frame_buffer != 0) > 0:
#             vox_vec[i] = np.sum(frame_data > np.mean(mean_frame_buffer[mean_frame_buffer > 0]))
#         else:
#             vox_vec[i] = np.sum(frame_data > np.mean(mean_frame_buffer))

#     return vox_vec

def get_thermal_voxel(num_frames, roi_matrix):
    mean_vec_iqr = np.zeros((num_frames))
    iqr_diff_vec = np.zeros((num_frames))
    vox_vec = np.zeros((num_frames))
    vox_vec_iqr = np.zeros((num_frames))
    
    mean_frame_buffer = np.zeros(5)  # for computing mean value from frames RoI
    mean_frame_buffer_vox_iqr = np.zeros(5)  # for computing mean value from frames RoI
    
    prev_iqr_diff = 0
    
    for i in range(num_frames):
        frame_data = roi_matrix[i, :, :]

        qa, q1, q3, qc, qd = np.percentile(frame_data, [10, 25, 75, 95, 99])
        
        new_frame_iqr = frame_data[(frame_data > q1) & ((frame_data < q3))]
        new_frame_vox_iqr = frame_data[(frame_data > qa) & ((frame_data < qc))]
        
        mean_frame_val = np.mean(frame_data)
        mean_frame_val_iqr = np.mean(new_frame_iqr)
        mean_frame_val_vox_iqr = np.mean(new_frame_vox_iqr)

        iqr_diff = (np.mean(frame_data[(frame_data > q3) & (frame_data < qd)]) -
                    np.mean(frame_data[(frame_data > qa) & (frame_data < q1)])) or prev_iqr_diff
        iqr_diff_vec[i] = iqr_diff
        mean_vec_iqr[i] = mean_frame_val_iqr
        prev_iqr_diff = iqr_diff
        
        # print("q1, q2, q3", q1, q2, q3, iqr_diff, np.sum(frame_data < q1), np.sum(frame_data > q3), frame_data.size)

        mean_frame_buffer = np.roll(mean_frame_buffer, -1)
        mean_frame_buffer[-1] = mean_frame_val
        if np.sum(mean_frame_buffer != 0) > 0:
            vox_vec[i] = np.sum(frame_data > np.mean(mean_frame_buffer[mean_frame_buffer > 0]))
        else:
            vox_vec[i] = np.sum(frame_data > np.mean(mean_frame_buffer))

        mean_frame_buffer_vox_iqr = np.roll(mean_frame_buffer_vox_iqr, -1)
        mean_frame_buffer_vox_iqr[-1] = mean_frame_val_vox_iqr
        if np.sum(mean_frame_buffer_vox_iqr != 0) > 0:
            vox_vec_iqr[i] = np.sum(new_frame_vox_iqr > np.mean(mean_frame_buffer_vox_iqr[mean_frame_buffer_vox_iqr > 0]))
        else:
            vox_vec_iqr[i] = np.sum(new_frame_vox_iqr > np.mean(mean_frame_buffer_vox_iqr))

    # iqr_diff_vec -= np.mean(iqr_diff_vec)
    return mean_vec_iqr, iqr_diff_vec, vox_vec, vox_vec_iqr


# def get_thermal_voxel(num_frames, roi_matrix):
#     vox_vec = np.zeros((num_frames))
#     vox_vec_masked = np.zeros((num_frames))
#     mean_frame_buffer = np.zeros(2) # can work with much larger buffer in stationary conditions
#     mean_frame_buffer_masked = np.zeros(2)

#     for i in range(num_frames):
#         frame_data = roi_matrix[i, :, :]

#         mean_frame_val = np.mean(frame_data)
#         mean_frame_buffer = np.roll(mean_frame_buffer, -1)
#         mean_frame_buffer[-1] = mean_frame_val

#         if np.sum(mean_frame_buffer != 0) > 0:
#             vox_vec[i] = np.sum(frame_data > np.mean(
#                 mean_frame_buffer[mean_frame_buffer > 0]))
#         else:
#             vox_vec[i] = np.sum(frame_data > np.mean(mean_frame_buffer))

#         frame_data_masked = np.zeros_like(frame_data)
#         height, width = frame_data.shape
#         # Ellipse parameters
#         radius = min(int(width/2), height)
#         center = (int(width / 2), height)
#         axes = (radius, radius)
#         angle = 0
#         startAngle = 180
#         endAngle = 360
#         thickness = -1
#         color = 1

#         frame_data_masked = cv2.ellipse(
#             frame_data_masked, center, axes, angle, startAngle, endAngle, color, thickness)

#         frame_data_masked = frame_data * frame_data_masked
#         # frame_data_masked[frame_data_masked == 0] = np.mean(frame_data_masked[frame_data_masked > 0])

#         mean_frame_val_masked = np.mean(frame_data_masked[frame_data_masked > 0])
#         mean_frame_buffer_masked = np.roll(mean_frame_buffer_masked, -1)
#         mean_frame_buffer_masked[-1] = mean_frame_val_masked

#         if np.sum(mean_frame_buffer_masked != 0) > 0:
#             vox_vec_masked[i] = np.sum(frame_data_masked > np.mean(
#                 mean_frame_buffer_masked[mean_frame_buffer_masked > 0]))
#         else:
#             vox_vec_masked[i] = np.sum(
#                 frame_data_masked > np.mean(mean_frame_buffer_masked))

#     return vox_vec, vox_vec_masked


def get_time_normalized_skew_and_mean(num_frames, roi_matrix, kernel, window=300):  #window = 10 sec * 30 FPS = 300
    mean_vec = np.zeros((num_frames))
    skew_vec = np.zeros((num_frames))
    mean_diff_vec = np.zeros((num_frames))

    norm_mean_vec = np.zeros((num_frames))
    norm_skew_vec = np.zeros((num_frames))
    norm_mean_diff_vec = np.zeros((num_frames))

    vox_vec = np.zeros((num_frames))
    # vox_vec = np.zeros((num_frames))

    norm_roi_matrix = deepcopy(roi_matrix)
    # temporal normalization
    segs = np.arange(0, num_frames, window)
    for win_seg in segs:
        mean_seg = np.mean(norm_roi_matrix[win_seg : win_seg + window, :, :])
        std_seg = np.std(norm_roi_matrix[win_seg : win_seg + window, :, :])
        norm_roi_matrix[win_seg : win_seg + window, :, :] = (norm_roi_matrix[win_seg : win_seg + window, :, :] - mean_seg)/std_seg

    mean_frame_buffer = np.zeros(2) #can work with much larger buffer in stationary conditions

    for i in range(num_frames):
        # Apply spatial filtering for removing camera noise
        frame_data = roi_matrix[i, :, :]
        norm_frame_data = norm_roi_matrix[i, :, :]
        # frame_data = cv2.filter2D(roi_matrix[i, :, :], -1, kernel)
        # norm_frame_data = cv2.filter2D(norm_roi_matrix[i, :, :], -1, kernel)

        mean_frame_val = np.mean(frame_data)
        mean_norm_frame_val = np.mean(norm_frame_data)
        std_frame_val = np.std(frame_data)
        std_norm_frame_val = np.std(norm_frame_data)

        mean_frame_buffer = np.roll(mean_frame_buffer, -1)
        mean_frame_buffer[-1] = mean_frame_val

        if np.sum(mean_frame_buffer != 0) > 0:
            vox_vec[i] = np.sum(frame_data > np.mean(mean_frame_buffer[mean_frame_buffer > 0]))
        else:
            vox_vec[i] = np.sum(frame_data > mean_frame_val)

        nroi = frame_data.size
        if (nroi <= 2):
            skew_vec[i] = skew_vec[i-1]
            norm_skew_vec[i] = norm_skew_vec[i-1]
        else:
            skew_vec[i] = np.sum(
                np.power((frame_data - mean_frame_val)/std_frame_val, 3)) \
                * (nroi/((nroi-1)*(nroi-2)))
            norm_skew_vec[i] = np.sum(
                np.power((norm_frame_data - mean_norm_frame_val)/std_norm_frame_val, 3)) \
                * (nroi/((nroi-1)*(nroi-2)))

        mean_vec[i] = mean_frame_val
        norm_mean_vec[i] = mean_norm_frame_val

        if i == 0:
            mean_diff_vec[i] = 0
            norm_mean_diff_vec[i] = 0
        else:
            diff_frame = frame_data - prev_frame_data
            norm_diff_frame = norm_frame_data - prev_norm_frame_data

            nroi = frame_data.size
            if (nroi <= 2):
                mean_diff_vec[i] = mean_diff_vec[i-1]
                norm_mean_diff_vec[i] = norm_mean_diff_vec[i-1]
            else:
                mean_diff_vec[i] = np.sum(
                    np.power((diff_frame - np.mean(diff_frame))/np.std(diff_frame), 3)) \
                    * (nroi/((nroi-1)*(nroi-2)))
                norm_mean_diff_vec[i] = np.sum(
                    np.power((norm_diff_frame - np.mean(norm_diff_frame))/np.std(norm_diff_frame), 3)) \
                    * (nroi/((nroi-1)*(nroi-2)))

            # mean_diff_vec[i] = np.mean(diff_frame)
            # norm_mean_diff_vec[i] = np.mean(norm_diff_frame)

        prev_frame_data = frame_data
        prev_norm_frame_data = norm_frame_data

    mean_vec -= np.mean(mean_vec)
     
    return mean_vec, skew_vec, mean_diff_vec, norm_mean_vec, norm_skew_vec, norm_mean_diff_vec, vox_vec


def get_skew_and_mean(num_frames, roi_matrix, bgr=0):

    ft_vector = np.zeros((num_frames))
    skew_vector = np.zeros((num_frames))

    for i in range(num_frames):
        frame_data = roi_matrix[i, :, :]
        frame_data_bg_sub = frame_data[frame_data > bgr]
        # frame_data_bg_sub = frame_data
        nroi = frame_data_bg_sub.size
        if (nroi <= 2):
            skew_vector[i] = skew_vector[i-1]
        else:
            skew_vector[i] = np.sum(
                np.power((frame_data_bg_sub - np.mean(frame_data_bg_sub))/np.std(frame_data_bg_sub), 3)) \
                        * (nroi/((nroi-1)*(nroi-2)))

        ft_vector[i] = np.mean(frame_data[frame_data > bgr])
        if (np.isnan(ft_vector[i])):
            ft_vector[i] = ft_vector[i-1]
    ft_vector -= np.mean(ft_vector)
    return skew_vector, ft_vector


def explore_roi(path_to_file, bgr=0):
    roi = []

    if os.path.exists(path_to_file):
        with open(path_to_file, 'rb') as f:
            num_frames = np.load(f)
            ft_vector = np.zeros((num_frames))
            skew_vector = np.zeros((num_frames))
            thermal_timestamps = np.load(f)
            re_init = np.load(f)
            ROI_seq = np.load(f)
            for i in range(num_frames):
                frame_data = np.load(f)

                roi.append(frame_data)

                frame_data_bg_sub = frame_data[frame_data > bgr]

                # roi.append(frame_data_bg_sub)
        return roi

    else:
        print(f"{path_to_file} not found!")
        return False, 0, 0
