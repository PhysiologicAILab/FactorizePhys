import numpy as np
import cv2
from pathlib import Path
import argparse


def batch_resize(opt):
    target_size = opt.target_size
    newH, newW = target_size
    print("Target size:", target_size)

    datadir = Path(opt.datadir)
    if datadir.exists():
        print("Root data path:", str(datadir))
        files_list = sorted(list(datadir.glob("*input*.npy")))
    else:
        print(str(datadir) + " - does not exist")
        exit()

    for fp in files_list:
        print("Processing:", fp.name)
        frames = np.load(str(fp))
        T, H, W, C = frames.shape
        new_frames = np.zeros((T, newH, newW, C))
        for idx in range(T):
            new_frames[idx, ...] = cv2.resize(frames[idx, ...], target_size)
        np.save(str(fp), new_frames)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, dest="datadir", help='root path with npy files')
    parser.add_argument('--size', nargs="+", type=int, dest="target_size", help='target resize dimension in [H,W]')
    opt = parser.parse_args()
    
    batch_resize(opt)
    