import os

import nibabel as nib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def interlaced_split(frame: np.ndarray) -> tuple:
    frame = frame.astype(np.uint16)
    odd_frame = np.zeros_like(frame, dtype=np.uint8)
    even_frame = np.zeros_like(frame, dtype=np.uint8)

    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            number = bin(frame[i][j])[2:].zfill(16)
            odd_frame[i][j] = int("0b" + number[0::2], 2)
            even_frame[i][j] = int("0b" + number[1::2], 2)

    return odd_frame, even_frame


def write_yuv(frames: list, yuv_path: str) -> None:
    with open(yuv_path, mode='w') as f:
        for frame in frames:
            np.asarray(frame, dtype=np.uint8).tofile(f)


def convert_ct_to_sequences(nii_folder_: str, rec_folder: str) -> None:
    mse_list = []
    for i, nii_filename in enumerate(os.listdir(nii_folder_)):
        if os.path.splitext(nii_filename)[-1] != ".nii":
            continue

        nii_file = nib.load(os.path.join(nii_folder_, nii_filename))  # axis is [z, x, y]

        origin_data = np.array(nii_file.get_fdata(dtype=np.float64))

        recon_data = np.load(os.path.join(rec_folder, os.path.splitext(nii_filename)[0] + ".npy"))

        mse = np.mean((origin_data - recon_data) ** 2)
        mse_list.append(mse)

    plt.plot(list(range(len(mse_list))), mse_list)
    plt.savefig('./mse.png')


if __name__ == "__main__":
    nii_folder = r"D:\MedicalImageDataset\Mosmed_COVID-19_CT\CT-2"
    rec_folder = r"D:\Rec"
    convert_ct_to_sequences(nii_folder_=nii_folder, rec_folder=rec_folder)
