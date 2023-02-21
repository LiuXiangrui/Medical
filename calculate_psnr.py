import os
import math

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def calculate_psnr(nii_folder_: str, rec_folder_: str) -> None:
    mse_list = []
    psnr_list = []
    for i, nii_filename in enumerate(os.listdir(nii_folder_)):
        if os.path.splitext(nii_filename)[-1] != ".nii":
            continue

        nii_file = nib.load(os.path.join(nii_folder_, nii_filename))  # axis is [z, x, y]

        origin_data = np.array(nii_file.get_fdata(dtype=np.float64))

        recon_data = np.load(os.path.join(rec_folder_, os.path.splitext(nii_filename)[0] + ".npy"))

        mse = np.mean((origin_data - recon_data) ** 2)
        mse_list.append(mse)

        psnr = 10 * math.log10(65536 / mse)
        psnr_list.append(psnr)

    print("average psnr = ", sum(psnr_list)/len(psnr_list))

    plt.plot(list(range(len(mse_list))), mse_list)
    plt.savefig('./mse.png')


if __name__ == "__main__":
    nii_folder = r"D:\MedicalImageDataset\Mosmed_COVID-19_CT\CT-2"
    rec_folder = r"D:\Rec"
    calculate_psnr(nii_folder_=nii_folder, rec_folder_=rec_folder)
