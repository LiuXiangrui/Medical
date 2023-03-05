import math
import os

import nibabel as nib
import numpy as np


def calculate_psnr(nii_folder_: str, rec_folder_: str):
    mse_list = []
    psnr_list = []
    for i, nii_filename in enumerate(os.listdir(nii_folder_)):
        if os.path.splitext(nii_filename)[-1] != ".nii":
            continue

        nii_file = nib.load(os.path.join(nii_folder_, nii_filename))  # axis is [z, x, y]

        origin_data = np.array(nii_file.get_fdata(dtype=np.float64))

        rec_nii_file = os.path.join(rec_folder_, os.path.splitext(nii_filename)[0] + ".npy")
        recon_data = np.load(rec_nii_file)

        mse = np.mean((origin_data - recon_data) ** 2)
        mse_list.append(mse)

        psnr = 10 * math.log10(65535 * 65535 / mse)

        psnr_list.append(psnr)
    average_psnr = sum(psnr_list)/len(psnr_list)

    return average_psnr


if __name__ == "__main__":
    # qp_list = [22, 26, 30, 34]
    qp_list = [16, 20, 25, 29]
    # qp_list = [-16, -8, 1, 63]

    for qp in qp_list:
        nii_folder = r"D:\MedicalImageDataset\Mosmed_COVID-19_CT\CT-3"
        # nii_folder = r"D:\MedicalImageDataset\TRABIT2019_MRI\test"
        # rec_folder = r"C:\Users\xiangrliu3\Desktop\10bitExperiments\MRIRecNIITest10Bit"
        # rec_folder = r"D:\AVS3_MRI_Anchor_Results\MRIRecNIITestInterlaced"
        rec_folder = r"D:\AVS3_CT_Anchor_Results\CTRecNIITestInterlaced"

        rec_folder = os.path.join(rec_folder, str(qp))
        avg_psnr = calculate_psnr(nii_folder_=nii_folder, rec_folder_=rec_folder)
        print("QP {}: PSNR = {}".format(qp, avg_psnr))