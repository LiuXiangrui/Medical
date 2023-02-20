import os

import nibabel as nib
import numpy as np
from PIL import Image
import cv2

sequences_folder = r"C:\Users\XiangruiLiu\Desktop\CT_16bit"
nii_folder = r"\\XRLIU-NAS\Datasets\Mosmed_COVID-19_CT_Scans\CT-2"


def convert_ct_to_sequences(nii_filepath: str, min_: float = -2048., max_: float = 2048., bit_depth: int = 16):
    os.makedirs(sequences_folder, exist_ok=True)
    frames_folder = os.path.join(sequences_folder, os.path.split(os.path.splitext(nii_filepath)[0])[-1])
    os.makedirs(frames_folder, exist_ok=True)

    nii_file = nib.load(nii_filepath)  # axis is [z, x, y]
    affine_matrix = nii_file.affine

    data = np.array(nii_file.get_fdata(dtype=np.float64)).clip(min=min_, max=max_)
    data = (data - min_) / (max_ - min_) * (2 ** bit_depth)
    data = data.astype(np.uint16)

    frames = [data[:, :, i] for i in range(data.shape[-1])]

    for poc, frame in enumerate(frames):
        frame = Image.fromarray(frame)
        frame.save(os.path.join(frames_folder, "{}.png".format(poc)))


if __name__ == '__main__':
    for nii_filename in os.listdir(nii_folder):
        convert_ct_to_sequences(os.path.join(nii_folder, nii_filename))
