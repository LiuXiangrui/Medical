import os

import nibabel as nib
import numpy as np
from tqdm import tqdm

from Compress import write_yuv, bit_depth_conversion_based_on_interlacing


def convert_ct_to_sequences(nii_folder_: str, sequences_folder_: str, bit_depth: int = 16, crop_height: int = 128, crop_width: int = 128) -> None:
    os.makedirs(sequences_folder_, exist_ok=True)
    for nii_filename in tqdm(os.listdir(nii_folder_)):
        if os.path.splitext(nii_filename)[-1] != ".nii":
            continue

        sequence_folder = os.path.join(sequences_folder_, os.path.splitext(nii_filename)[0])
        os.makedirs(sequence_folder, exist_ok=True)

        nii_file = nib.load(os.path.join(nii_folder_, nii_filename))
        data = np.array(nii_file.get_fdata(dtype=np.float64)).transpose([1, 0, 2])

        min_ = data.min()
        max_ = data.max()
        data = (data - min_) / (max_ - min_) * (2 ** bit_depth)

        data = data.astype(np.uint16)

        height, width, num_frames = data.shape

        if crop_width < width and crop_height < height:
            top_corner = np.random.randint(low=0, high=height - crop_height)
            left_corner = np.random.randint(low=0, high=width - crop_width)

            frames = [data[top_corner: top_corner + crop_height, left_corner: left_corner + crop_width, i] for i in range(data.shape[-1])]
        else:
            frames = [data[:, :, i] for i in range(data.shape[-1])]

        odd_frames = []
        even_frames = []

        for frame in frames:
            odd_frame, even_frame = bit_depth_conversion_based_on_interlacing(frame)
            odd_frames.append(odd_frame)
            even_frames.append(even_frame)

        write_yuv(frames=odd_frames, yuv_path=os.path.join(sequence_folder, os.path.splitext(nii_filename)[0] + "_{}x{}_{}_{}_{}_odd.yuv".format(crop_width, crop_height, num_frames, int(min_), int(max_))))
        write_yuv(frames=even_frames, yuv_path=os.path.join(sequence_folder, os.path.splitext(nii_filename)[0] + "_{}x{}_{}_{}_{}_even.yuv".format(crop_width, crop_height, num_frames, int(min_), int(max_))))


if __name__ == "__main__":
    nii_folder = r"D:\MedicalImageDataset\Mosmed_COVID-19_CT\CT-3"
    seq_folder = r"C:\Users\xiangrliu3\Desktop\CTSequencesTest"
    convert_ct_to_sequences(nii_folder_=nii_folder, sequences_folder_=seq_folder, crop_height=512, crop_width=512)
