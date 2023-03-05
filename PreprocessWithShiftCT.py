import os

import nibabel as nib
import numpy as np
from tqdm import tqdm


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


def shift_16bit_to_x_bit(frame: np.ndarray, x: int) -> np.ndarray:
    assert x == 10 or x == 8
    frame = np.right_shift(frame, 16 - x)
    return frame


def write_8bit_yuv(frames: list, yuv_path: str) -> None:
    with open(yuv_path, mode='w') as f:
        for frame in frames:
            np.asarray(frame, dtype=np.uint8).tofile(f)
            # fill chroma channels
            chroma_placeholder = np.ones([frame.shape[0] // 2, frame.shape[1] // 2]) * 128
            chroma_placeholder.astype(np.uint8).tofile(f)
            chroma_placeholder.astype(np.uint8).tofile(f)


def write_10bit_yuv(frames: list, yuv_path: str) -> None:
    with open(yuv_path, mode='w') as f:
        for frame in frames:
            np.asarray(frame, dtype=np.uint16).tofile(f)
            # fill chroma channels
            chroma_placeholder = np.ones([frame.shape[0] // 2, frame.shape[1] // 2]) * 512
            chroma_placeholder.astype(np.uint16).tofile(f)
            chroma_placeholder.astype(np.uint16).tofile(f)


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

        frames = [shift_16bit_to_x_bit(frame, x=10) for frame in frames]

        height = min(height, crop_height)
        width = min(width, crop_width)

        write_10bit_yuv(frames=frames, yuv_path=os.path.join(sequence_folder, os.path.splitext(nii_filename)[0] + "_{}x{}_{}_{}_{}.yuv".format(width, height, num_frames, int(min_), int(max_))))


if __name__ == "__main__":
    nii_folder = r"D:\MedicalImageDataset\Mosmed_COVID-19_CT\CT-3"
    seq_folder = r"C:\Users\xiangrliu3\Desktop\10bitExperiments\CTSequencesTest10Bit"
    convert_ct_to_sequences(nii_folder_=nii_folder, sequences_folder_=seq_folder, crop_height=512, crop_width=512)
