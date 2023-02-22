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


def write_yuv(frames: list, yuv_path: str) -> None:
    with open(yuv_path, mode='w') as f:
        for frame in frames:
            np.asarray(frame, dtype=np.uint8).tofile(f)
            # fill chroma channels
            chroma_placeholder = np.ones([frame.shape[0] // 2, frame.shape[1] // 2]) * 128
            chroma_placeholder.astype(np.uint8).tofile(f)
            chroma_placeholder.astype(np.uint8).tofile(f)


def convert_ct_to_sequences(nii_folder_: str, sequences_folder_: str, bit_depth: int = 16) -> None:
    os.makedirs(sequences_folder_, exist_ok=True)
    for nii_filename in tqdm(os.listdir(nii_folder_)):
        if os.path.splitext(nii_filename)[-1] != ".nii":
            continue

        sequence_folder = os.path.join(sequences_folder_, os.path.splitext(nii_filename)[0])
        os.makedirs(sequence_folder, exist_ok=True)

        nii_file = nib.load(os.path.join(nii_folder_, nii_filename))  # axis is [z, x, y]
        data = np.array(nii_file.get_fdata(dtype=np.float64))

        #
        # # first choice: clip
        # if choice == 1:
        #     data = (data - min_) / (max_ - min_) * (2 ** bit_depth)
        # # second choice: shift to zero point
        # elif choice == 2:
        #     data = data + data.min()
        # # third choice: scale to 0~65536
        # else:
        min_ = data.min()
        max_ = data.max()
        data = (data - min_) / (max_ - min_) * (2 ** bit_depth)

        data = data.astype(np.uint16)

        height, width, num_frames = data.shape

        frames = [data[:, :, i] for i in range(data.shape[-1])]

        odd_frames = []
        even_frames = []

        for frame in frames:
            odd_frame, even_frame = interlaced_split(frame)
            odd_frames.append(odd_frame)
            even_frames.append(even_frame)

        write_yuv(frames=odd_frames, yuv_path=os.path.join(sequence_folder, os.path.splitext(nii_filename)[0] + "_{}x{}_{}_{}_{}_odd.yuv".format(width, height, num_frames, int(min_), int(max_))))
        write_yuv(frames=even_frames, yuv_path=os.path.join(sequence_folder, os.path.splitext(nii_filename)[0] + "_{}x{}_{}_{}_{}_even.yuv".format(width, height, num_frames, int(min_), int(max_))))


if __name__ == "__main__":
    nii_folder = ""
    seq_folder = ""
    convert_ct_to_sequences(nii_folder_=nii_folder, sequences_folder_=seq_folder)
