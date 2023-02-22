import os
from functools import partial

import numpy as np
from tqdm import tqdm


def interlaced_concat(odd_frame: np.ndarray, even_frame: np.ndarray) -> np.ndarray:
    odd_frame = odd_frame.astype(np.uint8)
    even_frame = even_frame.astype(np.uint8)

    assert odd_frame.shape == even_frame.shape

    frame = np.zeros_like(odd_frame, dtype=np.uint16)

    for i in range(odd_frame.shape[0]):
        for j in range(odd_frame.shape[1]):
            odd_number = bin(odd_frame[i][j])[2:].zfill(8)
            even_number = bin(even_frame[i][j])[2:].zfill(8)
            number = "".join(i + j for i, j in zip(odd_number, even_number))
            frame[i][j] = int("0b" + number, 2)

    return frame


def read_10bit_yuv_luma_component_convert_to_8bit(yuv_path: str, height: int, width: int) -> list:
    frames = []

    chroma_height = height // 2
    chroma_width = width // 2

    bytes2num = partial(int.from_bytes, byteorder='little', signed=False)

    with open(yuv_path, mode='rb') as f:
        while True:
            luma_array = np.zeros((height, width), dtype=np.uint)
            for h in range(height):
                for w in range(width):
                    data_bytes = f.read(2)
                    if data_bytes == b'':
                        return frames
                    pixel = np.uint16(bytes2num(data_bytes)) / 1024 * 256
                    luma_array[h, w] = pixel.astype(np.uint8)

            frames.append(luma_array)

            f.read(2 * chroma_height * chroma_width)  # drop U data
            f.read(2 * chroma_height * chroma_width)  # drop V data


def convert_sequences_to_ct(nii_folder_: str, sequences_folder_: str, height: int = 512, width: int = 512,
                            min_: float = -2048., max_: float = 2048., bit_depth: int = 16) -> None:
    os.makedirs(nii_folder_, exist_ok=True)

    for nii_filename in tqdm(os.listdir(sequences_folder_)):
        odd_frames = read_10bit_yuv_luma_component_convert_to_8bit(os.path.join(sequences_folder_, nii_filename, nii_filename + "_odd.yuv"), height=height, width=width)
        even_frames = read_10bit_yuv_luma_component_convert_to_8bit(os.path.join(sequences_folder_, nii_filename, nii_filename + "_even.yuv"), height=height, width=width)

        frames = []
        for odd_frame, even_frame in zip(odd_frames, even_frames):
            frame = interlaced_concat(odd_frame, even_frame=even_frame).astype(np.uint16)
            frames.append(frame)

        data = np.stack(frames, axis=-1).astype(np.float64)
        data = data / (2 ** bit_depth) * (max_ - min_) + min_

        np.save(os.path.join(nii_folder_, nii_filename + ".npy"), data)


if __name__ == "__main__":
    nii_folder = r"D:\Rec"
    sequences_folder = r"D:\CTSequences"
    convert_sequences_to_ct(nii_folder_=nii_folder, sequences_folder_=sequences_folder)
