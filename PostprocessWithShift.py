import os
from multiprocessing import Pool

import numpy as np

num_proc = 16


def shift_x_bit_to_16_bit(frame: np.ndarray, x: int) -> np.ndarray:
    assert x == 10 or x == 8
    frame = np.left_shift(frame, 16 - x)
    return frame

def read_10bit_yuv_luma_component(yuv_path: str, height: int, width: int, num_frames: int) -> list:
    frames = []

    chroma_height = height // 2
    chroma_width = width // 2

    with open(yuv_path, mode='rb') as f:
        for idx in range(num_frames):
            luma_array = np.zeros((height, width), dtype=np.uint16)
            for h in range(height):
                for w in range(width):
                    data_bytes = f.read(2)
                    pixel = np.frombuffer(data_bytes, np.uint16)
                    luma_array[h, w] = pixel.astype(np.uint16)
            frames.append(luma_array)

            f.read(2 * chroma_height * chroma_width)  # drop U data
            f.read(2 * chroma_height * chroma_width)  # drop V data
    return frames


def convert_sequences_to_ct(nii_folder_: str, sequences_folder_: str, nii_filename: str, qp: int, bit_depth: int = 16) -> None:

    min_ = max_ = num_frames = height = width = 0
    for yuv_filename in os.listdir(os.path.join(sequences_folder_, nii_filename, str(qp))):
        if os.path.splitext(yuv_filename)[-1] != ".yuv":
            continue
        yuv_filename = os.path.splitext(yuv_filename)[0]

        max_ = int(yuv_filename.split("_")[-1])
        min_ = int(yuv_filename.split("_")[-2])
        num_frames = int(yuv_filename.split("_")[-3])
        height = int(yuv_filename.split("_")[-4].split('x')[1])
        width = int(yuv_filename.split("_")[-4].split('x')[0])

    frames = read_10bit_yuv_luma_component(os.path.join(sequences_folder_, nii_filename, str(qp), "{}_{}x{}_{}_{}_{}.yuv".format(nii_filename, width, height, num_frames, min_, max_)),
                                           height=height, width=width, num_frames=num_frames)

    data = np.stack(frames, axis=-1)  # shape (H, W, C)
    data = np.flipud(data)

    data = data.transpose([2, 1, 0])
    data = data[:, :, :, np.newaxis]

    data = shift_x_bit_to_16_bit(data, x=10)

    data = data / (2 ** bit_depth) * (max_ - min_) + min_

    np.save(os.path.join(nii_folder_, str(qp), nii_filename + ".npy"), data)


if __name__ == "__main__":
    nii_folder = r"C:\Users\xiangrliu3\Desktop\10bitExperiments\MRIRecNIITest10Bit"
    sequences_folder = r"C:\Users\xiangrliu3\Desktop\10bitExperiments\MRIRecSeqTest10Bit"
    qp_list = [-16, -8, 1, 63]
    p = Pool(processes=num_proc)
    for qp in qp_list:
        os.makedirs(os.path.join(nii_folder, str(qp)), exist_ok=True)
        for nii_filename in os.listdir(sequences_folder):
            p.apply_async(convert_sequences_to_ct, args=(nii_folder, sequences_folder, nii_filename, qp))

    p.close()
    p.join()