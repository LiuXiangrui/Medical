import os
from multiprocessing import Pool

import numpy as np

num_proc = 16


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


def read_10bit_yuv_luma_component_convert_to_8bit(yuv_path: str, height: int, width: int, num_frames: int) -> list:
    frames = []

    chroma_height = height // 2
    chroma_width = width // 2

    with open(yuv_path, mode='rb') as f:
        for idx in range(num_frames):
            luma_array = np.zeros((height, width), dtype=np.uint8)
            for h in range(height):
                for w in range(width):
                    data_bytes = f.read(2)
                    pixel = np.frombuffer(data_bytes, np.uint16)
                    pixel = pixel.astype(np.uint16) / 1024 * 256
                    luma_array[h, w] = pixel.astype(np.uint8)
            frames.append(luma_array)

            f.read(2 * chroma_height * chroma_width)  # drop U data
            f.read(2 * chroma_height * chroma_width)  # drop V data
    return frames


def convert_sequences_to_ct(nii_folder_: str, sequences_folder_: str, nii_filename: str, qp: int, bit_depth: int = 16) -> None:

    min_ = max_ = num_frames = height = width = 0
    for yuv_filename in os.listdir(os.path.join(sequences_folder_, nii_filename, str(qp))):
        max_ = int(yuv_filename.split("_")[-2])
        min_ = int(yuv_filename.split("_")[-3])
        num_frames = int(yuv_filename.split("_")[-4])
        height = int(yuv_filename.split("_")[-5].split('x')[1])
        width = int(yuv_filename.split("_")[-5].split('x')[0])

    odd_yuv_path = os.path.join(sequences_folder_, nii_filename, str(qp),  "{}_{}x{}_{}_{}_{}_odd.yuv".format(nii_filename, width, height, num_frames, min_, max_))
    even_yuv_path = os.path.join(sequences_folder_, nii_filename, str(qp), "{}_{}x{}_{}_{}_{}_even.yuv".format(nii_filename, width, height, num_frames, min_, max_))
    odd_frames = read_10bit_yuv_luma_component_convert_to_8bit(odd_yuv_path, height=height, width=width, num_frames=num_frames)
    even_frames = read_10bit_yuv_luma_component_convert_to_8bit(even_yuv_path, height=height, width=width, num_frames=num_frames)

    frames = []
    for odd_frame, even_frame in zip(odd_frames, even_frames):
        frame = interlaced_concat(odd_frame, even_frame=even_frame).astype(np.uint16)
        frames.append(frame)

    # data = np.flipud(data) # for CT, no flipud

    data = np.stack(frames, axis=-1)

    # data = data.transpose([2, 1, 0])
    # data = data[:, :, :, np.newaxis]  # MRI

    data = data.transpose([1, 0, 2])  # CT

    data = data / (2 ** bit_depth) * (max_ - min_) + min_

    np.save(os.path.join(nii_folder_, str(qp), nii_filename + ".npy"), data)

if __name__ == "__main__":
    # MRI
    # nii_folder = r"D:\AVS3_MRI_Anchor_Results\MRIRecNIITestInterlaced"
    # sequences_folder = r"D:\AVS3_MRI_Anchor_Results\MRI_Test_Results_AVS3"
    # qp_list = [22, 26, 30, 34]
    # p = Pool(processes=num_proc)
    # for qp in qp_list:
    #     os.makedirs(os.path.join(nii_folder, str(qp)), exist_ok=True)
    #     for nii_filename in os.listdir(sequences_folder):
    #         p.apply_async(convert_sequences_to_ct, args=(nii_folder, sequences_folder, nii_filename, qp))
    # p.close()
    # p.join()
    nii_folder = r"D:\AVS3_CT_Anchor_Results\CTRecNIITestInterlaced"
    sequences_folder = r"D:\AVS3_CT_Anchor_Results\CT_Test_Results_AVS3"
    qp_list = [16, 20, 25, 29]
    p = Pool(processes=num_proc)
    os.makedirs(nii_folder, exist_ok=True)
    for qp in qp_list:
        os.makedirs(os.path.join(nii_folder, str(qp)), exist_ok=True)
        for nii_filename in os.listdir(sequences_folder):
            p.apply_async(convert_sequences_to_ct, args=(nii_folder, sequences_folder, nii_filename, qp))
            # convert_sequences_to_ct(nii_folder, sequences_folder, nii_filename, qp)
            # break
    p.close()
    p.join()
