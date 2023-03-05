import os
from functools import partial
from multiprocessing import Pool

import h5py
import numpy as np

ori_folder = r"C:\Users\xiangrliu3\Desktop\CTSequences"
rec_folder = r"C:\Users\xiangrliu3\Desktop\RecCTSequences22"

h5_temp_folder = r"C:\Users\xiangrliu3\Desktop\H5QP22"
h5_file = r"C:\Users\xiangrliu3\Desktop\QP22.h5py"

num_proc = 16


def make_h5py(ori_yuv: str, rec_yuv: str):
    width = int(ori_yuv.split('_')[-5].split('x')[0])
    height = int(ori_yuv.split('_')[-5].split('x')[0])
    num_frames = int(ori_yuv.split('_')[-4])

    chroma_height = height // 2
    chroma_width = width // 2

    ori_frames = []
    with open(ori_yuv, mode='rb') as f:
        frame_count = 0
        while frame_count < num_frames:
            data_bytes = f.read(height * width)
            luma_array = np.reshape(np.frombuffer(data_bytes, 'B'), (height, width))
            ori_frames.append(luma_array)

            f.read(chroma_height * chroma_width)  # drop U data
            f.read(chroma_height * chroma_width)  # drop V data

            frame_count += 1

    bytes2num = partial(int.from_bytes, byteorder='little', signed=False)

    rec_frames = []
    with open(rec_yuv, mode='rb') as f:
        frame_count = 0
        while frame_count < num_frames:
            luma_array = np.zeros((height, width), dtype=np.uint)
            for h in range(height):
                for w in range(width):
                    data_bytes = f.read(2)
                    pixel = np.uint16(bytes2num(data_bytes)) / 1024 * 256
                    luma_array[h, w] = pixel.astype(np.uint8)
            rec_frames.append(luma_array)

            f.read(2 * chroma_height * chroma_width)  # drop U data
            f.read(2 * chroma_height * chroma_width)  # drop V data

            frame_count += 1

    h5_filepath = os.path.join(h5_temp_folder, os.path.splitext(os.path.split(ori_yuv)[-1])[0] + ".h5py")
    with h5py.File(h5_filepath, mode='w') as f:
        frame_count = 0
        for ori_frame, rec_frame in zip(ori_frames, rec_frames):
            f.create_dataset(name=str(frame_count), data=np.array([ori_frame, rec_frame]), compression="gzip", compression_opts=4)
            frame_count += 1


def merge_h5():
    with h5py.File(h5_file, mode='w') as h:
        for h5_filename in os.listdir(h5_temp_folder):
            if os.path.splitext(h5_filename)[-1] != '.h5py':
                continue
            seq_name = '_'.join(h5_filename.split('_')[:2])
            seq_name += '_{}'.format(os.path.splitext(h5_filename)[0].split('_')[-1])
            with h5py.File(os.path.join(h5_temp_folder, h5_filename), mode='r') as f:
                for idx, data in f.items():
                    h.create_dataset(name="{}_{}".format(seq_name, idx), data=data, compression="gzip", compression_opts=4)


if __name__ == "__main__":
    assert os.path.split(h5_file)[0] != h5_temp_folder
    args_list = []
    for seq_name in os.listdir(rec_folder):
        for yuv_name in os.listdir(os.path.join(rec_folder, seq_name)):
            ori_yuv = os.path.join(ori_folder, seq_name, yuv_name)
            rec_yuv = os.path.join(rec_folder, seq_name, yuv_name)
            args_list.append([ori_yuv, rec_yuv])
    os.makedirs(h5_folder, exist_ok=True)

    p = Pool(processes=num_proc)
    for arg in args_list:
        p.apply_async(make_h5py, args=(arg[0], arg[1]))
    p.close()
    p.join()

    # merge_h5()



