import struct

import numpy as np
import nibabel as nib
import os


def write_yuv(frames: list, yuv_path: str, bit_depth: int) -> None:
    assert bit_depth == 10 or bit_depth == 8

    data_type = np.uint8 if bit_depth == 8 else np.uint10
    chroma_value = (2 ** bit_depth) // 2

    with open(yuv_path, mode='w') as f:
        for frame in frames:
            np.asarray(frame, dtype=data_type).tofile(f)
            # fill chroma channels
            chroma_placeholder = np.ones([frame.shape[0] // 2, frame.shape[1] // 2]) * chroma_value
            chroma_placeholder.astype(data_type).tofile(f)
            chroma_placeholder.astype(data_type).tofile(f)


def bit_depth_conversion_based_on_interlacing(slice_16_bit: np.ndarray) -> tuple:
    slice_16_bit = slice_16_bit.astype(np.uint16)
    odd_slice = np.zeros_like(slice_16_bit, dtype=np.uint8)
    even_slice = np.zeros_like(slice_16_bit, dtype=np.uint8)

    for i in range(slice_16_bit.shape[0]):
        for j in range(slice_16_bit.shape[1]):
            number = bin(slice_16_bit[i][j])[2:].zfill(16)
            odd_slice[i][j] = int("0b" + number[0::2], 2)
            even_slice[i][j] = int("0b" + number[1::2], 2)

    return odd_slice, even_slice


def shift_16bit_to_x_bit(slice_16_bit: np.ndarray, x: int) -> np.ndarray:
    assert x == 10 or x == 8
    slice_x_bit = np.right_shift(slice_16_bit, 16 - x)
    return slice_x_bit


def slice_partition(nii_filepath: str):
    bit_depth = 16

    nii_file = nib.load(nii_filepath)
    data = np.array(nii_file.get_fdata(dtype=np.float64)).transpose([1, 0, 2])

    min_ = int(data.min())
    max_ = int(data.max())
    data = (data - min_) / (max_ - min_) * (2 ** bit_depth)

    data = data.astype(np.uint16)

    height, width, num_frames = data.shape

    slices = [data[:, :, i] for i in range(data.shape[-1])]

    return {"slices": slices, "height": height, "width": width, "num_frames": num_frames, "min_": min_, "max_": max_}


def encode_flags(bin_filepath: str, min_: int, max_: int, use_interlaced_split: bool, shift_to_10_bit: bool,
                 height: int, width: int, num_frames: int) -> None:
    with open(bin_filepath, mode='wb') as f:
        flag = 2 * int(not use_interlaced_split) + int((not use_interlaced_split) and shift_to_10_bit)
        f.write(struct.pack('B', flag))  #  4-bit

        f.write(struct.pack('i', min_))  # 32-bit
        f.write(struct.pack('i', max_))  # 32-bit

        # note that below flags is duplicated in the bitstream of sequences
        f.write(struct.pack('H', height))  # 16-bit
        f.write(struct.pack('H', width))  # 16-bit
        f.write(struct.pack('H', num_frames))  # 16-bit


def preprocess(nii_filepath: str, temp_folder: str, use_interlaced_split: bool = True, shift_to_10_bit: bool = True):
    assert os.path.splitext(nii_filepath)[-1] == ".nii"

    partition_results = slice_partition(nii_filepath=nii_filepath)
    slices = partition_results["slices"]

    nii_filename = os.path.splitext(os.path.split(nii_filepath)[-1])[0]
    yuv_name_prefix = "{}_{}x{}_{}".format(nii_filename, str(partition_results["width"]),
                                           str(partition_results["height"]), str(partition_results["num_frames"]))

    # convert 3D medical image to 2D medical sequence
    if use_interlaced_split:
        odd_slices = []
        even_slices = []
        for slice_16_bit in slices:
            odd_slice, even_slice = bit_depth_conversion_based_on_interlacing(slice_16_bit)
            odd_slices.append(odd_slice)
            even_slices.append(even_slice)

        write_yuv(frames=odd_slices, yuv_path=os.path.join(temp_folder, yuv_name_prefix + "_odd.yuv"), bit_depth=8)
        write_yuv(frames=odd_slices, yuv_path=os.path.join(temp_folder, yuv_name_prefix + "_even.yuv"), bit_depth=8)

    else:
        if shift_to_10_bit:
            slices = [shift_16bit_to_x_bit(slice_16_bit, x=10) for slice_16_bit in slices]
            write_yuv(frames=slices, yuv_path=os.path.join(temp_folder, yuv_name_prefix + "_10bit.yuv"), bit_depth=10)
        else:
            slices = [shift_16bit_to_x_bit(slice_16_bit, x=8) for slice_16_bit in slices]
            write_yuv(frames=slices, yuv_path=os.path.join(temp_folder, yuv_name_prefix + "_8bit.yuv"), bit_depth=8)

    # encode flags
    flag_bin_filepath = os.path.join(temp_folder, nii_filename + ".flag")
    encode_flags(bin_filepath=flag_bin_filepath, min_=partition_results["min_"], max_=partition_results["max_"],
                 use_interlaced_split=use_interlaced_split, shift_to_10_bit=shift_to_10_bit,
                 height=partition_results["height"], width=partition_results["width"], num_frames=partition_results["num_frames"])


def encode(encoder_path: str, cfg_path: str, yuv_folder: str, qp: int, shift_to_10_bit: bool = True):
    for yuv_filename in os.listdir(yuv_folder):
        if os.path.splitext(yuv_filename)[-1] != '.yuv':
            continue

        yuv_filepath = os.path.join(yuv_folder, yuv_filename)
        bin_filepath = os.path.join(yuv_folder, os.path.splitext(yuv_filename)[0] + '.bin')
        res_filepath = os.path.join(yuv_folder, os.path.splitext(yuv_filename)[0] + '.txt')

        num_frames = yuv_filename.split('_')[-2]
        resolution = yuv_filename.split('_')[-3]
        width, height = resolution.split('x')
        bit_depth = 10 if shift_to_10_bit else 8

        enc_cmd = "{} --config {} --input {} --output {} --width {} --height {} --frames {} --op_qp {} --input_bit_depth {} > {}".format(
                encoder_path, cfg_path, yuv_filepath, bin_filepath, width, height, num_frames, qp, bit_depth, res_filepath
        )

        os.system(enc_cmd)
        os.remove(yuv_filepath)


def compress(encoder_path: str, cfg_path: str, qp: int, nii_filepath: str, temp_root: str, use_interlaced_split: bool = True, shift_to_10_bit: bool = True):
    nii_filename = os.path.splitext(nii_filepath)[0]

    temp_folder = os.path.join(temp_root, nii_filename)
    os.makedirs(temp_folder, exist_ok=True)

    preprocess(nii_filepath=nii_filepath, temp_folder=temp_folder, use_interlaced_split=use_interlaced_split, shift_to_10_bit=shift_to_10_bit)
    encode(encoder_path=encoder_path, cfg_path=cfg_path, yuv_folder=temp_folder, shift_to_10_bit=shift_to_10_bit, qp=qp)




