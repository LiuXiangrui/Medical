import struct

import numpy as np
import nibabel as nib
import os


def read_yuv(yuv_10bit_path: str, height: int, width: int, num_frames: int, output_8_bit: bool = False) -> list:
    frames = []

    chroma_height = height // 2
    chroma_width = width // 2

    output_data_type = np.uint8 if output_8_bit else np.uint16

    with open(yuv_10bit_path, mode='rb') as f:
        for idx in range(num_frames):
            luma_array = np.zeros((height, width), dtype=output_data_type)
            for h in range(height):
                for w in range(width):
                    data_bytes = f.read(2)
                    pixel = np.frombuffer(data_bytes, np.uint16)
                    if output_8_bit:
                        pixel = pixel / 1024. * 256.
                    luma_array[h, w] = pixel.astype(output_data_type)
            frames.append(luma_array)

            f.read(2 * chroma_height * chroma_width)  # drop U data
            f.read(2 * chroma_height * chroma_width)  # drop V data

    return frames


def inverse_bit_depth_conversion_based_on_interlacing(odd_slice: np.ndarray, even_slice: np.ndarray) -> np.ndarray:
    odd_slice = odd_slice.astype(np.uint8)
    even_slice = even_slice.astype(np.uint8)

    assert odd_slice.shape == even_slice.shape

    slices_16bit = np.zeros_like(odd_slice, dtype=np.uint16)

    for i in range(odd_slice.shape[0]):
        for j in range(odd_slice.shape[1]):
            odd_number = bin(odd_slice[i][j])[2:].zfill(8)
            even_number = bin(even_slice[i][j])[2:].zfill(8)
            number = "".join(i + j for i, j in zip(odd_number, even_number))
            slices_16bit[i][j] = int("0b" + number, 2)

    return slices_16bit


def shift_x_bit_to_16_bit(slices: np.ndarray, x: int) -> np.ndarray:
    assert x == 10 or x == 8
    slices = np.left_shift(slices, 16 - x)
    return slices


def slice_splicing(slices: list, min_: int, max_: int, rec_nii_path: str) -> None:
    bit_depth = 16
    rec_nii = np.stack(slices, axis=-1).transpose([1, 0, 2])
    rec_nii = rec_nii / (2 ** bit_depth) * (max_ - min_) + min_
    np.save(rec_nii_path, rec_nii)


def decode_flags(bin_filepath: str) -> dict:
    with open(bin_filepath, mode='rb') as f:
        flag = struct.unpack('B', f.read(1))[0]
        use_interlaced_split = True if flag == 0 else False
        shift_to_10_bit = True if flag == 3 else False

        min_ = struct.unpack('i', f.read(4))[0]
        max_ = struct.unpack('i', f.read(4))[0]

        # note that below flags is duplicated in the bitstream of sequences
        height = struct.unpack('H', f.read(2))[0]
        width = struct.unpack('H', f.read(2))[0]
        num_frames = struct.unpack('H', f.read(2))[0]

    return {"min_": min_, "max_": max_, "use_interlaced_split": use_interlaced_split, "shift_to_10_bit": shift_to_10_bit,
            "height": height, "width": width, "num_frames": num_frames}


def postprocessing(temp_folder: str, rec_nii_path: str):
    # decode flags
    flags = None
    for flag_bin_filename in os.listdir(temp_folder):
        if os.path.splitext(flag_bin_filename)[-1] != '.flag':
            continue
        flags = decode_flags(bin_filepath=os.path.join(temp_folder, flag_bin_filename))

    assert flags is not None

    min_ = flags["min_"]
    max_ = flags["max_"]
    use_interlaced_split = flags["use_interlaced_split"]
    shift_to_10_bit = flags["shift_to_10_bit"]
    height = flags["height"]
    width = flags["width"]
    num_frames = flags["num_frames"]

    rec_slices = []
    if use_interlaced_split:
        odd_slices = even_slices = None

        for yuv_filename in os.listdir(temp_folder):
            if "_even_rec.yuv" in yuv_filename:
                even_slices = read_yuv(os.path.join(temp_folder, yuv_filename), height=height, width=width,
                                       num_frames=num_frames, output_8_bit=True)
            if "_odd_rec.yuv" in yuv_filename:
                odd_slices = read_yuv(os.path.join(temp_folder, yuv_filename), height=height, width=width,
                                      num_frames=num_frames, output_8_bit=True)
            os.remove(os.path.join(temp_folder, yuv_filename))

        assert odd_slices is not None and even_slices is not None

        for odd_slice, even_slice in zip(odd_slices, even_slices):
            rec_slice = inverse_bit_depth_conversion_based_on_interlacing(odd_slice=odd_slice, even_slice=even_slice).astype(np.uint16)
            rec_slices.append(rec_slice)

    else:
        for yuv_filename in os.listdir(temp_folder):
            if os.path.splitext(yuv_filename)[-1] != ".yuv":
                continue
            rec_slices_x_bit = read_yuv(os.path.join(temp_folder, yuv_filename), height=height, width=width,
                                        num_frames=num_frames, output_8_bit=(not shift_to_10_bit))
            for rec_slice in rec_slices_x_bit:
                rec_slices.append(shift_x_bit_to_16_bit(slices=rec_slice, x=10 if shift_to_10_bit else 8))

    assert len(rec_slices) != 0

    slice_splicing(slices=rec_slices, max_=max_, min_=min_, rec_nii_path=rec_nii_path)


def decode(decoder_path: str, temp_folder: str):
    for bin_filename in os.listdir(temp_folder):
        if os.path.splitext(bin_filename)[-1] != '.bin':
            continue
        bin_filepath = os.path.join(temp_folder, bin_filename)
        rec_filepath = os.path.join(temp_folder, os.path.splitext(bin_filepath)[0] + '_rec.yuv')
        dec_cmd = "{} --input {} --output {}".format(decoder_path, bin_filepath, rec_filepath)
        os.system(dec_cmd)


def decompress(decoder_path: str, temp_root: str):
    for nii_filename in os.listdir(temp_root):
        if not os.path.isdir(nii_filename):
            continue

        decode(decoder_path=decoder_path, temp_folder=os.path.join(temp_root, nii_filename))

        postprocessing(temp_folder=os.path.join(temp_root, nii_filename), rec_nii_path=os.path.join(temp_root, nii_filename, nii_filename + ".nii"))
