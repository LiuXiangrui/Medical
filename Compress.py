import argparse
import os
import struct
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np


class Compression:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_type", type=str, choices=["CT", "MRI"], help="data type of input 3D medical image, option = [CT, MRI]")
        parser.add_argument("--nii_filepath", type=str, help="filepath of 3D medical image to be encoded")
        parser.add_argument("--bin_filepath", type=str, help="filepath to store bit streams")

        parser.add_argument("--use_DCBS", action="store_true", help="use bit conversion based on bit-shifting")
        parser.add_argument("--shift_to_10_bit", action="store_true", help="generate 10-bit sequences when using bit conversion based on bit-shifting")

        parser.add_argument("--encoder_path", type=str, help="path of AVS3 encoder")
        parser.add_argument("--cfg_path", type=str, help="filepath of configuration files of AVS3 encoder")
        parser.add_argument("--qp", type=int, help="quantization parameter sent to AVS3 encoder")

        self.args = parser.parse_args()

        self.height = self.width = self.num_frames = 0
        self.temp_folder_path = None

    def compress(self) -> None:
        with TemporaryDirectory() as temp_folder_path:
            self.temp_folder_path = temp_folder_path

            self.preprocess()

            self.encode_medical_sequence()

            self.merge_bitstreams()

    def merge_bitstreams(self):
        bitstream_list = [os.path.join(self.temp_folder_path, "flag.bin"), ]
        if self.args.use_DCBS:
            bitstream_list.append(os.path.join(self.temp_folder_path, "shift.bin"))
        else:
            bitstream_list.append(os.path.join(self.temp_folder_path, "odd.bin"))
            bitstream_list.append(os.path.join(self.temp_folder_path, "even.bin"))

        with open(self.args.bin_filepath, mode='wb') as dst_f:
            for bitstream_path in bitstream_list:
                len_bitstream = os.path.getsize(bitstream_path)  # length of bitstream in bytes
                dst_f.write(struct.pack('I', len_bitstream))
                with open(bitstream_path, mode='rb') as src_f:
                    data = src_f.read()
                    dst_f.write(data)
                os.remove(bitstream_path)

    def preprocess(self):
        # slice partition
        partition_results = self.slice_partition()
        slices = partition_results["slices"]

        self.height = partition_results["height"]
        self.width = partition_results["width"]
        self.num_frames = partition_results["num_frames"]

        # convert 3D medical image to 2D medical sequence
        if self.args.use_DCBS:
            slices = [self.shift_16bit_to_x_bit(slice_16_bit, x=10) for slice_16_bit in slices] if self.args.shift_to_10_bit \
                else [self.shift_16bit_to_x_bit(slice_16_bit, x=8) for slice_16_bit in slices]

            self.write_yuv(frames=slices, yuv_path=os.path.join(self.temp_folder_path, "shift.yuv"))
        else:
            odd_slices = []
            even_slices = []
            for slice_16_bit in slices:
                odd_slice, even_slice = self.bit_depth_conversion_based_on_interlacing(slice_16_bit)
                odd_slices.append(odd_slice)
                even_slices.append(even_slice)

            self.write_yuv(frames=odd_slices, yuv_path=os.path.join(self.temp_folder_path, "odd.yuv"))
            self.write_yuv(frames=even_slices, yuv_path=os.path.join(self.temp_folder_path, "even.yuv"))

        # encode flags
        self.encode_flags(min_=partition_results["min_"], max_=partition_results["max_"],
                          height=partition_results["height"], width=partition_results["width"],
                          num_frames=partition_results["num_frames"])

    def encode_medical_sequence(self):
        yuv_list = [os.path.join(self.temp_folder_path, "shift.yuv"), ] if self.args.use_DCBS \
            else [os.path.join(self.temp_folder_path, "odd.yuv"), os.path.join(self.temp_folder_path, "even.yuv")]

        for yuv_filepath in yuv_list:

            bin_filepath = os.path.splitext(yuv_filepath)[0] + '.bin'

            bit_depth = 10 if self.args.use_DCBS and self.args.shift_to_10_bit else 8

            enc_cmd = "{} --config {} --input {} --output {} --width {} --height {} --frames {} --op_qp {} --i_period 48 --input_bit_depth {}".format(
                self.args.encoder_path, self.args.cfg_path, yuv_filepath, bin_filepath, self.width, self.height, self.num_frames, self.args.qp, bit_depth
            )

            os.system(enc_cmd)
            os.remove(yuv_filepath)

    def slice_partition(self):
        bit_depth = 16

        assert os.path.splitext(self.args.nii_filepath)[-1] == ".nii", "invalid file format"
        nii_file = nib.load(self.args.nii_filepath)

        if self.args.data_type == "CT":
            data = np.array(nii_file.get_fdata(dtype=np.float64)).transpose([1, 0, 2])
        else:
            data = np.array(nii_file.get_fdata(dtype=np.float64)).transpose([3, 2, 1, 0])[0]

        # scale to the range of 16-bit unsigned integer
        min_ = int(data.min())
        max_ = int(data.max())
        data = (data - min_) / (max_ - min_) * (2 ** bit_depth)

        data = data.astype(np.uint16)

        data = np.flipud(data)

        height, width, num_frames = data.shape

        # partition to slices along the last axis
        slices = [data[:, :, i] for i in range(data.shape[-1])]

        return {"slices": slices, "height": height, "width": width, "num_frames": num_frames, "min_": min_, "max_": max_}

    def encode_flags(self, min_: int, max_: int, height: int, width: int, num_frames: int) -> None:
        """
        encode flags to bitstream and return the bits-per-voxel
        """
        flag_filepath = os.path.join(self.temp_folder_path, "flag.bin")

        data_type_ct = True if self.args.data_type == "CT" else False

        with open(flag_filepath, mode='wb') as f:
            """
            |  Data Type | Depth Conversion Method | Shift-to-10bit (if use DCBS)|
            | CT=1, MRI=0|      DCBS=1, else=0     |       10bit=1, 8bit=0       |
            """
            flag = 4 * int(data_type_ct) + 2 * int(self.args.use_DCBS) + int(self.args.shift_to_10_bit)
            f.write(struct.pack('B', flag))  # 8-bit  # note that I can't fine a way to write 3 bits in Python, so just use 8 bit -_-|||

            f.write(struct.pack('i', min_))  # 32-bit
            f.write(struct.pack('i', max_))  # 32-bit

            # note that below flags is duplicated in the bitstream of sequences
            f.write(struct.pack('H', height))  # 16-bit
            f.write(struct.pack('H', width))  # 16-bit
            f.write(struct.pack('H', num_frames))  # 16-bit

    def write_yuv(self, frames: list, yuv_path: str) -> None:
        """
        write converted sequences as yuv file
        """
        bit_depth = 10 if self.args.use_DCBS and self.args.shift_to_10_bit else 8

        data_type = np.uint8 if bit_depth == 8 else np.uint16
        chroma_value = (2 ** bit_depth) // 2

        with open(yuv_path, mode='w') as f:
            for frame in frames:
                np.asarray(frame, dtype=data_type).tofile(f)
                # fill chroma channels with constant value
                chroma_placeholder = np.ones([frame.shape[0] // 2, frame.shape[1] // 2]) * chroma_value
                chroma_placeholder.astype(data_type).tofile(f)
                chroma_placeholder.astype(data_type).tofile(f)

    @staticmethod
    def shift_16bit_to_x_bit(slice_16_bit: np.ndarray, x: int) -> np.ndarray:
        assert x == 10 or x == 8
        slice_x_bit = np.right_shift(slice_16_bit, 16 - x)
        return slice_x_bit

    @staticmethod
    def bit_depth_conversion_based_on_interlacing(slice_16_bit: np.ndarray) -> tuple:
        """
        convert a 16-bit slice to two 8-bit slices using bit-interlacing
        """
        slice_16_bit = slice_16_bit.astype(np.uint16)
        odd_slice = np.zeros_like(slice_16_bit, dtype=np.uint8)
        even_slice = np.zeros_like(slice_16_bit, dtype=np.uint8)

        for i in range(slice_16_bit.shape[0]):
            for j in range(slice_16_bit.shape[1]):
                number = bin(slice_16_bit[i][j])[2:].zfill(16)
                odd_slice[i][j] = int("0b" + number[0::2], 2)
                even_slice[i][j] = int("0b" + number[1::2], 2)

        return odd_slice, even_slice


if __name__ == "__main__":
    compressor = Compression()
    compressor.compress()
