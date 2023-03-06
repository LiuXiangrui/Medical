import argparse
import os
import struct
from tempfile import TemporaryDirectory

import numpy as np


class Decompression:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--bin_filepath", type=str, help="filepath of bit streams")
        parser.add_argument("--nii_filepath", type=str, help="filepath of reconstructed 3D medical image")

        parser.add_argument("--decoder_path", type=str, help="path of AVS3 decoder")

        self.args = parser.parse_args()

        self.min_ = self.max_ = self.height = self.width = self.num_frames = 0
        self.temp_folder_path = None

        self.data_type = None
        self.use_DCBS = False
        self.shift_to_10_bit = False

    def decompress(self):
        with TemporaryDirectory() as temp_folder_path:
            self.temp_folder_path = temp_folder_path

            self.split_bitstream_and_decode_flags()

            self.decode_medical_sequences()

            self.postprocess()

    def split_bitstream_and_decode_flags(self):
        with open(self.args.bin_filepath, mode='rb') as src_f:
            len_bitstream = struct.unpack('I', src_f.read(4))[0]
            with open(os.path.join(self.temp_folder_path, "flag.bin"), mode='wb') as dst_f:
                data = src_f.read(len_bitstream)
                dst_f.write(data)

            self.decode_flags()

            if self.use_DCBS:
                bitstream_list = [os.path.join(self.temp_folder_path, "shift.bin"), ]
            else:
                bitstream_list = [os.path.join(self.temp_folder_path, "odd.bin"), os.path.join(self.temp_folder_path, "even.bin")]

            for bitstream_path in bitstream_list:
                len_bitstream = struct.unpack('I', src_f.read(4))[0]
                with open(bitstream_path, mode='wb') as dst_f:
                    data = src_f.read(len_bitstream)
                    dst_f.write(data)

    def postprocess(self):
        rec_slices = []
        if self.use_DCBS:
            rec_slices_x_bit = self.read_yuv(os.path.join(self.temp_folder_path, "shift.yuv"))
            for rec_slice in rec_slices_x_bit:
                rec_slices.append(self.shift_x_bit_to_16_bit(slices=rec_slice, x=10 if self.shift_to_10_bit else 8))
        else:
            odd_slices = self.read_yuv(os.path.join(self.temp_folder_path, "odd.yuv"))
            even_slices = self.read_yuv(os.path.join(self.temp_folder_path, "even.yuv"))

            os.remove(os.path.join(self.temp_folder_path, "odd.yuv"))
            os.remove(os.path.join(self.temp_folder_path, "even.yuv"))

            for odd_slice, even_slice in zip(odd_slices, even_slices):
                rec_slice = self.inverse_bit_depth_conversion_based_on_interlacing(odd_slice=odd_slice, even_slice=even_slice)
                rec_slices.append(rec_slice.astype(np.uint16))

        self.slice_splicing(slices=rec_slices)

    def decode_medical_sequences(self):
        yuv_list = [os.path.join(self.temp_folder_path, "shift.yuv"), ] if self.use_DCBS \
            else [os.path.join(self.temp_folder_path, "odd.yuv"), os.path.join(self.temp_folder_path, "even.yuv")]

        for yuv_filepath in yuv_list:

            bin_filepath = os.path.splitext(yuv_filepath)[0] + '.bin'

            dec_cmd = "{} --input {} --output {}".format(self.args.decoder_path, bin_filepath, yuv_filepath)

            os.system(dec_cmd)
            os.remove(bin_filepath)

    def slice_splicing(self, slices: list) -> None:
        bit_depth = 16

        if self.data_type == "CT":
            rec_nii = np.stack(slices, axis=-1).transpose([1, 0, 2])
        else:
            rec_nii = np.stack(slices, axis=-1).transpose([2, 1, 0])
            rec_nii = rec_nii[:, :, :, np.newaxis]

            rec_nii = rec_nii[:, :, ::-1, :]  # FOR MRI, I found the sequence is inverse

        rec_nii = rec_nii / (2 ** bit_depth) * (self.max_ - self.min_) + self.min_
        np.save(self.args.nii_filepath, rec_nii)

    def decode_flags(self):
        flag_filepath = os.path.join(self.temp_folder_path, "flag.bin")
        with open(flag_filepath, mode='rb') as f:
            flag = struct.unpack('B', f.read(1))[0]
            self.data_type = "CT" if flag > 4 else "MRI"
            self.use_DCBS = False if flag == 4 or flag == 0 else True
            self.shift_to_10_bit = False if flag == 6 or flag == 2 else True

            self.min_ = struct.unpack('i', f.read(4))[0]
            self.max_ = struct.unpack('i', f.read(4))[0]

            # note that below flags is duplicated in the bitstream of sequences
            self.height = struct.unpack('H', f.read(2))[0]
            self.width = struct.unpack('H', f.read(2))[0]
            self.num_frames = struct.unpack('H', f.read(2))[0]

    def read_yuv(self, yuv_10bit_path: str) -> list:
        frames = []

        chroma_height = self.height // 2
        chroma_width = self.width // 2

        output_8_bit = not(self.use_DCBS and self.shift_to_10_bit)

        output_data_type = np.uint8 if output_8_bit else np.uint16

        with open(yuv_10bit_path, mode='rb') as f:
            for idx in range(self.num_frames):
                luma_array = np.zeros((self.height, self.width), dtype=output_data_type)
                for h in range(self.height):
                    for w in range(self.width):
                        data_bytes = f.read(2)
                        pixel = np.frombuffer(data_bytes, np.uint16).astype(np.uint16)
                        if output_8_bit:
                            pixel = pixel / 1024 * 256
                        luma_array[h, w] = pixel.astype(output_data_type)
                frames.append(luma_array)

                f.read(2 * chroma_height * chroma_width)  # drop U data
                f.read(2 * chroma_height * chroma_width)  # drop V data

        return frames

    @staticmethod
    def shift_x_bit_to_16_bit(slices: np.ndarray, x: int) -> np.ndarray:
        assert x == 10 or x == 8
        slices = np.left_shift(slices, 16 - x)
        return slices

    @staticmethod
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


if __name__ == "__main__":
    decompressor = Decompression()
    decompressor.decompress()