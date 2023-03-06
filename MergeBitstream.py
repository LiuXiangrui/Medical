import struct
import os

data_type = "MRI"
qp_list = [22, 26, 30, 34]
results_folder = r"D:\AVS3_MRI_Anchor_Results\MRI_Test_Results_AVS3"
bin_folder = r"D:\AVS3_MRI_Anchor_Results\Bitstreams"
def encode_flags(bin_path: str, min_: int, max_: int, height: int, width: int, num_frames: int) -> None:
    """
    encode flags to bitstream and return the bits-per-voxel
    """

    data_type_ct = True if data_type == "CT" else False

    with open(bin_path, mode='wb') as f:
        """
        |  Data Type | Depth Conversion Method | Shift-to-10bit (if use DCBS)|
        | CT=1, MRI=0|      DCBS=1, else=0     |       10bit=1, 8bit=0       |
        """
        flag = 4 * int(data_type_ct) + 2 * int(False) + int(False)
        f.write(struct.pack('B',
                            flag))  # 8-bit  # note that I can't fine a way to write 3 bits in Python, so just use 8 bit -_-|||

        f.write(struct.pack('i', min_))  # 32-bit
        f.write(struct.pack('i', max_))  # 32-bit

        # note that below flags is duplicated in the bitstream of sequences
        f.write(struct.pack('H', height))  # 16-bit
        f.write(struct.pack('H', width))  # 16-bit
        f.write(struct.pack('H', num_frames))  # 16-bit


os.makedirs(bin_folder, exist_ok=True)

for qp in qp_list:
    for nii_name in os.listdir(results_folder):
        os.makedirs(os.path.join(bin_folder, str(qp)), exist_ok=True)
        bin_filepath = os.path.join(bin_folder, str(qp), nii_name + ".bin")
        even_filepath = odd_filepath = None

        seq_folder = os.path.join(results_folder, nii_name, str(qp))
        for file in os.listdir(seq_folder):
            if os.path.splitext(file)[-1] != '.bin':
                continue
            if "_even.bin" in file:
                even_filepath = os.path.join(seq_folder, file)
            if "_odd.bin" in file:
                odd_filepath = os.path.join(seq_folder, file)
            info = file.split('_')
            min_ = int(info[-3])
            max_ = int(info[-2])
            num_frames = int(info[-4])
            resolution = info[-5]
            width, height = resolution.split('x')
            width = int(width)
            height = int(height)

        encode_flags(os.path.join(bin_folder, "temp.bin"), height=height, width=width,
                     num_frames=num_frames, min_=min_, max_=max_)

        bitstream_list = [
            os.path.join(bin_folder, "temp.bin"),
            odd_filepath, even_filepath
        ]


        with open(bin_filepath, mode='wb') as dst_f:
            for bitstream_path in bitstream_list:
                len_bitstream = os.path.getsize(bitstream_path)  # length of bitstream in bytes
                dst_f.write(struct.pack('I', len_bitstream))
                with open(bitstream_path, mode='rb') as src_f:
                    data = src_f.read()
                    dst_f.write(data)
        os.remove(os.path.join(bin_folder, "temp.bin"))