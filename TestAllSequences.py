import os
from multiprocessing import Pool
import numpy as np
import math
import nibabel as nib
import time
import json

num_proc = 16
data_type = "MRI"


encoder_path = r"C:\Users\XiangruiLiu\Desktop\hpm-HPM-13.0\build\x86_windows\x64\Release\encoder_app.exe"
decoder_path = r"C:\Users\XiangruiLiu\Desktop\hpm-HPM-13.0\build\x86_windows\x64\Release\decoder_app.exe"

cfg_path = r"C:\Users\XiangruiLiu\Desktop\hpm-HPM-13.0\cfg\encode_RA.cfg"

qp = 34
nii_file_folder = r"C:\Users\XiangruiLiu\Desktop\test"

bin_file_folder = r"C:\Users\XiangruiLiu\Desktop\bin"
rec_file_folder = r"C:\Users\XiangruiLiu\Desktop\rec"
res_file_folder = r"C:\Users\XiangruiLiu\Desktop\res"


def calculate_psnr_and_bpp(nii_filepath: str, rec_filepath: str, bin_filepath: str):
    nii_file = nib.load(nii_filepath)
    origin_data = np.array(nii_file.get_fdata(dtype=np.float64))
    recon_data = np.load(rec_filepath)

    # from PIL import Image
    # a = origin_data[:, :, 100, 0]
    # a = (a - a.min()) / (a.max() - a.min()) * 255
    # a = a.astype(np.uint8)
    # a = Image.fromarray(a)
    # a.show()
    #
    # b = recon_data[:, :, 100, 0]
    # b = (b - b.min()) / (b.max() - b.min()) * 255
    # b = b.astype(np.uint8)
    # b = Image.fromarray(b)
    # b.show()
    # input()

    bits = os.path.getsize(bin_filepath) * 8

    voxels = 1
    for axis in range(len(recon_data.shape)):
        voxels = voxels * recon_data.shape[axis]

    bpv = bits / voxels

    mse = np.mean((origin_data - recon_data) ** 2)
    psnr = 10 * math.log10(65535 * 65535 / mse)

    return psnr, bpv


def call_cmd(nii_filepath, bin_filepath, rec_filepath, res_filepath):
    enc_start_time = time.time()

    cmd = r"python C:\Users\XiangruiLiu\Desktop\Medical\Compress.py --data_type {} --encoder_path {}" \
          " --cfg_path {} --qp {} --nii_filepath {} --bin_filepath {}".format(data_type, encoder_path, cfg_path, qp,
                                                                              nii_filepath, bin_filepath)
    # os.system(cmd)
    enc_end_time = time.time()
    dec_start_time = time.time()
    cmd = r"python C:\Users\XiangruiLiu\Desktop\Medical\Decompress.py --decoder_path {}" \
          " --nii_filepath {} --bin_filepath {}".format(decoder_path, rec_filepath, bin_filepath)

    # os.system(cmd)
    dec_end_time = time.time()

    psnr, bpv = calculate_psnr_and_bpp(nii_filepath=nii_filepath, bin_filepath=bin_filepath, rec_filepath=rec_filepath)
    enc_time = enc_end_time - enc_start_time
    dec_time = dec_end_time - dec_start_time

    record = {"PSNR": psnr, "BPV": bpv, "ENC_TIME": enc_time, "DEC_TIME": dec_time}

    with open(res_filepath, mode='w') as f:
        json.dump(record, f)


if __name__ == "__main__":
    os.makedirs(bin_file_folder, exist_ok=True)
    os.makedirs(rec_file_folder, exist_ok=True)
    os.makedirs(res_file_folder, exist_ok=True)

    p = Pool(processes=num_proc)

    for nii_file in os.listdir(rec_file_folder):
        nii_filepath = os.path.join(nii_file_folder, os.path.splitext(nii_file)[0])
        bin_filepath = os.path.join(bin_file_folder, os.path.splitext(os.path.splitext(nii_file)[0])[0] + '.bin')
        rec_filepath = os.path.join(rec_file_folder, nii_file)
        res_filepath = os.path.join(res_file_folder, os.path.splitext(os.path.splitext(nii_file)[0])[0] + '.json')

        call_cmd(nii_filepath, bin_filepath, rec_filepath, res_filepath)
        # p.apply_async(call_cmd, args=(nii_filepath, bin_filepath, rec_filepath))
        break
    p.close()
    p.join()