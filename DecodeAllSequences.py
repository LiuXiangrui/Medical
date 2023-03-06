import os
from multiprocessing import Pool
import numpy as np
import math
import nibabel as nib
import time
import json

num_proc = 16

encoder_path = r"C:\Users\xiangrliu3\Desktop\avs_enc.exe"
decoder_path = r"C:\Users\xiangrliu3\Desktop\avs_dec.exe"
# cfg_path = r"C:\Users\XiangruiLiu\Desktop\hpm-HPM-13.0\cfg\encode_RA.cfg"

# For CT
# data_type = "CT"

# qp_list = [16, 20, 25, 29]
# nii_file_folder = r"D:\MedicalImageDataset\Mosmed_COVID-19_CT\CT-3"
# bin_file_folder = r"D:\AVS3_CT_Anchor_Results\Bitstreams"
# rec_file_folder = r"D:\AVS3_CT_Anchor_Results\Recon"
# res_file_folder = r"D:\AVS3_CT_Anchor_Results\Results"

# For MRI
data_type = "MRI"

qp_list = [22, 26, 30, 34]
nii_file_folder = r"D:\MedicalImageDataset\TRABIT2019_MRI\test"

bin_file_folder = r"D:\AVS3_MRI_Anchor_Results\Bitstreams"
rec_file_folder = r"D:\AVS3_MRI_Anchor_Results\Recon"
res_file_folder = r"D:\AVS3_MRI_Anchor_Results\Results"


def calculate_psnr_and_bpp(nii_filepath: str, rec_filepath: str, bin_filepath: str):
    nii_file = nib.load(nii_filepath)
    origin_data = np.array(nii_file.get_fdata(dtype=np.float64))
    recon_data = np.load(rec_filepath)

    bits = os.path.getsize(bin_filepath) * 8

    voxels = 1
    for axis in range(len(recon_data.shape)):
        voxels = voxels * recon_data.shape[axis]

    bpv = bits / voxels

    mse = np.mean((origin_data - recon_data) ** 2)
    psnr = 10 * math.log10(65535 * 65535 / mse)

    return psnr, bpv


def call_cmd(nii_filepath, bin_filepath, rec_filepath, res_filepath):
    # enc_start_time = time.time()
    #
    # cmd = r"python C:\Users\XiangruiLiu\Desktop\Medical\Compress.py --data_type {} --encoder_path {}" \
    #       " --cfg_path {} --qp {} --nii_filepath {} --bin_filepath {}".format(data_type, encoder_path, cfg_path, qp,
    #                                                                           nii_filepath, bin_filepath)
    # # os.system(cmd)
    # enc_end_time = time.time()
    dec_start_time = time.time()
    cmd = r"python C:\Users\xiangrliu3\Desktop\Medical\Decompress.py --decoder_path {}" \
          " --nii_filepath {} --bin_filepath {}".format(decoder_path, rec_filepath, bin_filepath)

    os.system(cmd)
    dec_end_time = time.time()

    psnr, bpv = calculate_psnr_and_bpp(nii_filepath=nii_filepath, bin_filepath=bin_filepath, rec_filepath=rec_filepath)
    # enc_time = enc_end_time - enc_start_time
    enc_time = 0
    dec_time = dec_end_time - dec_start_time

    record = {"PSNR": psnr, "BPV": bpv, "ENC_TIME": enc_time, "DEC_TIME": dec_time}

    with open(res_filepath, mode='w') as f:
        json.dump(record, f)


if __name__ == "__main__":
    os.makedirs(bin_file_folder, exist_ok=True)
    os.makedirs(rec_file_folder, exist_ok=True)
    os.makedirs(res_file_folder, exist_ok=True)

    p = Pool(processes=num_proc)

    for qp in qp_list:
        os.makedirs(os.path.join(rec_file_folder, str(qp)), exist_ok=True)
        os.makedirs(os.path.join(res_file_folder, str(qp)), exist_ok=True)

        for bin_filename in os.listdir(os.path.join(bin_file_folder, str(qp))):
            nii_filepath = os.path.join(nii_file_folder, os.path.splitext(bin_filename)[0]+'.nii')
            bin_filepath = os.path.join(bin_file_folder, str(qp), bin_filename)
            rec_filepath = os.path.join(rec_file_folder, str(qp), os.path.splitext(bin_filename)[0]+'.nii.gz')
            res_filepath = os.path.join(res_file_folder, str(qp), os.path.splitext(bin_filename)[0]+'.json')

            p.apply_async(call_cmd, args=(nii_filepath, bin_filepath, rec_filepath, res_filepath))

    p.close()
    p.join()

    for qp in qp_list:
        avg_bpv = avg_psnr = avg_dec_time = 0

        for res_filepath in os.listdir(os.path.join(res_file_folder, str(qp))):
            with open(os.path.join(res_file_folder, str(qp), res_filepath), mode='r') as f:
                record = json.load(f)
                avg_bpv += record["BPV"]
                avg_psnr += record["PSNR"]
                avg_dec_time += record["DEC_TIME"]
        avg_bpv /= len(list(os.listdir(os.path.join(res_file_folder, str(qp)))))
        avg_psnr /= len(list(os.listdir(os.path.join(res_file_folder, str(qp)))))
        avg_dec_time /= len(list(os.listdir(os.path.join(res_file_folder, str(qp)))))
        print("QP: {}, BPV={}, PSNR={}, DEC_TIME={}".format(qp, avg_bpv, avg_psnr, avg_dec_time))