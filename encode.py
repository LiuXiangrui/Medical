import os
from multiprocessing import Pool


encoder_path = r"C:\Users\XiangruiLiu\Desktop\avs_enc.exe"

decoder_path = r"C:\Users\XiangruiLiu\Desktop\avs_dec.exe"

cfg_path = r"C:\Users\xiangrliu3\Desktop\Medical\encode_RA.cfg"

qp_list = [22, 23, 24, 25]

sequences_folder = r"D:\CTSample"
rec_sequences_folder = r"D:\RecCTSample"

enc_cmd_list = []
dec_cmd_list = []

for seq_name in os.listdir(sequences_folder):
    os.makedirs(os.path.join(rec_sequences_folder, seq_name), exist_ok=True)
    for qp in qp_list:
        os.makedirs(os.path.join(rec_sequences_folder, seq_name, str(qp)), exist_ok=True)
        for yuv_filename in os.listdir(os.path.join(sequences_folder, seq_name)):
            yuv_filepath = os.path.join(sequences_folder, seq_name, yuv_filename)
            bin_filepath = os.path.join(rec_sequences_folder, seq_name, str(qp), os.path.splitext(yuv_filename)[0] + '.bin')
            rec_filepath = os.path.join(rec_sequences_folder, seq_name, str(qp), yuv_filename)

            _, _, resolution, frames, _ = os.path.split(yuv_filepath)[-1].split('_')
            width, height = resolution.split('x')

            enc_cmd = "{} --config {} --input {} --output {} --width {} --height {} --frames {} --op_qp {}".format(
                encoder_path, cfg_path, yuv_filepath, bin_filepath, width, height, frames, qp
            )

            dec_cmd = "{} --input {} --output".format(decoder_path, bin_filepath, rec_filepath)

            enc_cmd_list.append(enc_cmd)
            dec_cmd_list.append(dec_cmd)


p = Pool(processes=8)
for cmd in enc_cmd_list:
    p.apply_async(os.system, args=(cmd,))
p.close()
p.join()

p = Pool(processes=8)
for cmd in dec_cmd_list:
    p.apply_async(os.system, args=(cmd,))
p.close()
p.join()


