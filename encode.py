import os
from multiprocessing import Pool


num_proc = 28

encoder_path = r"/public/xiangrliu3/hpm-HPM-13.0/bin/app_encoder"

decoder_path = r"/public/xiangrliu3/hpm-HPM-13.0/bin/app_decoder"

cfg_path = r"/public/xiangrliu3/Medical/encode_RA.cfg"

qp_list = [22, 23, 24, 25]

sequences_folder = r"/public/xiangrliu3/CT"
rec_sequences_folder = r"/public/xiangrliu3/RecCT"

os.makedirs(rec_sequences_folder, exist_ok=True)

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
            res_filepath = os.path.join(rec_sequences_folder, seq_name, str(qp), os.path.splitext(yuv_filename)[0] + '.txt')

            _, _, resolution, frames, _ = os.path.split(yuv_filepath)[-1].split('_')
            width, height = resolution.split('x')

            enc_cmd = "{} --config {} --input {} --output {} --width {} --height {} --frames {} --op_qp {} > {}".format(
                encoder_path, cfg_path, yuv_filepath, bin_filepath, width, height, frames, qp, res_filepath
            )

            dec_cmd = "{} --input {} --output".format(decoder_path, bin_filepath, rec_filepath)

            enc_cmd_list.append(enc_cmd)
            dec_cmd_list.append(dec_cmd)


p = Pool(processes=num_proc)
for cmd in enc_cmd_list:
    p.apply_async(os.system, args=(cmd,))
p.close()
p.join()

p = Pool(processes=num_proc)
for cmd in dec_cmd_list:
    p.apply_async(os.system, args=(cmd,))
p.close()
p.join()


