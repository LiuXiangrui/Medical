import os
from multiprocessing import Pool
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--qp", nargs='+')

qp_list = [int(item) for item in parser.parse_args().qp]


num_proc = 28

encoder_path = r"/public/xiangrliu3/hpm-HPM-13.0/bin/app_encoder"

decoder_path = r"/public/xiangrliu3/hpm-HPM-13.0/bin/app_decoder"

cfg_path = r"/public/xiangrliu3/Medical/encode_RA.cfg"



sequences_folder = r"/public/xiangrliu3/CTSequences"
rec_sequences_folder = r"/public/xiangrliu3/RecCTSequences"

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

            _, _, resolution, frames, _, _, _ = os.path.split(yuv_filepath)[-1].split('_')
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


