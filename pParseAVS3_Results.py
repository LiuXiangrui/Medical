import os

results_folder = './'

bpp_per_qp = {22: 0, 23: 0, 24: 0, 25: 0}


def parse_results(results_path: str):
    with open(results_path, mode='r') as f:
        data = f.readlines()
    bits = 0
    num_frames = 0
    width = height = 0
    for d in data:
        if "Total bits(bits)" in d:
            bits = int(d.split()[-1])
        if "frames  " in d:
            num_frames = int(d.split()[-1])
        if "width  " in d:
            width = int(d.split()[-1])
        if "height  " in d:
            height = int(d.split()[-1])

        bpp = bits / (num_frames * height * width)
    return bpp


num_seqs = len(list(os.listdir(results_folder)))

for seq in os.listdir(results_folder):
    for qp in bpp_per_qp.keys():
        for file in os.listdir(os.path.join(results_folder, seq, str(qp))):
            if os.path.splitext(file)[-1] != '.txt':
                continue
            else:
                bpp_per_qp[qp] += parse_results(os.path.join(results_folder, seq, str(qp), file)) / num_seqs

for qp, bpp in bpp_per_qp.items():
    print("QP: {}, BPP = {}".format(qp, bpp))
