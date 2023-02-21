import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from tqdm import tqdm

from Network import Network


class MedicalDataset(Dataset):
    def __init__(self, h5_file: str, transform=None):
        super(MedicalDataset, self).__init__()
        self.h5 = h5py.File(name=h5_file, mode='r')
        self.file_list = list(self.h5.keys())
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx) -> torch.Tensor:
        x = torch.from_numpy(np.array(self.h5[self.file_list[idx]])).permute(0, 3, 1, 2).contiguous()  # TODO: 记得在做数据的时候归一化
        return self.transform(x) if self.transform else x


if __name__ == "__main__":
    h5_filepath = "./train.h5py"
    experiments_dir = "./Experiments"

    batch_size = 4
    lr = 1e-3
    lr_decay_milestone = [100, 200, 300]
    lr_decay_factor = 0.5
    device = "cuda"
    checkpoints = None
    max_epochs = 500

    eval_epochs = 5

    experiments_dir = Path(experiments_dir)
    experiments_dir.mkdir(exist_ok=True)
    experiment_dir = Path(str(experiments_dir) + '/' + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))
    experiment_dir.mkdir(exist_ok=True)
    ckpt_dir = experiment_dir.joinpath("Checkpoints/")
    ckpt_dir.mkdir(exist_ok=True)

    train_dataloader = DataLoader(dataset=MedicalDataset(h5_file=h5_filepath, transform=RandomCrop(size=256)),
                                  batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(dataset=MedicalDataset(h5_file=h5_filepath), batch_size=1, shuffle=False)

    net = Network().to(device)
    optimizer = AdamW([{'params': net.parameters(), 'initial_lr': lr}], lr=lr)

    metric = nn.MSELoss()

    # load checkpoints
    start_epoch = 0
    best_psnr = 0
    if checkpoints:
        print("\n===========Load checkpoints {0}===========\n".format(checkpoints))
        ckpt = torch.load(checkpoints, map_location=device)
        best_psnr = ckpt['best_psnr']
        start_epoch = ckpt["epoch"] + 1
        net.load_state_dict(ckpt["network"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        print("\n===========Training from scratch===========\n")

    scheduler = MultiStepLR(optimizer, last_epoch=start_epoch - 1, milestones=lr_decay_milestone, gamma=lr_decay_factor)

    # training loop
    for epoch in range(start_epoch, max_epochs):
        print("\n===============Epoch {0}===============\n".format(epoch))

        # train
        net.train()
        for frames in tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.9, ncols=50):
            ori = frames[:, 0, :, :, :].to(device)
            rec = frames[:, 0, :, :, :].to(device)

            enhanced_rec = net(rec)

            rec_mse = metric(ori, rec)
            rec_psnr = -10 * torch.log10(rec_mse)

            enhanced_mse = metric(ori, enhanced_rec)
            enhanced_psnr = -10 * torch.log10(enhanced_mse)

            optimizer.zero_grad()
            enhanced_mse.backward()
            optimizer.step()

        scheduler.step()

        # eval
        if epoch % eval_epochs == 0:
            net.eval()
            average_rec_psnr = average_enhanced_psnr = 0.0
            for frames in tqdm(eval_dataloader, total=len(eval_dataloader), smoothing=0.9, ncols=50):
                ori = frames[:, 0, :, :, :].to(device)
                rec = frames[:, 0, :, :, :].to(device)

                with torch.no_grad():
                    enhanced_rec = net(rec)

                rec_mse = metric(ori, rec)
                rec_psnr = -10 * torch.log10(rec_mse)

                enhanced_mse = metric(ori, enhanced_rec)
                enhanced_psnr = -10 * torch.log10(enhanced_mse)

                average_rec_psnr += rec_psnr
                average_enhanced_psnr += enhanced_psnr

            average_enhanced_psnr = average_enhanced_psnr / len(eval_dataloader)
            average_rec_psnr = average_rec_psnr / len(eval_dataloader)

            if average_enhanced_psnr > best_psnr:
                best_psnr = average_enhanced_psnr

                ckpt = {
                    "network": net.state_dict(),
                    "epoch": epoch,
                    "best_psnr": best_psnr,
                    "optimizer": optimizer.state_dict()
                }

                save_path = "%s/Network_%.3d.pth" % (ckpt_dir, epoch)
                torch.save(ckpt, save_path)
                print("\n=================Save model to {}=================\n".format(save_path))
