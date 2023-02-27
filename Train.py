import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from Network import Network

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class MedicalDataset(Dataset):
    def __init__(self, h5_file: str, transform=None):
        super(MedicalDataset, self).__init__()
        self.h5 = h5py.File(name=h5_file, mode='r')
        self.file_list = list(self.h5.keys())
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx) -> torch.Tensor:
        x = torch.from_numpy(np.array(self.h5[self.file_list[idx]]).astype(np.uint8)).contiguous() / 255.0
        return self.transform(x) if self.transform else x


if __name__ == "__main__":
    h5_filepath = r"C:\Users\xiangrliu3\Desktop\QP22.h5py"

    experiments_dir = r"C:\Users\xiangrliu3\Desktop\ExperimentsQP22"

    # checkpoints = r"C:\Users\xiangrliu3\Desktop\ExperimentsQP22\2023-02-24_18-10\Checkpoints\Network_039.pth"
    checkpoints = None

    batch_size = 8
    lr = 2e-4
    lr_decay_milestone = [50, 200, 300, 400]
    lr_decay_factor = 0.5
    device = "cuda"

    max_epochs = 500
    eval_epochs = 1

    experiments_dir = Path(experiments_dir)
    experiments_dir.mkdir(exist_ok=True)
    experiment_dir = Path(str(experiments_dir) + '/' + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))
    experiment_dir.mkdir(exist_ok=True)
    ckpt_dir = experiment_dir.joinpath("Checkpoints/")
    ckpt_dir.mkdir(exist_ok=True)
    tb_dir = experiment_dir.joinpath('Tensorboard/')
    tb_dir.mkdir(exist_ok=True)
    tensorboard = SummaryWriter(log_dir=str(tb_dir), flush_secs=30)

    train_dataloader = DataLoader(
        dataset=MedicalDataset(h5_file=h5_filepath, transform=transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.4), transforms.RandomVerticalFlip(p=0.4)])),
        batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(dataset=MedicalDataset(h5_file=h5_filepath), batch_size=batch_size, shuffle=False)

    net = Network().to(device)
    optimizer = Adam([{'params': net.parameters(), 'initial_lr': lr}], lr=lr)

    metric = nn.MSELoss().to(device)

    # load checkpoints
    start_epoch = 0
    best_psnr = 0
    if checkpoints:
        print("\n===========Load checkpoints {0}===========\n".format(checkpoints))
        # noinspection PyTypeChecker
        ckpt = torch.load(checkpoints, map_location=device)
        best_psnr = ckpt['best_psnr']
        start_epoch = ckpt["epoch"] + 1
        net.load_state_dict(ckpt["network"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        print("\n===========Training from scratch===========\n")

    scheduler = MultiStepLR(optimizer, last_epoch=start_epoch - 1, milestones=lr_decay_milestone, gamma=lr_decay_factor)

    train_steps = eval_steps = 0
    # training loop
    for epoch in range(start_epoch, max_epochs):
        print("\n===============Epoch {0}===============\n".format(epoch))

        # train
        net.train()
        for frames in tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.9, ncols=50):
            ori = frames[:, :1, :, :].to(device)
            rec = frames[:, 1:, :, :].to(device)

            enhanced_rec = net(rec)

            rec_mse = metric(ori, rec)
            rec_psnr = -10 * torch.log10(rec_mse)

            enhanced_mse = metric(ori, enhanced_rec)
            enhanced_psnr = -10 * torch.log10(enhanced_mse)

            optimizer.zero_grad()
            enhanced_mse.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

            tensorboard.add_scalars(main_tag="Training/PSNR", global_step=train_steps,
                                    tag_scalar_dict={"Reconstruction": rec_psnr, "Enhancement": enhanced_psnr})
            tensorboard.add_scalars(main_tag="Training/MSE", global_step=train_steps,
                                    tag_scalar_dict={"Reconstruction": rec_mse, "Enhancement": enhanced_mse})
            train_steps += 1

        scheduler.step()

        # eval
        if epoch % eval_epochs == 0:
            net.eval()
            average_rec_psnr = average_enhanced_psnr = 0.0
            for frames in tqdm(eval_dataloader, total=len(eval_dataloader), smoothing=0.9, ncols=50):
                ori = frames[:, :1, :, :].to(device)
                rec = frames[:, 1:, :, :].to(device)

                with torch.no_grad():
                    enhanced_rec = net(rec)

                rec_mse = metric(ori, rec)
                rec_psnr = -10 * torch.log10(rec_mse)

                enhanced_mse = metric(ori, enhanced_rec)
                enhanced_psnr = -10 * torch.log10(enhanced_mse)

                average_rec_psnr += rec_psnr
                average_enhanced_psnr += enhanced_psnr

                eval_steps += 1

                if eval_steps % 100 == 0:
                    tensorboard.add_images(tag="Test/Origin", global_step=eval_steps,
                                           img_tensor=ori[0:4, :, :, :].clone().detach().cpu())
                    tensorboard.add_images(tag="Test/Reconstruction", global_step=eval_steps,
                                           img_tensor=rec[0:4, :, :, :].clone().detach().cpu())
                    tensorboard.add_images(tag="Test/Enhancement", global_step=eval_steps,
                                           img_tensor=enhanced_rec[0:4, :, :, :].clone().detach().cpu())

            average_enhanced_psnr = average_enhanced_psnr / len(eval_dataloader)
            average_rec_psnr = average_rec_psnr / len(eval_dataloader)

            print("Epoch {}:  Reconstruction PSNR = {}, Enhancement PSNR = {}, Delta PSNR = {}".format(
                epoch, average_rec_psnr, average_enhanced_psnr, average_enhanced_psnr - average_rec_psnr
            ))

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
