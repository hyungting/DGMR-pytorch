import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model import FourierLSTM, SpatialDiscriminator, TemporalDiscriminator
from dataset import NimrodDataset
from utilities3 import LpLoss

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def plot_test_image(arrs, true_title, pred_title, output_path=None):
    if isinstance(arrs, torch.Tensor):
        arrs = arrs.detach().cpu().numpy()

    title = [true_title, pred_title]
    batch = [1, 0]
    S = arrs.shape[1]
    fig = plt.figure(figsize=(5*S, 20))

    for b in batch:
        arr = arrs[b] # S, H, W
        for s in range(S):
            img = arr[s, :, :]

            ax = fig.add_subplot(b+1, S, s+1)
            im = ax.imshow(img, vmin=0, vmax=arrs.max(), cmap="jet")
            ax.set_title(title[b][s])

            if s == S-1 and b == 1:
                cbar_ax = fig.add_axes([0.92, 0.38, 0.01, 0.4])
                plt.colorbar(im, cbar_ax)

    if output_path: plt.savefig(output_path)
    else: return fig
    return


def plot_image(arrs, title, batch=0, output_path=None):
    if isinstance(arrs, torch.Tensor):
        arrs = arrs.detach().cpu().numpy()

    if isinstance(batch, int):
        batch = [batch]

    for b in batch:
        arr = arrs[b] # S, H, W
        S, H, W = arr.shape
        fig = plt.figure(figsize=(5*S, 5))
        for s in range(S):
            img = arr[s, :, :]

            ax = fig.add_subplot(1, S, s+1)
            im = ax.imshow(img, vmin=0, vmax=img.max(), cmap="jet")
            ax.set_title(title[s])

            if s == S-1:
                cbar_ax = fig.add_axes([0.92, 0.38, 0.01, 0.4])
                plt.colorbar(im, cbar_ax)

        if output_path: plt.savefig(output_path)
        else: return fig
    return

def plot_single_image(tensor):
    img = tensor#np.real(tensor.numpy())
    print(img.shape)
    torch.fft.irfft2(img, s=(256, 256))
    fig = plt.figure()
    return fig

def main():
    setup_seed(0)
    
    ##### DATA #####
    root = "/home/yihan/yh/research/Nimrod/"
    train_file = os.path.join(root, "Nimrod_2019.dat")
    train_csv = os.path.join(root, "log", "rain_record_2019.csv")
    train_year = 2019
    test_file = os.path.join(root, "Nimrod_2020.dat")
    test_csv = os.path.join(root, "log", "rain_record_2020.csv")
    test_year = 2020
    ################
    
    ##### PARAMS #####
    nonzeros_th = 200000
    in_step = 10
    out_step = 1
    total_out_step = 10
    img_size = 128
    dbz = True
    batch_size = 32
    #######################

    train_dataset = NimrodDataset(file=train_file, target_year=train_year, record_file=train_csv,
            nonzeros_th=nonzeros_th, in_step=in_step, out_step=total_out_step, 
            cropsize=img_size, dbz=dbz, return_time=False, data_type="train")
    
    train_loader = DataLoader(train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=32)

    test_dataset = NimrodDataset(file=test_file, target_year=test_year, record_file=test_csv,
            nonzeros_th=nonzeros_th, in_step=in_step, out_step=total_out_step,
            cropsize=img_size, dbz=dbz, return_time=True, data_type="test")

    test_loader = DataLoader(test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=32)

    print("##### Data Loaded! #####")

    ##### HYPER PARAMS #####
    LOAD = False
    ckpt_path = None
    lr = 1e-3
    weight_decay = 1e-4
    start_e = 0
    n_epoch = 100
    hidden_dim = 16
    ########################

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"##### Train with {device}! #####")
    

    ##### MODEL SETTINGS #####
    generator = FourierLSTM(in_step=in_step, out_step=total_out_step, hidden_dim=hidden_dim, img_size=img_size)
    spatial_discriminator = SpatialDiscriminator()
    temporal_discriminator = TemporalDiscriminator(total_out_step)

    random_crop = transforms.RandomCrop(128)
    #########################

    ##### LOSS SETTINGS #####
    generator_loss = nn.L1Loss()
    spatial_loss = nn.BCELoss()
    temporal_loss = nn.BCELoss()
    #########################

    ##### OPTIMIZER SETTINGS #####
    optim_G = torch.optim.Adam(generator.parameters(), lr=lr, weight_decay=weight_decay)
    optim_SD = torch.optim.Adam(spatial_discriminator.parameters(), lr=lr, weight_decay=weight_decay)
    optim_TD = torch.optim.Adam(temporal_discriminator.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optim_G, 10)
    scheduler_SD = torch.optim.lr_scheduler.CosineAnnealingLR(optim_SD, 10)
    scheduler_TD = torch.optim.lr_scheduler.CosineAnnealingLR(optim_TD, 10)
    ##############################
    
    generator.to(device)
    spatial_discriminator.to(device)
    temporal_discriminator.to(device)

    print("##### Model loaded! #####")
    writer = SummaryWriter(f"runs/Fourier-1220-LSTM")

    if LOAD:
        ckpt = torch.load(f"./ckpt/{ckpt_path}")
        start_e = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        print("##### Checkpoint loaded! #####")

    #print(generator)
    #print(spatial_discriminator)
    #print(temporal_discriminator)
    
    print(f"##### Start training! #####")
    spatial_discriminator.train()
    temporal_discriminator.train()
    
    ##### REQUIRED VARIABLES #####
    min_loss = np.inf
    true_spatial_validity = torch.ones((1, 1)).to(device)
    false_spatial_validity = torch.zeros((1, 1)).to(device)
    true_temporal_validity = torch.ones((1, 1)).to(device)
    false_temporal_validity = torch.zeros((1, 1)).to(device)

    for e in range(start_e, start_e+n_epoch):
        TD_losses, SD_losses = [], []
        val_losses = []
        for i, img in enumerate(train_loader):
            b = img.shape[0]
            if b != batch_size: continue
            img_x = img[:, 0:in_step, ...].to(device)
            img_y = img[:, in_step:, ...].to(device)            
            
            ##### GENERATE FAKE DATA #####
            generator.eval()
            G_pred = generator(img_x)
            
            ##### TRAIN SPATIAL D #####
            random_idx = np.random.randint(low=0, high=total_out_step, size=8)
            
            spatial_discriminator.train()
            spatial_discriminator.zero_grad()
            real_spatial_validity = spatial_discriminator(img_y[:, random_idx, ...])
            fake_spatial_validity = spatial_discriminator(G_pred[:, random_idx, ...].detach())

            real_loss_SD = spatial_loss(real_spatial_validity, true_spatial_validity.repeat(b, 1))
            fake_loss_SD = spatial_loss(real_spatial_validity, false_spatial_validity.repeat(b, 1))
            loss_SD = (real_loss_SD + fake_loss_SD) / 2
            loss_SD.backward()
            optim_SD.step()
            SD_losses.append(loss_SD.item())
            
            ##### TRAIN TEMPORAL D #####
            temporal_discriminator.train()
            temporal_discriminator.zero_grad()
            real_temporal_validity = temporal_discriminator(random_crop(img_y))
            fake_temporal_validity = temporal_discriminator(random_crop(G_pred).detach())

            real_loss_TD = temporal_loss(real_temporal_validity, true_temporal_validity.repeat(b, 1))
            fake_loss_TD = temporal_loss(fake_temporal_validity, false_temporal_validity.repeat(b, 1))
            loss_TD = (real_loss_TD + fake_loss_TD) / 2
            loss_TD.backward()
            optim_TD.step()
            TD_losses.append(loss_TD.item())
            
            ##### TRAIN G #####
            generator.train()
            G_losses = []
            for _ in range(3):
                optim_G.zero_grad()
                G_pred = generator(img_x)

                loss = generator_loss(G_pred, img_y)
                G_losses.append(loss.item())
                
                G_temporal_validity = temporal_discriminator(random_crop(G_pred).detach())
                loss_TD = temporal_loss(G_temporal_validity, true_temporal_validity.repeat(b, 1))
                
                G_spatial_validity = spatial_discriminator(G_pred[:, random_idx, ...].detach())
                loss_SD = spatial_loss(G_spatial_validity, true_spatial_validity.repeat(b, 1))

                loss = loss + 0.02 * (loss_SD.item() + loss_TD.item())
                loss.backward()
                optim_G.step()
            
        with torch.no_grad():
            generator.eval()
            for j, (img, time) in enumerate(test_loader):
                if j % 100 != 0: continue # save some time
                b = img.shape[0]
                img_x = img[:, :in_step, ...].to(device)
                img_y = img[:, in_step:, ...].to(device)

                pred = generator(img_x)

                loss = generator_loss(pred.view(b, -1), img_y.view(b, -1))
                val_loss = loss.item()
                val_losses.append(val_loss)

                if j == 0:
                    image = torch.cat((img_y[0].unsqueeze(0), G_pred[0].unsqueeze(0)), dim=0)
                    true_title = [f"{time[_][0]} (True)" for _ in range(total_out_step)]
                    pred_title = [f"{time[_][0]} (Pred)" for _ in range(total_out_step)]
                    figure = plot_test_image(image, true_title, pred_title)
                    writer.add_figure("Output", figure, e)
                    if e % 10 == 0:
                        writer.add_figure("Output_ckpt", figure, e)
        if min_loss > np.mean(val_losses):
            min_loss = np.mean(val_losses)
            state = dict(
                    epoch = e,
                    model = dict(
                        G = generator.state_dict(),
                        SD = spatial_discriminator.state_dict(),
                        TD = temporal_discriminator.state_dict()
                        ),
                    optim = dict(
                        G = optim_G.state_dict(),
                        SD = optim_SD.state_dict(),
                        TD = optim_TD.state_dict()
                        ),
                    params = dict(
                        # DATA
                        in_step = in_step,
                        total_out_step = total_out_step,
                        nonzeros_th = nonzeros_th,
                        img_size = img_size,
                        dbz = dbz,
                        batch_size = batch_size,
                        # HYPER
                        lr = lr,
                        weight_decay = weight_decay
                        )
                    )
            torch.save(state, "model_best.pth.tar")
            print(f">>>>>>>>> Model saved. Minimum generator loss: {min_loss: .4f}")

        writer.add_scalar("generator loss", np.mean(G_losses), e)
        writer.add_scalar("temporal loss", np.mean(TD_losses), e)
        writer.add_scalar("spatial loss", np.mean(SD_losses), e)
        writer.add_scalar("validation loss", np.mean(val_losses), e)

        print(f"Epoch: {e: 3d}, Generator loss: {np.mean(G_losses): .4f}, Temporal loss: {np.mean(TD_losses): .4f}, Spatial loss: {np.mean(SD_losses): .4f}, Val loss: {np.mean(val_losses): .4f}")
        torch.cuda.empty_cache()
    return

if __name__ == "__main__":
    main()
