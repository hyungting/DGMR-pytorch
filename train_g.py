import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model import Generator
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

def plot_noise(noise_list,  output_path=None):
    L = len(noise_list)
    
    fig = plt.figure(figsize=(5*L, 5))
    for step in range(L):
        noise = noise_list[step] # Bi=0, C, H, W
        C, H, W = noise.shape # 768, 8, 8
        
        all_img = []
        for r in range(H):
            row = []
            for c in range(W):
                img = noise[:, r, c].reshape(24, 32)
                row.append(img)
            all_img.append(row)
        
        ax = fig.add_subplot(1, L, step+1)
        ax.axis('off')
        im = ax.imshow(np.array(all_img).reshape(24*H, 32*W), vmin=noise.min(), vmax=noise.max(), cmap="jet")
        
        mean, std = np.mean(noise), np.std(noise)
        ax.set_title(f"timestep = {step+1}\n mean = {mean:.3f}, std = {std:.3f}")

        if step == L-1:
            cbar_ax = fig.add_axes([0.92, 0.38, 0.01, 0.4])
            plt.colorbar(im, cbar_ax)

    if output_path is not None: plt.savefig(output_path)
    else: return fig
    
    return

#if __name__ == "__main__":
#    x = [torch.randn(768, 8, 8).numpy() for i in range(18)]
#    plot_noise_allstep(x)

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
    in_step = 4
    out_step = 18
    img_size = 256
    dbz = True
    batch_size = 8
    #######################

    train_dataset = NimrodDataset(file=train_file, target_year=train_year, record_file=train_csv,
            nonzeros_th=nonzeros_th, in_step=in_step, out_step=out_step, 
            cropsize=img_size, dbz=dbz, return_time=False, data_type="train")
    
    train_loader = DataLoader(train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=32)

    test_dataset = NimrodDataset(file=test_file, target_year=test_year, record_file=test_csv,
            nonzeros_th=nonzeros_th, in_step=in_step, out_step=out_step,
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
    weight_decay = 1e-3
    start_e = 0
    n_epoch = 100
    ########################

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"##### Train with {device}! #####") 

    ##### MODEL SETTINGS #####
    generator = Generator(in_step, out_step, debug=False)
    
    generator_loss = nn.L1Loss()
    
    optim_G = torch.optim.Adam(generator.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optim_G, 10)
    
    generator.to(device)

    print("##### Model loaded! #####")
    writer = SummaryWriter(f"runs/DGMR-Generator-only")

    if LOAD:
        ckpt = torch.load(ckpt_path)
        start_e = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        print("##### Checkpoint loaded! #####")

    #print(generator) 
    print(f"##### Start training! #####")
    
    min_loss = np.inf
    
    for e in range(start_e, start_e+n_epoch):
        train_losses = []
        val_losses = []
        
        generator.train()
        for i, img in enumerate(train_loader):
            b = img.shape[0]
            img_x = img[:, 0:in_step, ...].to(device)
            img_y = img[:, in_step:, ...].to(device)            
            
            optim_G.zero_grad()
            pred = generator(img_x)

            loss = generator_loss(pred, img_y)
            train_losses.append(loss.item())
                
            loss.backward()
            optim_G.step()
            
        with torch.no_grad():
            generator.eval()
            for j, (img, time) in enumerate(test_loader):
                if j % 10 != 0: continue # save some time
                b = img.shape[0]
                img_x = img[:, :in_step, ...].to(device)
                img_y = img[:, in_step:, ...].to(device)

                if j == 0:
                    pred, noise = generator(img_x, True)
                    fig = plot_noise(noise)
                    writer.add_figure(f"Noise of {in_step} time steps, 768 channels", fig, e)
                else:
                    pred = generator(img_x)

                loss = generator_loss(pred, img_y)
                val_losses.append(loss.item())

                if j == 0: # time = (total_step, batch)
                    image = torch.cat((img_y[0].unsqueeze(0), pred[0].unsqueeze(0)), dim=0)
                    true_title = [f"{time[in_step+_][0]} (True)" for _ in range(out_step)]
                    pred_title = [f"{time[in_step+_][0]} (Pred)" for _ in range(out_step)]
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
                        ),
                    optim = dict(
                        G = optim_G.state_dict(),
                        ),
                    params = dict(
                        # DATA
                        in_step = in_step,
                        out_step = out_step,
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

        writer.add_scalar("generator loss", np.mean(train_losses), e)
        writer.add_scalar("validation loss", np.mean(val_losses), e)

        print(f"Epoch: {e: 3d}, Generator loss: {np.mean(train_losses): .4f}, Val loss: {np.mean(val_losses): .4f}")

    return

if __name__ == "__main__":
    main()
