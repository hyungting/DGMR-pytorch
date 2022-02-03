import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model import FourierNet
from dataset import NimrodDataset
from utilities3 import LpLoss

def plot_test_image(arrs, true_title, pred_title, output_path=None):
    if isinstance(arrs, torch.Tensor):
        arrs = arrs.detach().cpu().numpy()

    title = [true_title, pred_title]
    batch = [1, 0]
    S = arrs.shape[1]
    fig = plt.figure(figsize=(5*S, 20))

    for b in batch:
        arr = arrs[b] # S, H, W
        print(arr.shape)
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

def main(ckpt_path,year=2020, th=200000, total_out=10, size=256):
    
    ##### DATA PARAMS #####
    root = "/home/yihan/yh/research/Nimrod/"
    test_file = os.path.join(root, f"Nimrod_{year}.dat")
    test_csv = os.path.join(root, "log", f"rain_record_{year}.csv")
    test_year = year
    nonzeros_th = th #200000
    in_step = 10
    out_step = 1
    total_out_step = total_out
    centercrop = size
    dbz = True
    return_time = False
    batch_size = 128
    #######################

    test_dataset = NimrodDataset(test_file, test_year, test_csv,
            nonzeros_th, in_step, total_out_step,
            centercrop, dbz, return_time)

    test_loader = DataLoader(test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=32)

    print("##### Data Loaded! #####")

    ##### HYPER PARAMS #####
    #ckpt_path = "./model_best.pth.tar"
    size = centercrop
    ########################

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"##### Test with {device}! #####")
    
    model = FourierNet(in_step, out_step, size)
    print("##### Model loaded! #####")
    
    writer = SummaryWriter(f"runs/Fourier-test")

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model"])
    print("##### Checkpoint loaded! #####")

    model.to(device)
    print(model)
    print(f"##### Start testing! #####")

    model.eval()
    
    with torch.no_grad():
        for j, (img_x, img_y) in enumerate(test_loader):
            b = img_x.shape[0]
            img_x = img_x.to(device)
            img_y = img_y.to(device)
            
            pred = model(img_x)
            total_pred = pred
            for t in range(1, total_out_step):
                img_in =  torch.cat((img_x[:, t:, ...], total_pred), dim=1)
                pred = model(img_in)
                total_pred = torch.cat((total_pred, pred), dim=1)
            
            if j % 10 == 0:
                img = torch.cat((img_y[0].unsqueeze(0), total_pred[0].unsqueeze(0)), dim=0)
                print(img.shape)
                true_title = [f"step_{i+1} (True)" for i in range(total_out_step)]
                pred_title = [f"step_{i+1} (Pred)" for i in range(total_out_step)]
                figure = plot_test_image(img, true_title, pred_title)

                writer.add_figure(f"Output {size}*{size}, id: {j}", figure)

    return

if __name__ == "__main__":
    main(sys.argv[1])
