import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_test_image(pred, target, pred_title, true_title, output_path=None):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach()
    if isinstance(target, torch.Tensor):
        target = target.detach()

    arrs = torch.stack((target, pred), dim=0).cpu().numpy()
    title = [true_title, pred_title]
    batch = [1, 0]
    S = arrs.shape[1]
    fig = plt.figure(figsize=(3*S, 15))

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
