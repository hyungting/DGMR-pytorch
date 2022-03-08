import os
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import Generator, SpatialDiscriminator, TemporalDiscriminator
from dataset import MultiNimrodDataset, NimrodDataset
from utils.loss import Regularizer, DiscriminatorLoss
from utils.plot import plot_test_image, plot_image, plot_noise

warnings.filterwarnings('ignore')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

def main():
    setup_seed(0)
    
    ##### DATA #####
    root = "/home/yihan/yh/research/Nimrod/"
    train_year_list = [2016, 2017, 2018]

    test_year = 2019
    test_file = os.path.join(root, f"Nimrod_{test_year}.dat")
    test_csv = os.path.join(root, "log", f"rain_record_{test_year}.csv")
    ################

    ##### PARAMS #####
    nonzeros_th = 150000
    in_step = 4
    out_step = 6
    img_size = 256
    dbz = True
    batch_size = 4
    #######################

    train_dataset = MultiNimrodDataset(root, target_year_list=train_year_list,
            nonzeros_th=nonzeros_th, in_step=in_step, out_step=out_step, 
            cropsize=img_size, dbz=dbz, return_time=False, data_type="train")
    
    train_loader = DataLoader(train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=32,
            pin_memory=True)

    test_dataset = NimrodDataset(file=test_file, target_year=test_year, record_file=test_csv,
            nonzeros_th=nonzeros_th, in_step=in_step, out_step=out_step,
            cropsize=img_size, dbz=dbz, return_time=True, data_type="test")

    test_loader = DataLoader(test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=32,
            pin_memory=True)

    print("##### Data Loaded! #####")

    ##### HYPER PARAMS #####
    LOAD = False
    ckpt_path = "model_best.pth.tar"
    lr_G = 5e-4
    lr_D = 2e-3
    start_e = 0
    n_epoch = 500
    n_G = 2
    alpha = 20 # weight of regularization term
    margin = 0.5 # margin of discriminator's hinge loss
    ########################

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"##### Train with {device}! #####")
    

    ##### MODEL SETTINGS #####
    generator = Generator(in_step, out_step, debug=False)
    spatial_discriminator = SpatialDiscriminator(n_frame=6)
    temporal_discriminator = TemporalDiscriminator()
    #########################

    ##### LOSS SETTINGS #####
    generator_loss = Regularizer(alpha=alpha)
    spatial_loss = DiscriminatorLoss(margin=margin)
    temporal_loss = DiscriminatorLoss(margin=margin)
    #########################

    ##### OPTIMIZER SETTINGS #####
    optim_G = torch.optim.Adam(generator.parameters(), lr=lr_G, betas=(0, 0.999))#, weight_decay=weight_decay)
    optim_SD = torch.optim.Adam(spatial_discriminator.parameters(), lr=lr_D, betas=(0, 0.999))#, weight_decay=weight_decay)
    optim_TD = torch.optim.Adam(temporal_discriminator.parameters(), lr=lr_D, betas=(0, 0.999))#, weight_decay=weight_decay)
    
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optim_G, 10)
    scheduler_SD = torch.optim.lr_scheduler.CosineAnnealingLR(optim_SD, 10)
    scheduler_TD = torch.optim.lr_scheduler.CosineAnnealingLR(optim_TD, 10)
    ##############################
    
    generator.to(device)
    spatial_discriminator.to(device)
    temporal_discriminator.to(device)

    print("##### Model loaded! #####")
    writer = SummaryWriter(f"../experiments/DGMR-0304")

    if LOAD:
        ckpt = torch.load(ckpt_path)
        start_e = ckpt["epoch"]

        generator.load_state_dict(ckpt["model"]["G"])
        spatial_discriminator.load_state_dict(ckpt["model"]["SD"])
        temporal_discriminator.load_state_dict(ckpt["model"]["TD"])
        
        optim_G.load_state_dict(ckpt["optim"]["G"])
        optim_SD.load_state_dict(ckpt["optim"]["SD"])
        optim_TD.load_state_dict(ckpt["optim"]["TD"])
        print("##### Checkpoint loaded! #####")
        del ckpt
    
    writer.add_text("Settings", f"nonzeros: {nonzeros_th}, in steps: {in_step}, out steps: {out_step}, img size: {img_size}, batch size: {batch_size}, epoch: {n_epoch}")
    print(f"##### Start training! #####")

    for e in range(start_e, start_e+n_epoch):
        print(f"####################    EPOCH {e}    ####################")
        G_train_losses, SD_train_losses, TD_train_losses = [], [], []
        G_val_losses, SD_val_losses, TD_val_losses = [], [], []
        SD_trainG_losses, TD_trainG_losses = [], []
        
        pbar = tqdm(train_loader)
        
        for batch in pbar:
            pbar.set_description("   >>>>> TRAINING D")
            img_x, img_y = batch
            b = img_x.shape[0]
            
            img_x = img_x.to(device)
            img_y = img_y.to(device)

            for p in generator.parameters(): p.requires_grad_(False)
            for p in spatial_discriminator.parameters(): p.requires_grad_(True)
            for p in temporal_discriminator.parameters(): p.requires_grad_(False)
            ##### GENERATE FAKE DATA #####
            generator.eval()
            pred = generator(img_x).detach()

            ##### TRAIN SPATIAL D #####
            spatial_discriminator.train()
            
            real_spatial_validity = spatial_discriminator(img_y)
            fake_spatial_validity = spatial_discriminator(pred)

            real_loss_SD = spatial_loss(real_spatial_validity, True)
            fake_loss_SD = spatial_loss(fake_spatial_validity, False)
            
            loss_SD = real_loss_SD + fake_loss_SD
            SD_train_losses.append(loss_SD.item())

            optim_SD.zero_grad()
            loss_SD.backward()
            optim_SD.step()
            
            ##### TRAIN TEMPORAL D #####
            for p in spatial_discriminator.parameters(): p.requires_grad_(False)
            for p in temporal_discriminator.parameters(): p.requires_grad_(True)
            temporal_discriminator.train()
            
            real_temporal_validity = temporal_discriminator(torch.cat((img_x, img_y), dim=1))
            fake_temporal_validity = temporal_discriminator(torch.cat((img_x, pred), dim=1))
            
            real_loss_TD = temporal_loss(real_temporal_validity, True)
            fake_loss_TD = temporal_loss(fake_temporal_validity, False)

            loss_TD = real_loss_TD + fake_loss_TD
            TD_train_losses.append(loss_TD.item())

            ##### BACKWARD #####
            #loss_D = loss_SD + loss_TD
            
            optim_TD.zero_grad()
            loss_TD.backward()
            optim_TD.step()
            #scheduler_TD.step()

        torch.cuda.empty_cache()
        writer.add_text("Real spatial validity", f"{real_spatial_validity.detach().cpu().tolist()}", e)
        writer.add_text("Fake spatial validity", f"{fake_spatial_validity.detach().cpu().tolist()}", e)
        writer.add_text("Real temporal validity", f"{real_temporal_validity.detach().cpu().tolist()}", e)
        writer.add_text("Fake temporal validity", f"{fake_temporal_validity.detach().cpu().tolist()}", e)
        
        writer.add_scalar("[TRAIN D] TD loss", np.mean(TD_train_losses), e)
        writer.add_scalar("[TRAIN D] SD loss", np.mean(SD_train_losses), e)

        ##### TRAIN G #####
        if e % n_G == (n_G - 1):
            generator.train()
            spatial_discriminator.eval()
            temporal_discriminator.eval()
            for p in generator.parameters(): p.requires_grad_(True)
            for p in spatial_discriminator.parameters(): p.requires_grad_(False)
            for p in temporal_discriminator.parameters(): p.requires_grad_(False)
            
            pbar = tqdm(train_loader)
            for img in pbar:
                pbar.set_description("   >>>>> TRAINING G")
                img_x, img_y = batch
                b = img_x.shape[0]
                
                img_x = img_x.to(device)
                img_y = img_y.to(device)

                loss, loss_SD, loss_TD = 0, 0, 0

                for _ in range(6):
                    pred = generator(img_x)
                    loss += generator_loss(pred, img_y)
                    
                    temporal_validity = temporal_discriminator(torch.cat((img_x, pred.detach()), dim=1))
                    spatial_validity = spatial_discriminator(pred.detach())

                    loss_SD += (-torch.mean(spatial_validity))
                    loss_TD += (-torch.mean(temporal_validity))
                
                G_train_losses.append(loss.item()/6)
                SD_trainG_losses.append(loss_SD.item()/6)
                TD_trainG_losses.append(loss_TD.item()/6)
                
                loss = (loss + loss_SD + loss_TD) / 6
                
                optim_G.zero_grad()
                loss.backward()
                optim_G.step()
                #scheduler_G.step()
                
                torch.cuda.empty_cache()

            writer.add_scalar("[TRAIN G] G loss", np.mean(G_train_losses), e)
            writer.add_scalar("[TRAIN G] SD loss", np.mean(SD_trainG_losses), e)
            writer.add_scalar("[TRAIN G] TD loss", np.mean(TD_trainG_losses), e)

        ##### VALIDATION #####
        if e % n_G == (n_G - 1):
            generator.eval()
            spatial_discriminator.eval()
            temporal_discriminator.eval()
            with torch.no_grad():
                j = 0
                pbar = tqdm(test_loader)
                for batch in pbar:
                    pbar.set_description("   >>>>> VALIDATION")
                    img_x, img_y, time = batch
                    b = img_x.shape[0]
                    
                    img_x = img_x.to(device)
                    img_y = img_y.to(device)

                    pred = generator(img_x).detach()
                    spatial_validity = spatial_discriminator(pred)
                    temporal_validity = temporal_discriminator(torch.cat((img_x, pred), dim=1))
                        
                    loss = generator_loss(pred, img_y)
                    loss_SD = -torch.mean(spatial_validity)
                    loss_TD = -torch.mean(temporal_validity)
                    
                    G_val_losses.append(loss.item())
                    SD_val_losses.append(loss_SD.item())
                    TD_val_losses.append(loss_TD.item())

                    if j == 0 or j == 30:
                        true_title = [f"{time[in_step+_][0]} (True)" for _ in range(out_step)]
                        pred_title = [f"{time[in_step+_][0]} (Pred)" for _ in range(out_step)]
                        figure = plot_test_image(pred[0], img_y[0], pred_title, true_title)
                        writer.add_figure(f"Output: {j}", figure, e)
                    j += 1
            writer.add_scalar("[VAL] G loss", np.mean(G_val_losses), e)
            writer.add_scalar("[VAL] SD loss", np.mean(SD_val_losses), e)
            writer.add_scalar("[VAL] TD loss", np.mean(TD_val_losses), e)
        
        print(f"   >>>>> [training D] SD loss: {np.mean(SD_train_losses):.3f}, TD loss: {np.mean(TD_train_losses):.3f}")
        if e % n_G == (n_G - 1):
            print(f"   >>>>> [training G] G loss: {np.mean(G_train_losses):.3f}, SD loss: {np.mean(SD_trainG_losses):.3f}, TD loss: {np.mean(TD_trainG_losses):.3f}")
            print(f"   >>>>> [validation] G loss: {np.mean(G_val_losses):.3f}, SD loss: {np.mean(SD_val_losses):.3f}, TD loss: {np.mean(TD_val_losses):.3f}")
        

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
                        out_step = out_step,
                        nonzeros_th = nonzeros_th,
                        img_size = img_size,
                        dbz = dbz,
                        batch_size = batch_size,
                        )
                    )
        torch.save(state, "model_best.pth.tar")

        torch.cuda.empty_cache()
    return

if __name__ == "__main__":
    main()
