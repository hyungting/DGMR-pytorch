import os
import warnings
import random
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model import Generator, SpatialDiscriminator, TemporalDiscriminator
from dataset import MultiNimrodDataset, NimrodDataset
from utils import Regularizer
from plot import plot_test_image, plot_image, plot_noise

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
    #train_file = os.path.join(root, "Nimrod_2019.dat")
    #train_csv = os.path.join(root, "log", "rain_record_2019.csv")
    train_year_list = [2016, 2017, 2018]

    test_year = 2019
    test_file = os.path.join(root, f"Nimrod_{test_year}.dat")
    test_csv = os.path.join(root, "log", f"rain_record_{test_year}.csv")
    ################
    

    ##### PARAMS #####
    nonzeros_th = 150000
    in_step = 4
    out_step = 12
    img_size = 256
    dbz = True
    batch_size = 8
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
    lr_G = 5e-5
    lr_D = 2e-4
    weight_decay = 1e-4
    start_e = 0
    n_epoch = 100
    ########################

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"##### Train with {device}! #####")
    

    ##### MODEL SETTINGS #####
    generator = Generator(in_step, out_step, debug=False)
    spatial_discriminator = SpatialDiscriminator()
    temporal_discriminator = TemporalDiscriminator()
    #########################

    ##### LOSS SETTINGS #####
    generator_loss = Regularizer
    spatial_loss = nn.HingeEmbeddingLoss()
    temporal_loss = nn.HingeEmbeddingLoss()
    #########################

    ##### OPTIMIZER SETTINGS #####
    optim_G = torch.optim.Adam(generator.parameters(), lr=lr_G, betas=(0, 0.999))
    optim_SD = torch.optim.Adam(spatial_discriminator.parameters(), lr=lr_D, betas=(0, 0.999))
    optim_TD = torch.optim.Adam(temporal_discriminator.parameters(), lr=lr_D, betas=(0, 0.999))
    
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optim_G, 10)
    scheduler_SD = torch.optim.lr_scheduler.CosineAnnealingLR(optim_SD, 10)
    scheduler_TD = torch.optim.lr_scheduler.CosineAnnealingLR(optim_TD, 10)
    ##############################
    
    generator.to(device)
    spatial_discriminator.to(device)
    temporal_discriminator.to(device)

    #generator.half()
    #spatial_discriminator.half()
    #temporal_discriminator.half()
    print("##### Model loaded! #####")
    writer = SummaryWriter(f"runs/DGMR-all-0126-2016-2018")

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

    scaler_G = GradScaler()
    scaler_SD = GradScaler()
    scaler_TD = GradScaler()

    for e in range(start_e, start_e+n_epoch):
        TD_losses, SD_losses = [], []
        val_losses = []
        for img in tqdm(train_loader):
            b = img.shape[0]
            img_x = img[:, 0:in_step, ...].to(device)
            img_y = img[:, in_step:, ...].to(device)            
            
            ##### GENERATE FAKE DATA #####
            generator.eval()
            if True:
                #with autocast():
                pred = generator(img_x)
                torch.cuda.empty_cache()

            ##### TRAIN SPATIAL D #####
            spatial_discriminator.train()
            spatial_discriminator.zero_grad()
            if True:
                #with autocast():
                real_spatial_validity = spatial_discriminator(img_y.unsqueeze(2))
                torch.cuda.empty_cache()
                fake_spatial_validity = spatial_discriminator(pred.unsqueeze(2).detach())
                torch.cuda.empty_cache()
                real_loss_SD = spatial_loss(real_spatial_validity, true_spatial_validity.repeat(b, 1))
                torch.cuda.empty_cache()
                fake_loss_SD = spatial_loss(real_spatial_validity, false_spatial_validity.repeat(b, 1))
                torch.cuda.empty_cache()
            loss_SD = (real_loss_SD + fake_loss_SD) / 2
            #scaler_SD.scale(loss_SD).backward()
            loss_SD.backward()
            #scaler_SD.unscale_(optim_SD)
            #torch.nn.utils.clip_grad_norm_(spatial_discriminator.parameters(), 2e-6)
            #scaler_SD.step(optim_SD)
            optim_SD.step()
            #scheduler_SD.step()
            #scaler_SD.update()
            SD_losses.append(loss_SD.item())
            torch.cuda.empty_cache()

            ##### TRAIN TEMPORAL D #####
            temporal_discriminator.train()
            temporal_discriminator.zero_grad()
            if True:
                #with autocast():
                real_temporal_validity = temporal_discriminator(img_y.unsqueeze(1))
                torch.cuda.empty_cache()
                fake_temporal_validity = temporal_discriminator(pred.unsqueeze(1).detach())
                torch.cuda.empty_cache()
                real_loss_TD = temporal_loss(real_temporal_validity, true_temporal_validity.repeat(b, 1))
                torch.cuda.empty_cache()
                fake_loss_TD = temporal_loss(fake_temporal_validity, false_temporal_validity.repeat(b, 1))
                torch.cuda.empty_cache()
            loss_TD = (real_loss_TD + fake_loss_TD) / 2
            #scaler_TD.scale(loss_TD).backward()
            loss_TD.backward()
            #scaler_TD.unscale_(optim_TD)
            #torch.nn.utils.clip_grad_norm_(temporal_discriminator.parameters(), 2e-6)
            #scaler_TD.step(optim_TD)
            optim_TD.step()
            #scheduler_TD.step()
            #scaler_TD.update()
            TD_losses.append(loss_TD.item())
            torch.cuda.empty_cache()

            ##### TRAIN G #####
            generator.train()
            temporal_discriminator.eval()
            spatial_discriminator.eval()
            G_losses = []
            if e % 2 == 0:
                optim_G.zero_grad()
                if True:
                    #with autocast():
                    pred = generator(img_x)
                    torch.cuda.empty_cache()
                    loss = generator_loss(pred, img_y)
                    torch.cuda.empty_cache()
                G_losses.append(loss.item())
                
                if True:
                    #with autocast():
                    temporal_validity = temporal_discriminator(pred.unsqueeze(1).detach())
                    torch.cuda.empty_cache()
                    spatial_validity = spatial_discriminator(pred.unsqueeze(2).detach())
                    torch.cuda.empty_cache()
                    loss_SD = spatial_loss(real_spatial_validity, true_spatial_validity.repeat(b, 1))
                    torch.cuda.empty_cache()
                    loss_TD = temporal_loss(real_temporal_validity, true_temporal_validity.repeat(b, 1))
                    torch.cuda.empty_cache()
                loss = loss + 1 * (loss_SD.item() + loss_TD.item())
                #scaler_G.scale(loss).backward()
                loss.backward()
                #scaler_G.unscale_(optim_G)
                #torch.nn.utils.clip_grad_norm_(generator.parameters(), 2e-6)
                #scaler_G.step(optim_G)
                optim_G.step()
                #scheduler_G.step()
                #scaler_G.update()
                torch.cuda.empty_cache()

        if e % 2 == 0:
            with torch.no_grad():
                generator.eval()
                for j, (img, time) in enumerate(test_loader):
                    #if j % 100 != 0: continue # save some time
                    b = img.shape[0]
                    img_x = img[:, :in_step, ...].to(device)
                    img_y = img[:, in_step:, ...].to(device)
                    
                    if True:
                        #with autocast():
                        pred = generator(img_x)
                        loss = generator_loss(pred, img_y)
                    val_losses.append(loss.item())

                    if j == 0 or j == 100:
                        image = torch.cat((img_y[0].unsqueeze(0), pred[0].unsqueeze(0)), dim=0)
                        true_title = [f"{time[in_step+_][0]} (True)" for _ in range(out_step)]
                        pred_title = [f"{time[in_step+_][0]} (Pred)" for _ in range(out_step)]
                        figure = plot_test_image(image, true_title, pred_title)
                        writer.add_figure(f"Output: {j}", figure, e)
                        #if e % 10 == 0:
                        #    writer.add_figure("Output_ckpt", figure, e)
            print(f"Epoch: {e}, G loss: {np.mean(G_losses):.3f}, SD loss: {np.mean(SD_losses):.3f}, TD loss: {np.mean(TD_losses):.3f}, Val loss: {np.mean(val_losses):.3f}")

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
                        out_step = out_step,
                        nonzeros_th = nonzeros_th,
                        img_size = img_size,
                        dbz = dbz,
                        batch_size = batch_size,
                        )
                    )
            torch.save(state, "model_best.pth.tar")
            #print(f">>>>>>>>> Model saved. Minimum generator loss: {min_loss: .4f}")

            writer.add_scalar("generator loss", np.mean(G_losses), e)
            writer.add_scalar("temporal loss", np.mean(TD_losses), e)
            writer.add_scalar("spatial loss", np.mean(SD_losses), e)
            writer.add_scalar("validation loss", np.mean(val_losses), e)

        torch.cuda.empty_cache()
    return

if __name__ == "__main__":
    main()
