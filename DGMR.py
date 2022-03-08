import tensorboard
import torch
import pytorch_lightning as pl
from collections import OrderedDict
from pytorch_lightning import loggers, Trainer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.Generator import Generator
from models.Discriminators import Discriminators
from utils.loss import DiscriminatorLoss, Regularizer
from utils.plot import plot_test_image

class DGMR(pl.LightningModule):
    """
    Trainer of DGMR,
    models:
        Generator(in_step, out_step, debug)
        SpatialDiscriminator(n_frame, debug)
        TemporalDiscriminator(crop_size, debug)
    """
    def __init__(
        self,
        in_step = 4,
        out_step = 6,
        n_sample = 6, # numbers of samples generated when training generator
        n_frame = 6, # numbers of frames spatial discriminator samples
        crop_size = 128, # input image size of temporal discriminator
        batch_size = 1,
        lr_G = 5e-5,
        lr_D = 2e-4,
        alpha = 20, # weight of generator loss
        margin = 0.5, # margin of discriminator loss
        beta = (0, 0.999),
        debug = False,
        fixed_val_img = None,
        **kwargs
    ):
        super(DGMR, self).__init__()
        self.in_step = in_step
        self.out_step = out_step
        self.n_sample = n_sample
        self.n_frame = n_frame
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.alpha = alpha
        self.margin = margin
        self.beta = beta
        self.debug = debug
        self.fixed_val_img = fixed_val_img

        self.generator = Generator(
            in_step = self.in_step,
            out_step = self.out_step,
            debug = self.debug)

        self.discriminators = Discriminators(
            n_frame=self.n_frame,
            crop_size=self.crop_size,
            debug=self.debug)

        self.cal_lossG = Regularizer(alpha=self.alpha)
        self.cal_lossD = DiscriminatorLoss(margin=self.margin)

        self.automatic_optimization = False

    def prepare_data(self):
        pass

    def configure_optimizers(self):
        optim_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr_G, betas=self.beta)
        optim_D = torch.optim.Adam(self.discriminators.parameters(), lr=self.lr_D, betas=self.beta)
        return [optim_G, optim_D], []

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x.requires_grad = True
        optim_G, optim_D = self.optimizers()

        is_last_batch_to_accumulate = (batch_idx + 1) % 2 == 0 or self.trainer.is_last_batch
        with optim_D.toggle_model(sync_grad=is_last_batch_to_accumulate):
            ##### TRAIN D #####
            for _ in range(2):
                pred = self(x)

                real_scoreD = self.discriminators(x, y)
                fake_scoreD = self.discriminators(x, pred.detach())

                loss_D = self.cal_lossD(real_scoreD, True) + self.cal_lossD(fake_scoreD, False)

                self.manual_backward(loss_D)
            if is_last_batch_to_accumulate:
                optim_D.zero_grad()
                optim_D.step()

        ##### TRAIN G #####
        with optim_G.toggle_model(sync_grad=is_last_batch_to_accumulate):
            pred = [self(x) for _ in range(self.n_sample)]
            loss_G = 0
            for p in pred:
                loss = self.cal_lossD(self.discriminators(p.detach(), y), True)
                loss_G += loss
            loss_G /= self.n_sample
            loss_G += self.cal_lossG(torch.stack(pred, dim=0).mean(), y)

            self.manual_backward(loss_G)
            if is_last_batch_to_accumulate:
                optim_G.zero_grad()
                optim_G.step()

        ##### UPDATE LOGGINGS #####
        self.log("train_g_loss", loss_G.detach(), on_step=True, prog_bar=True)
        self.log("train_d_loss", loss_D.detach(), on_step=True, prog_bar=True)
        
    def on_train_epoch_end(self):
        if self.fixed_val_img is not None:
            pass
        else: return

        x = self.fixed_val_img[:, :self.in_step, ...]
        y = self.fixed_val_img[:, self.in_step:, ...]

        pred = self(x)
        score = self.discriminators(x, pred)

        loss_G = self.cal_lossG(pred, y)
        loss_D = self.cal_lossD(score, True)

        self.log("val_g_loss", loss_G.detach(), on_epoch=True, prog_bar=True, logger=True)
        self.log("val_d_loss", loss_D.detach(), on_epoch=True, prog_bar=True, logger=True)
        
        tb = self.logger.experiment
        pred_title = [f"+{5*(t+1)} min (pred)" for t in range(y.shape[1])]
        true_title = [f"+{5*(t+1)} min (true)" for t in range(y.shape[1])]
        for i, (img_p, img_y) in enumerate(zip(pred, y)):
            fig = plot_test_image(img_p, img_y, pred_title, true_title)
            tb.add_figure(f"val/ image{i}", fig, global_step=self.current_epoch)

if __name__ == "__main__":
    # Prepare testing data
    import os
    from dataset import SparseCOONimrodDataset
    root = "/home/yihan/yh/research/Nimrod"
    files = [os.path.join(root, f"SparseCOONimrod_{y}.npz") for y in range(2016, 2019)]
    logfiles = [os.path.join(root, "log", f"rain_record_{y}.csv") for y in range(2016, 2019)]

    ##### PARAMS #####
    nonzeros_th = 200000
    in_step = 4
    out_step = 6
    img_size = 256
    dbz = True
    batch_size = 4
    #######################

    test_dataset = SparseCOONimrodDataset(root=root, target_year=range(2016, 2019),
            nonzeros_th=nonzeros_th, in_step=in_step, out_step=out_step,
            cropsize=img_size, dbz=dbz, return_time=False, data_type="train")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)

    fixed_x, fixed_y = test_dataset[10]
    fixed = torch.cat((fixed_x, fixed_y), dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    # Trainer test
    tb_logger = loggers.TensorBoardLogger("/home/yihan/yh/research/logs/")
    trainer = Trainer(logger=tb_logger)
    model = DGMR(fixed_val_img=fixed.cuda())
    trainer = Trainer(
        gpus="0",
        devices=1, 
        max_epochs=5,
        accelerator="gpu",
        enable_progress_bar=True)
    trainer.fit(model, test_loader)