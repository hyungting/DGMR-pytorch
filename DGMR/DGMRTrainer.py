import copy
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from DGMR.DGMRModels import DGMRGenerator, DGMRDiscriminators

from DGMR.utils.make_data import make_dataset
from DGMR.utils.loss import PixelWiseRegularizer, HingeLoss
from DGMR.utils.plot import plot_rain_field, plot_metric_boxplot, plot_metric_mean
from DGMR.utils.metrics import Evaluator
from DGMR.utils.config import ConfigException

class Trainer(pl.LightningModule):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        
        self.in_step = cfg.PARAMS.INPUT_FRAME
        self.out_step = cfg.PARAMS.OUTPUT_FRAME
        self.img_size = cfg.PARAMS.CROP_SIZE
        self.n_sample = cfg.PARAMS.N_SAMPLE
        self.output_csv = f"{cfg.GENERATOR.MODEL.FUNCTION}_{cfg.TENSORBOARD.VERSION}.csv"

        self.evaluator = Evaluator(
            thresholds=cfg.EVALUATION.THRESHOLDS,
            pooling_scales=cfg.EVALUATION.POOLING_SCALES,
            norm=cfg.PARAMS.NORMALIZE,
            min_value=cfg.PARAMS.MIN_VALUE,
            max_value=cfg.PARAMS.MAX_VALUE,
            dbz_to_rain=False)#cfg.PARAMS.CONVERT_TO_DBZ)

        self.thresholds = cfg.EVALUATION.THRESHOLDS
        self.pooling_scales = cfg.EVALUATION.POOLING_SCALES

        self.generator = self.get_model(cfg=cfg.GENERATOR, hparams=cfg.PARAMS)
        self.generator_loss = self.get_loss(cfg=cfg.GENERATOR)

        self.train_with_gan = False

        if cfg.DISCRIMINATOR.MODEL.FUNCTION is not None:
            self.discriminator = self.get_model(cfg=cfg.DISCRIMINATOR, hparams=cfg.PARAMS)
            self.discriminator_loss = self.get_loss(cfg=cfg.DISCRIMINATOR)
            self.train_d_iter = cfg.DISCRIMINATOR.ITER
            self.train_with_gan = True
            self.automatic_optimization = False

        self.save_hyperparameters()

    def save_hyperparameters(self):
        ISDICT = lambda x: isinstance(x, tuple) and hasattr(x, "_asdict") and hasattr(x, "_fields")
        def add_hparams(dictionary, **kwargs):
            for key in dir(dictionary):
                previous_key = []
                for k, v in kwargs.items():
                    previous_key += v
                if key[0].isupper():
                    current_key = previous_key
                    current_key.append(key)
                    if ISDICT(eval(f"dictionary.{key}")):
                        add_hparams(eval(f"dictionary.{key}"), add_key=current_key)
                    else:
                        if "PARAMS" in current_key:
                            text = ""
                            for i, t in enumerate(current_key):
                                if i > 0:
                                    text += "."
                                text += t
                            if i == len(current_key) - 1 and t != "PARAMS":
                                self.hparams[text] = eval(f"dictionary.{key}")      
        add_hparams(self.cfg)
        return

    def get_model(self, cfg, hparams):
        model = cfg.MODEL.FUNCTION
        model_params = cfg.MODEL.PARAMS
        if model == "DGMRGenerator":
            return DGMRGenerator(
                in_step=hparams.INPUT_FRAME,
                out_step=hparams.OUTPUT_FRAME
            )
        elif model == "DGMRDiscriminators":
            return DGMRDiscriminators(
                n_frame=model_params.N_FRAME,
                crop_size=model_params.CROP_SIZE
            )
        else:
            raise ConfigException(f"Model '{model}' is not supported.")

    def get_loss(self, cfg):
        loss = cfg.LOSS.FUNCTION
        hparams = cfg.LOSS.PARAMS
        if loss == "PixelWiseRegularizer":
            return PixelWiseRegularizer(cfg=hparams)
        elif loss == "HingeLoss":
            return HingeLoss(cfg=hparams)
        else:
            raise ConfigException(f"Loss '{loss}' is not supported.")

    def train_dataloader(self):
        train_dataset = make_dataset(cfg=self.cfg, mode="train")
        return DataLoader(
            train_dataset,
            batch_size=self.cfg.PARAMS.BATCH_SIZE,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            pin_memory=self.cfg.DATALOADER.PIN_MEMORY
        )

    def val_dataloader(self):
        val_dataset = make_dataset(cfg=self.cfg, mode="val")
        return DataLoader(
            val_dataset,
            batch_size=self.cfg.VAL.BATCH_SIZE,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            pin_memory=self.cfg.DATALOADER.PIN_MEMORY
        )
    
    def test_dataloader(self):
        test_dataset = make_dataset(cfg=self.cfg, mode="test")
        return DataLoader(
            test_dataset,
            batch_size=self.cfg.TEST.BATCH_SIZE,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            pin_memory=self.cfg.DATALOADER.PIN_MEMORY
        )

    def get_optimizer(self, optimizer, param):
        if optimizer == "SGD":
            return lambda parameters: torch.optim.SGD(parameters, lr=param.LR, momentum=param.MOMENTUM)
        elif optimizer == "Adam":
            return lambda parameters: torch.optim.Adam(parameters, lr=param.LR, betas=param.BETA)
        elif optimizer == "AdamW":
            return lambda parameters: torch.optim.AdamW(parameters, lr=param.LR, betas=param.BETA, weight_decay=param.WEIGHT_DECAY)
        else:
            return lambda parameters: eval(f"torch.optim.{optimizer}({parameters}, lr={param.LR})")    
        raise ConfigException(f"Optimizer '{loss}' is not supported.")

    def get_scheduler(self, scheduler, param):
        if scheduler == "StepLR":
            return lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=param.STEP_SIZE, gamma=param.GAMMA)
        elif scheduler == "CosineAnnealingLR":
            return lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=param.T_MAX)
        elif scheduler == "ReduceLROnPlateau":
            return lambda optimizer: {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=param.MODE, factor=param.FACTOR, patience=param.PATIENCE, threshold=param.THRESHOLD),
                "monitor": param.METRIC}
        else:
            return lambda optimizer: eval(f"torch.optim.lr_scheduler.{scheduler}({optimizer})")
        raise ConfigException(f"Scheduler '{scheduler}' is not supported.")

    def configure_optimizers(self):
        optim_G = self.get_optimizer(
            optimizer=self.cfg.GENERATOR.OPTIMIZER.FUNCTION,
            param=self.cfg.GENERATOR.OPTIMIZER.PARAMS)(self.generator.parameters())
        scheduler = []

        if self.cfg.GENERATOR.SCHEDULER.FUNCTION is not None:
            scheduler_G = self.get_scheduler(
                scheduler=self.cfg.GENERATOR.SCHEDULER.FUNCTION,
                param=self.cfg.GENERATOR.SCHEDULER.PARAMS)(optim_G)
            scheduler.append(scheduler_G)
            
        if self.train_with_gan:
            optim_D = self.get_optimizer(
                optimizer=self.cfg.DISCRIMINATOR.OPTIMIZER.FUNCTION,
                param=self.cfg.DISCRIMINATOR.OPTIMIZER.PARAMS)(self.discriminator.parameters())

            if self.cfg.DISCRIMINATOR.SCHEDULER.FUNCTION is not None:
                scheduler_D = self.get_scheduler(
                    schedulXer=self.cfg.DISCRIMINATOR.SCHEDULER.FUNCTION,
                    param=self.cfg.DISCRIMINATOR.SCHEDULER.PARAMS)(optim_D)
                scheduler.append(scheduler_D)

            if len(scheduler) > 0:
                return [optim_G, optim_D], scheduler
            else:
                return [optim_G, optim_D]
        else:
            if len(scheduler) > 0:
                return {"optimizer": optim_G, "scheduler": scheduler_G}
            else:
                return optim_G
    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x.requires_grad = True

        log_dict = {}

        if self.train_with_gan:
            optim_G, optim_D = self.optimizers()
            is_last_batch_to_accumulate = (batch_idx + 1) % 2 == 0 or self.trainer.is_last_batch
            ##### TRAIN D #####
            with optim_D.toggle_model(sync_grad=is_last_batch_to_accumulate):
                for _ in range(self.train_d_iter):
                    p = self(x)
                    f_out = self.discriminator(x, p.detach())
                    t_out = self.discriminator(x, y)
                    fake_feature_loss = self.discriminator_loss(f_out, False)
                    true_feature_loss = self.discriminator_loss(t_out, True)
                    feature_loss = (fake_feature_loss + true_feature_loss) / 2
                    
                    self.manual_backward(feature_loss)
                    if is_last_batch_to_accumulate:
                        optim_D.step()
                        optim_D.zero_grad()
            log_dict["D/ D loss"] = feature_loss.detach()
            log_dict["D/ true D loss"] = true_feature_loss.detach()
            log_dict["D/ fake D loss"] = fake_feature_loss.detach()

            ##### TRAIN MODEL #####
            with optim_G.toggle_model(sync_grad=is_last_batch_to_accumulate):
                if self.n_sample > 1:
                    pred = [self(x) for _ in range(self.n_sample)]
                    pred = torch.mean(torch.stack(pred, dim=0), dim=0)
                else:
                    pred = self(x)
                model_loss = self.generator_loss(pred, y)
                feature_loss = self.discriminator_loss(self.discriminator(x, pred), True)
                
                train_model_loss = model_loss + feature_loss
                self.manual_backward(train_model_loss)
                if is_last_batch_to_accumulate:
                    optim_G.step()
                    optim_G.zero_grad()
            log_dict["G/ G loss"] = model_loss.detach()
            log_dict["G/ D loss"] = feature_loss.detach()
        else:
            if self.n_sample > 1:
                pred = [self(x) for _ in range(self.n_sample)]
                pred = torch.mean(torch.stack(pred, dim=0), dim=0)
            else:
                pred = self(x)
            model_loss = self.generator_loss(pred, y)
            log_dict["loss"] = model_loss
            
        ##### UPDATE LOGGINGS #####
        if self.train_with_gan:
            self.log("train/ G loss", log_dict["G/ G loss"], on_step=True, prog_bar=True, logger=False)
            self.log("train/ G score", log_dict["G/ D loss"], on_step=True, prog_bar=True, logger=False)
            self.log("train/ D loss", log_dict["D/ D loss"], on_step=True, prog_bar=True, logger=False)
            self.log("train/ D true loss", log_dict["D/ true D loss"], on_step=True, prog_bar=True, logger=False)
            self.log("train/ D fake loss", log_dict["D/ fake D loss"], on_step=True, prog_bar=True, logger=False)
        else:
            self.log("train/ G loss", log_dict["loss"].detach(), on_step=True, prog_bar=True, logger=False)

        ##### VISUALIZE PREDICTIONS #####
        if batch_idx == 0:
            fig = self.visualize_image(x, pred.detach(), y)
            self.logger.experiment.add_figure("train/ image", fig, global_step=self.current_epoch)

        return log_dict

    def visualize_image(self, x, pred, y):
        pred_title = [f"step {t} (pred)" for t in range(self.out_step+1)]
        true_title = [f"step {t} (true)" for t in range(self.out_step+1)]
        pred_img = torch.cat((x[0, -1, ...].unsqueeze(0), pred[0]), dim=0)
        true_img = torch.cat((x[0, -1, ...].unsqueeze(0), y[0]), dim=0)
        return plot_rain_field(pred_img, true_img, pred_title, true_title)
    
    def training_epoch_end(self, outputs):
        if self.train_with_gan:
            model_loss = torch.mean(torch.stack([values["G/ G loss"] for values in outputs]))
            model_score = torch.mean(torch.stack([values["G/ D loss"] for values in outputs]))
            true_feature_loss = torch.mean(torch.stack([values["D/ true D loss"] for values in outputs]))
            fake_feature_loss = torch.mean(torch.stack([values["D/ fake D loss"] for values in outputs]))
            feature_loss = torch.mean(torch.stack([values["D/ D loss"] for values in outputs]))
            
            self.logger.experiment.add_scalar("G/ G loss", model_loss, global_step=self.current_epoch)
            self.logger.experiment.add_scalar("G/ D loss", model_score, global_step=self.current_epoch)
            self.logger.experiment.add_scalar("D/ true D loss", true_feature_loss, global_step=self.current_epoch)
            self.logger.experiment.add_scalar("D/ fake D loss", fake_feature_loss, global_step=self.current_epoch)
            self.logger.experiment.add_scalar("D/ D loss", feature_loss, global_step=self.current_epoch)
        else:
            model_loss = torch.mean(torch.stack([values["loss"] for values in outputs]))
            self.logger.experiment.add_scalar("G/ G loss", model_loss, global_step=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch

        pred = self(x)
        loss = self.generator_loss(pred, y)
        self.log("val/ G loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        
        self.evaluator.calculate_all(pred, y)

        log_dict = {
            "val_loss": loss.detach(),
            "MSE": self.evaluator.MSE,
            "MAE": self.evaluator.MAE,
            "RMSE": self.evaluator.RMSE
            }

        for th in self.thresholds:
            log_dict[f"CSI_{th}"] = self.evaluator.CSI[th]
            log_dict[f"POD_{th}"] = self.evaluator.POD[th]
            log_dict[f"FAR_{th}"] = self.evaluator.FAR[th]
        
        for scale in self.pooling_scales:
            log_dict[f"CRPS_avg_{scale}"] = self.evaluator.CRPS_avg[scale]
            log_dict[f"CRPS_max_{scale}"] = self.evaluator.CRPS_max[scale]
        
        if batch_idx == 0:
            fig = self.visualize_image(x, pred.detach(), y)
            self.logger.experiment.add_figure(f"val/ image", fig, global_step=self.current_epoch)

        return log_dict
    
    def validation_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([values["val_loss"] for values in outputs]))
        self.logger.experiment.add_scalar("val/ loss", val_loss, global_step=self.current_epoch)
        
        mse = torch.stack([values["MSE"] for values in outputs], dim=0)
        mae = torch.stack([values["MAE"] for values in outputs], dim=0)
        rmse = torch.stack([values["RMSE"] for values in outputs], dim=0)
        
        self.logger.experiment.add_figure("hp/ MSE", plot_metric_boxplot(metric_time=mse, ylabel="MSE"), global_step=self.current_epoch)
        self.logger.experiment.add_figure("hp/ MAE", plot_metric_boxplot(metric_time=mae, ylabel="MAE"), global_step=self.current_epoch)
        self.logger.experiment.add_figure("hp/ RMSE", plot_metric_boxplot(metric_time=rmse, ylabel="RMSE"), global_step=self.current_epoch)

        for th in self.thresholds:
            csi = torch.stack([values[f"CSI_{th}"] for values in outputs], dim=0)
            pod = torch.stack([values[f"POD_{th}"] for values in outputs], dim=0)
            far = torch.stack([values[f"FAR_{th}"] for values in outputs], dim=0)
            self.logger.experiment.add_figure(f"hp/ CSI ({th}mm/h)", plot_metric_boxplot(metric_time=csi, ylabel="CSI", title=f"Precipitation ≧ {th} mm h⁻¹"), global_step=self.current_epoch)
            self.logger.experiment.add_figure(f"hp/ POD ({th}mm/h)", plot_metric_boxplot(metric_time=pod, ylabel="POD", title=f"Precipitation ≧ {th} mm h⁻¹"), global_step=self.current_epoch)
            self.logger.experiment.add_figure(f"hp/ FAR ({th}mm/h)", plot_metric_boxplot(metric_time=far, ylabel="FAR", title=f"Precipitation ≧ {th} mm h⁻¹"), global_step=self.current_epoch)

        for scale in self.pooling_scales:
            avg_crps = np.stack([values[f"CRPS_avg_{scale}"] for values in outputs], axis=0)
            max_crps = np.stack([values[f"CRPS_max_{scale}"] for values in outputs], axis=0)
            self.logger.experiment.add_figure(f"hp/ Avgpooled CRPS ({scale}km)", plot_metric_boxplot(metric_time=avg_crps, ylabel="Avg-pooled CRPS", title=f"Pooling scale = {scale} km"), global_step=self.current_epoch)
            self.logger.experiment.add_figure(f"hp/ Maxpooled CRPS ({scale}km)", plot_metric_boxplot(metric_time=max_crps, ylabel="Max-pooled CRPS", title=f"Pooling scale = {scale} km"), global_step=self.current_epoch)

    def test_step(self, batch, batch_idx):
        x, y = batch

        pred = self(x)
        loss = self.generator_loss(pred, y)
        self.log("test/ G loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=False)

        self.evaluator.calculate_all(pred, y)

        log_dict = {
            "test_loss": loss.detach(),
            "MSE": self.evaluator.MSE,
            "MAE": self.evaluator.MAE,
            "RMSE": self.evaluator.RMSE
            }

        for th in self.thresholds:
            log_dict[f"CSI_{th}"] = self.evaluator.CSI[th]
            log_dict[f"POD_{th}"] = self.evaluator.POD[th]
            log_dict[f"FAR_{th}"] = self.evaluator.FAR[th]
        
        for scale in self.pooling_scales:
            log_dict[f"CRPS_avg_{scale}"] = self.evaluator.CRPS_avg[scale]
            log_dict[f"CRPS_max_{scale}"] = self.evaluator.CRPS_max[scale]
        
        if batch_idx == 0:
            fig = self.visualize_image(x, pred.detach(), y)
            self.logger.experiment.add_figure(f"test/ image", fig, global_step=self.current_epoch)

        # persistence error
        pred_error = self.cal_persistence_error(x[:, -1].unsqueeze(1), pred)
        obs_error = self.cal_persistence_error(x[:, -1].unsqueeze(1), y)
        log_dict["pred_persistence_error"] = pred_error
        log_dict["obs_persistence_error"] = obs_error

        return log_dict
    
    def test_epoch_end(self, outputs):
        insert_to_df = lambda input_tensor: input_tensor[~torch.any(input_tensor.isnan(), dim=1)].cpu().numpy().mean(axis=0)
        test_loss = torch.mean(torch.stack([values["test_loss"] for values in outputs]))
        self.logger.experiment.add_scalar("test/ loss", test_loss, global_step=self.current_epoch)
        
        metric_df = pd.DataFrame(index=[5*(t+1)for t in range(self.out_step)])
        
        mse = torch.stack([values["MSE"] for values in outputs], dim=0)
        mae = torch.stack([values["MAE"] for values in outputs], dim=0)
        rmse = torch.stack([values["RMSE"] for values in outputs], dim=0)

        metric_df["MSE"] = insert_to_df(mse)
        metric_df["MAE"] = insert_to_df(mae)
        metric_df["RMSE"] = insert_to_df(rmse)
        
        self.logger.experiment.add_figure("hp/ MSE", plot_metric_boxplot(metric_time=mse, ylabel="MSE", showfliers=False), global_step=self.current_epoch)
        self.logger.experiment.add_figure("hp/ MAE", plot_metric_boxplot(metric_time=mae, ylabel="MAE", showfliers=False), global_step=self.current_epoch)
        self.logger.experiment.add_figure("hp/ RMSE", plot_metric_boxplot(metric_time=rmse, ylabel="RMSE", showfliers=False), global_step=self.current_epoch)
        self.logger.experiment.add_figure("hp/ MSE (mean)", plot_metric_mean(metric_time=mse, ylabel="MSE"), global_step=self.current_epoch)
        self.logger.experiment.add_figure("hp/ MAE (mean)", plot_metric_mean(metric_time=mae, ylabel="MAE"), global_step=self.current_epoch)
        self.logger.experiment.add_figure("hp/ RMSE (mean)", plot_metric_mean(metric_time=rmse, ylabel="RMSE"), global_step=self.current_epoch)

        for th in self.thresholds:
            csi = torch.stack([values[f"CSI_{th}"] for values in outputs], dim=0)
            pod = torch.stack([values[f"POD_{th}"] for values in outputs], dim=0)
            far = torch.stack([values[f"FAR_{th}"] for values in outputs], dim=0)

            metric_df[f"CSI_{th}"] = insert_to_df(csi)
            metric_df[f"POD_{th}"] = insert_to_df(pod)
            metric_df[f"FAR_{th}"] = insert_to_df(far)

            dbz2rain = ((10 ** (th / 10)) / 200) ** (1 / 1.6)

            self.logger.experiment.add_figure(f"hp/ CSI ({th}mm/h)", plot_metric_boxplot(metric_time=csi, ylabel="CSI", title=f"Precipitation ≧ {th} dBZ ({dbz2rain:.2f} mm hr⁻¹)", showfliers=False), global_step=self.current_epoch)
            self.logger.experiment.add_figure(f"hp/ POD ({th}mm/h)", plot_metric_boxplot(metric_time=pod, ylabel="POD", title=f"Precipitation ≧ {th} dBZ ({dbz2rain:.2f} mm hr⁻¹)", showfliers=False), global_step=self.current_epoch)
            self.logger.experiment.add_figure(f"hp/ FAR ({th}mm/h)", plot_metric_boxplot(metric_time=far, ylabel="FAR", title=f"Precipitation ≧ {th} dBZ ({dbz2rain:.2f} mm hr⁻¹)", showfliers=False), global_step=self.current_epoch)
            self.logger.experiment.add_figure(f"hp/ CSI ({th}mm/h) (mean)", plot_metric_mean(metric_time=csi, ylabel="CSI", title=f"Precipitation ≧ {th} dBZ ({dbz2rain:.2f} mm hr⁻¹)"), global_step=self.current_epoch)
            self.logger.experiment.add_figure(f"hp/ POD ({th}mm/h) (mean)", plot_metric_mean(metric_time=pod, ylabel="POD", title=f"Precipitation ≧ {th} dBZ ({dbz2rain:.2f} mm hr⁻¹)"), global_step=self.current_epoch)
            self.logger.experiment.add_figure(f"hp/ FAR ({th}mm/h) (mean)", plot_metric_mean(metric_time=far, ylabel="FAR", title=f"Precipitation ≧ {th} dBZ ({dbz2rain:.2f} mm hr⁻¹)"), global_step=self.current_epoch)

        for scale in self.pooling_scales:
            avg_crps = np.stack([values[f"CRPS_avg_{scale}"] for values in outputs], axis=0)
            max_crps = np.stack([values[f"CRPS_max_{scale}"] for values in outputs], axis=0)

            metric_df[f"CRPS_avg_{scale}"] = avg_crps.mean(axis=0)
            metric_df[f"CRPS_max_{scale}"] = max_crps.mean(axis=0)

            self.logger.experiment.add_figure(f"hp/ Avgpooled CRPS ({scale}km)", plot_metric_boxplot(metric_time=avg_crps, ylabel="Avg-pooled CRPS", title=f"Pooling scale = {scale} km", showfliers=False), global_step=self.current_epoch)
            self.logger.experiment.add_figure(f"hp/ Maxpooled CRPS ({scale}km)", plot_metric_boxplot(metric_time=max_crps, ylabel="Max-pooled CRPS", title=f"Pooling scale = {scale} km", showfliers=False), global_step=self.current_epoch)
            self.logger.experiment.add_figure(f"hp/ Avgpooled CRPS ({scale}km) (mean)", plot_metric_mean(metric_time=avg_crps, ylabel="Avg-pooled CRPS", title=f"Pooling scale = {scale} km"), global_step=self.current_epoch)
            self.logger.experiment.add_figure(f"hp/ Maxpooled CRPS ({scale}km) (mean)", plot_metric_mean(metric_time=max_crps, ylabel="Max-pooled CRPS", title=f"Pooling scale = {scale} km"), global_step=self.current_epoch)
            
        metric_df.to_csv(self.output_csv)


if __name__ == "__main__":
    # Prepare testing data
    import os
    import sys
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.loggers import TensorBoardLogger

    from utils.config import convert
    from collections import namedtuple

    cfg_file = sys.argv[1]
    cfg = convert(cfg_file)

    seed_everything(cfg.SETTINGS.RNG_SEED, workers=True)

    tb_logger = TensorBoardLogger(
        cfg.TENSORBOARD.SAVE_DIR,
        name=cfg.GENERATOR.MODEL.FUNCTION,
        version=cfg.TENSORBOARD.VERSION,
        log_graph=False,
        default_hp_metric=True
        )

    trainer = Trainer(
        accelerator="gpu",
        devices=cfg.SETTINGS.NUM_GPUS, 
        max_epochs=cfg.PARAMS.EPOCH,
        check_val_every_n_epoch=cfg.TRAIN.VAL_PERIOD,
        enable_progress_bar=True,
        logger=tb_logger
        )

    model = Experiment(cfg)
    print(model.hparams)
    #trainer.fit(model)
