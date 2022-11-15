import os
import argparse
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms

from DGMR.utils.config import convert
from DGMR.DGMRTrainer import DGMRTrainer


def validate(cfg):
    seed_everything(cfg.SETTINGS.RNG_SEED, workers=True)

    tb_logger = TensorBoardLogger(
        cfg.TENSORBOARD.SAVE_DIR,
        name=cfg.TENSORBOARD.NAME,
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

    model = DGMRTrainer(cfg)
    trainer.validate(model, ckpt_path=cfg.SETTINGS.IMPORT_CKPT_PATH)
    pass

def train(cfg):
    seed_everything(cfg.SETTINGS.RNG_SEED, workers=True)

    tb_logger = TensorBoardLogger(
        cfg.TENSORBOARD.SAVE_DIR,
        name=cfg.TENSORBOARD.NAME,
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

    model = DGMRTrainer(cfg)
    if cfg.SETTINGS.IMPORT_CKPT_PATH:
        trainer.fit(model, ckpt_path=cfg.SETTINGS.IMPORT_CKPT_PATH)
    else:
        trainer.fit(model)
    pass

def test(cfg):
    seed_everything(cfg.SETTINGS.RNG_SEED, workers=True)

    tb_logger = TensorBoardLogger(
        cfg.TENSORBOARD.SAVE_DIR,
        name=cfg.TENSORBOARD.NAME,
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

    model = DGMRTrainer(cfg)
    trainer.test(model, ckpt_path=cfg.SETTINGS.IMPORT_CKPT_PATH)
    pass

def main(args):
    cfg_file = args.config
    cfg = convert(cfg_file)

    if args.mode == "train":
        train(cfg)
    elif args.mode == "test":
        test(cfg)
    elif args.mode == "validate":
        validate(cfg)
    else:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--config", type=str,
        help="a config file in yaml format to control experiment")
    parser.add_argument(
        "-m", "--mode", type=str,
        choices=["train", "validate", "test"])

    args = parser.parse_args()

    main(args)
