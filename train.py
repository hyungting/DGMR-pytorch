import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torch.utils.data import DataLoader

from DGMR import DGMR
from dataset import SparseCOONimrodDataset

def main():
    # Prepare testing data
    root = "/home/yihan/yh/research"
    data_root = os.path.join(root, "Nimrod")
    
    nonzeros_th = 200000
    in_step = 4
    out_step = 6
    img_size = 256
    dbz = True
    batch_size = 4
    
    train_dataset = SparseCOONimrodDataset(root=data_root, target_year=range(2016, 2019),
            nonzeros_th=nonzeros_th, in_step=in_step, out_step=out_step,
            cropsize=img_size, dbz=dbz, return_time=False, data_type="train")
    val_dataset = SparseCOONimrodDataset(root=data_root, target_year=range(2019, 2020),
            nonzeros_th=nonzeros_th, in_step=in_step, out_step=out_step,
            cropsize=img_size, dbz=dbz, return_time=False, data_type="val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    # Trainer test
    seed_everything(42, workers=True)
    logger_root = os.path.join(root, "dgmr_logs")
    ckpt_root = os.path.join(root, "dgmr_logs")
    tb_logger = TensorBoardLogger(
            logger_root,
            name='DGMR',
            version=0,
            log_graph=False,
            default_hp_metric=True,
            prefix='',
            sub_dir=None)
    trainer = Trainer(logger=tb_logger)
    model = DGMR(in_step=in_step, out_step=out_step)
    trainer = Trainer(
        gpus=1,
        deterministic=True,
        max_epochs=500000,
        accelerator="gpu",
        #strategy="ddp",
        #sync_batchnorm=True,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,
        default_root_dir=ckpt_root)
    trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader)

main()