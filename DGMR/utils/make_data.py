import os
import numpy as np
import dask.array as da
from torchvision import transforms

from DGMR.utils.dataset import RainDataset
from DGMR.utils.data_parser import Nimrod_parser, min_max_normalizer


def count_nonzeros(memmap):
    os.system("echo")
    os.system("echo Counting nonzeros may take a while...")
    rain_record = []
    for map in memmap:
        rain_record.append((map>0).sum())
    return np.array(rain_record)

def get_memmap(npy_path, dtype=None, shape=None):
    if isinstance(npy_path, tuple):
        npy_path = eval(npy_path)
        memmap_list = []
        for npy, sp in zip(npy_path, shape):
            memmap = np.memmap(npy, dtype=dtype, shape=sp)
            memmap_list.append(memmap)
        final_memmap = da.concatenate(memmap_list, axis=0)
        return final_memmap

    if isinstance(npy_path, str):
        return np.memmap(npy_path, dtype=dtype, shape=shape)


def make_dataset(
    cfg,
    mode: str=None
    ):
    npy_path = cfg.SETTINGS.DATA_PATH

    memory_map = get_memmap(
        npy_path,
        dtype=getattr(np, cfg.SETTINGS.DATA_TYPE),
        shape=eval(cfg.SETTINGS.DATA_SHAPE)
        )
    
    if cfg.SETTINGS.RAIN_RECORD_PATH is not None:
        pass
    rain_record = count_nonzeros(memory_map)
    
    parser = lambda x: Nimrod_parser(x, rain_th=cfg.PARAMS.RAIN_TH, rain2dbz=cfg.PARAMS.CONVERT_TO_DBZ, min_dbz=cfg.PARAMS.MIN_VALUE)
    normalizer = lambda x: min_max_normalizer(x, min_value=cfg.PARAMS.MIN_VALUE, max_value=cfg.PARAMS.MAX_VALUE)
    
    if mode == "train":
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(cfg.PARAMS.CROP_SIZE),
            transforms.ToTensor()
        ])
    elif mode == "test" or mode == "val":
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(cfg.PARAMS.CROP_SIZE),
            transforms.ToTensor()
        ])
    
    dataset = RainDataset(
        memory_map=memory_map,
        rain_record=rain_record,
        rain_coverage=cfg.PARAMS.COVERAGE,
        in_step=cfg.PARAMS.INPUT_FRAME,
        out_step=cfg.PARAMS.OUTPUT_FRAME,
        parser=parser,
        normalizer=normalizer,
        transform=transform
    )
    return dataset