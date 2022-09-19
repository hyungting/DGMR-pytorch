import torch

def Nimrod_parser(
    array: torch.tensor=None,
    rain_th: float=0.01,
    rain2dbz: bool=True,
    min_dbz: float=None,
    ):
    array = array / 32
    if rain2dbz:
        array[array>rain_th] = torch.log10(((array[array>rain_th] ** (8 / 5)) * 200)) * 10
        array[array<=rain_th] = min_dbz
    return array


def min_max_normalizer(
    array: torch.tensor=None,
    min_value: float=None,
    max_value: float=None
    ):
    return (array - min_value) / (max_value - min_value)