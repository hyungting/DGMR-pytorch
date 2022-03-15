import math
import torch
import numpy as np
import properscoring as ps

from scipy import special as sp
from scipy.interpolate import interp1d

def cal_CSI(pred, target, th, TP=None, FP=None, FN=None):
    """
    Critical Success Index = TP / (TP + FP + FN)
    """
    if TP is not None and FP is not None and FN is not None:
        pass
    else:
        TP = np.sum(np.logical_and(pred>=th, target>=th))
        FP = np.sum(np.logical_and(pred>=th, target<th))
        FN = np.sum(np.logical_and(pred<th, target>=th))
    return TP / (TP + FP + FN)

def cal_POD(pred, target, th, TP=None, FN=None):
    """
    Porbability of Detection = TP / (TP + FN)
    """
    if TP is not None and FN is not None:
        pass
    else:
        TP = np.sum(np.logical_and(pred>=th, target>=th))
        FN = np.sum(np.logical_and(pred<th, target>=th))
    return TP / (TP + FN)

def cal_FAR(pred, target, th, FP=None, TP=None):
    """
    False Alarm Ratio = FP / (FP + TP)
    """
    if FP is not None and TP is not None:
        pass
    else:
        FP = np.sum(np.logical_and(pred>=th, target<th))
        TP = np.sum(np.logical_and(pred>=th, target>=th))

    return FP / (FP + TP)

def cal_CRPS(pred, target):
    """
    Continuous Ranked Probability Score
    This function uses the library written by researchers at The Climate Corporation.
    The original authors include Leon Barrett, Stephan Hoyer, Alex Kleeman and Drew O'Kane.
    """
    return ps.crps_ensemble(target, pred).mean()

def cal_MSE(pred, target):
    """
    Mean Squared Error = E[(pred - target) ** 2]
    """
    # TODO: error of frame or error of grid
    return np.mean(((pred - target) ** 2), axis=0) # error of frame

def cal_MAE(pred, target):
    """
    Mean Absolute Error = E[|(pred - target)|]
    """
    # TODO: error of frame or error of grid
    return np.mean(np.abs((pred - target)), axis=0) # error of frame

def cal_RMSE(pred, target):
    """
    Root Mean-Square Error = { E[(pred - target) ** 2] } ** 0.5
    """
    return np.sqrt(cal_MSE(pred, target))

def cal_confusion_matrix(pred, target, th):
    TP = np.sum(np.logical_and(pred>=th, target>=th))
    TN = np.sum(np.logical_and(pred<th, target<th))
    FP = np.sum(np.logical_and(pred>=th, target<th))
    FN = np.sum(np.logical_and(pred<th, target>=th))
    return TP, TN, FP, FN

def cal_performance(pred, target, thresholds):
    """
    Evaluate prediction in different thresholds

    pred: np.array or torch.tensor
    target: np.array or torch.tensor
    ranges: list of numbers
    """
    if not isinstance(thresholds, list):
        thresholds = [thresholds]
    if isinstance(pred, torch.tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.tensor):
        target = target.cpu().numpy()

    metrics = dict()
    for th in thresholds:
        output = dict()
        TP, TN, FP, FN = cal_confusion_matrix(pred, target, th)
        
        output["CSI"] = cal_CSI(TP=TP, FP=FP, FN=FN)
        output["POD"] = cal_POD(TP=TP, FN=FN)
        output["FAR"] = cal_FAR(FP=FP, TP=TP)
        output["CRPS"] = cal_CRPS(pred, target)
        output["MSE"] = cal_MSE(pred, target)
        output["MAE"] = cal_MAE(pred, target)
        output["RMSE"] = cal_RMSE(pred, target)

        metrics["th"] = output

    return metrics