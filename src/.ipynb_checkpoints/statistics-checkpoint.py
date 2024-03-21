import numpy as np
from math import sqrt
from scipy import stats
from lifelines.utils import concordance_index as ci
from sklearn.metrics import roc_auc_score, average_precision_score


def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def optimized_ci(y, f):
    return ci(y, f)


'''
AUROC and AUPR are specific metrics for Davis
'''


def auroc(G, P):
    threshold = 5
    binary_real = (G > threshold).astype(int)
    binary_pred = P - 5
    return roc_auc_score(binary_real, binary_pred)


def aupr(G, P):
    threshold = 5
    binary_real = (G > threshold).astype(int)
    binary_pred = P - 5
    return average_precision_score(binary_real, binary_pred)
