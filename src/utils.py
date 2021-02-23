import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

def my_savefig(filename, **kwargs):
    print("saving figure to:", filename)
    fig_object = plt.gcf()
    filename_pickle = "{}.pickle".format(filename)
    with open(filename_pickle, 'wb') as f:
        pickle.dump(fig_object, f)
    plt.savefig(filename, **kwargs)

def reduce_from_mask(arr, mask):
    if mask is None:
        return arr
    assert len(arr) == len(mask), "{} {}".format(len(arr), sum(mask))
    return arr[mask]

def pad_from_mask(arr, mask, ttype='np'):
    if mask is None:
        return arr
    assert len(arr) == sum(mask), "{} {}".format(len(arr), sum(mask))
    if ttype == 'np':
        t = np.zeros(len(mask))
    elif ttype == 'torch':
        t = torch.zeros(len(mask), device=arr.device, dtype=arr.dtype)
    t[mask] = arr
    return t

def idx_to_mask(N, idx):
    mask = torch.zeros((N,), dtype=bool)
    mask[idx] = True
    return mask
