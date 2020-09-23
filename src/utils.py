import torch
import numpy as np

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
