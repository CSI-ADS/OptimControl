from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm

def get_cl_init(N, loc=-6, scale=0.1, device=None):
    cl = np.random.normal(loc=loc, scale=0.1, size=N)
    #cl[0] = 0
    cl = torch.tensor(cl, requires_grad=True, device=device)
    return cl

def compute_value_and_grad(fn, cl, *args, **kwargs):
    value = fn(cl, *args, **kwargs)
    value.backward()
    grads = cl.grad
    return value, grads


def get_params(optimizer):
    for group_param in optimizer.param_groups:
        for param in group_param["params"]:
            return param # just one

def update(loss_fn, optimizer, params, *args, **kwargs):
    optimizer.zero_grad()
    params = get_params(optimizer)
    cost, grads = compute_value_and_grad(loss_fn, params, *args, **kwargs)
    optimizer.step()
    return params, cost

def optimize_control(loss_fn, cl, g, lambd=0, verbose=False, return_hist=False, lr=0.1, num_steps=10000):
    params = cl
    optimizer = torch.optim.Adam([{"params" : params}], lr=lr)
    hist = defaultdict(list)
    for i in tqdm(range(num_steps), disable=not verbose):
        params, loss = update(loss_fn, optimizer, params, g, lambd=lambd)
        #print("params = ", params)
        #print("loss = ", loss)
        with torch.no_grad():
            hist["loss"].append(loss.cpu().numpy())
            if i % 10 == 0:
                hist["params"].append(params.cpu().numpy())
                hist["params_sm"].append(torch.sigmoid(params).cpu().numpy())
    with torch.no_grad():
        hist["final_params_sm"] = torch.sigmoid(params).cpu().numpy()
    ret = (params, loss)
    if return_hist:
        ret += (hist, )
    return ret
