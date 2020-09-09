from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm

def get_cl_init(N, loc=-6, scale=0.1, device=None, dtype=torch.float):
    cl_normal = torch.normal(mean=loc, std=0.1, size=(N,))
    #cl[0] = 0
    cl = torch.tensor(cl_normal.clone().detach(), requires_grad=True, device=device, dtype=dtype)
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

def optimize_control(loss_fn, cl, g, lambd=0, verbose=False, return_hist=False, lr=0.1, num_steps=10000, device=None, save_params_every=100, **kwargs):
    params = cl
    g.to(device)
    #print("Optimize for lambd={} on device={} and dtype={}".format(lambd, device, params.dtype))
    optimizer = torch.optim.Adam([{"params" : params}], lr=lr)
    hist = defaultdict(list)
    for i in tqdm(range(num_steps), disable=not verbose):
        params, loss = update(loss_fn, optimizer, params, g, lambd=lambd, **kwargs)
        #print("params = ", params)
        #print("loss = ", loss)
        with torch.no_grad():
            hist["loss"].append(loss.detach().cpu().numpy())
            if i % save_params_every == 0:
                hist["params"].append(params.detach().cpu().numpy())
                hist["params_sm"].append(torch.sigmoid(params).detach().cpu().numpy())
    with torch.no_grad():
        hist["final_params_sm"] = torch.sigmoid(params).detach().cpu().numpy()
    ret = (params, loss)
    if return_hist:
        ret += (hist, )
    return ret
