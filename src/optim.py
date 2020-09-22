from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm

def get_cl_init(N, loc=-7, scale=1e-4, device=None, dtype=torch.float):
    cl_normal = torch.normal(mean=loc, std=0.1, size=(N,))
    #cl[0] = 0
    cl = torch.tensor(cl_normal.clone().detach(), requires_grad=True, device=device, dtype=dtype)
    return cl

def cutoff_values(cl, cutoff=1e-8):
    #cl[cl < cutoff] = 0.0
    #cl[cl > 1-cutoff] = 1.0
    cl_cut = torch.zeros_like(cl)
    cl_cut[cl > cutoff] = cl[cl > cutoff]
    cl_cut[cl > 1-cutoff] = cl[cl > 1-cutoff]
    return cl_cut

def compute_value(fn, cl, *args, **kwargs):
    cl_soft = torch.sigmoid(cl) # !
    # cl_soft = cutoff_values(cl_soft)
    return fn(cl_soft, *args, **kwargs)

def compute_value_and_grad(fn, cl, *args, **kwargs):
    value = compute_value(fn, cl, *args, **kwargs)
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

def optimize_control(
        loss_fn, cl, g,
        lambd=0,
        verbose=False, return_hist=False,
        lr=0.1, scheduler=None, num_steps=10000,
        device=None, save_params_every=100, save_loss_arr=False, **kwargs):
    params = cl
    g.to(device)
    #print("Optimize for lambd={} on device={} and dtype={}".format(lambd, device, params.dtype))
    optimizer = torch.optim.Adam([{"params" : params}], lr=lr)
    if scheduler is not None:
        scheduler = scheduler(optimizer)
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

                if save_loss_arr:
                    losses = compute_value(loss_fn, params, g, lambd=lambd, as_separate=True, as_array=True, **kwargs)
                    hist["loss_control"].append(losses[0].detach().cpu().numpy())
                    hist["loss_cost"].append(losses[1].detach().cpu().numpy())
        if scheduler is not None:
            hist["lr"].append(scheduler.get_lr())
            scheduler.step()

    with torch.no_grad():
        hist["final_params_sm"] = torch.sigmoid(params).detach().cpu().numpy()
        losses = compute_value(loss_fn, params, g, lambd=lambd, as_separate=True, as_array=True, **kwargs)
        hist["final_loss_control"]= losses[0].detach().cpu().numpy()
        hist["final_loss_cost"] = losses[1].detach().cpu().numpy()
    ret = (params, loss)
    if return_hist:
        ret += (hist, )
    return ret


def constraint_optimize_control(
        loss_fns, cl, g, budget,
        verbose=False, return_hist=False,
        lr=0.1, scheduler=None,
        max_iter=100, num_steps=10000,
        device=None, save_params_every=100, save_loss_arr=False,
        constr_tol = 1e-8,
        loss_thr = 0.3,
        **kwargs
        ):
    params = cl
    rho, alpha, constr, constr_new = 1.0, 0.0, float("Inf"), float("Inf")
    flag_max_iter = True

    hist = []

    for i in range(max_iter):
        while rho < 1e+20:
            # optimize the actual loss

            def augm_loss(*args, **kwargs):
                if "as_separate" in kwargs:
                    kwargs.pop("as_separate")
                l, c = loss_fns(*args, as_separate=True, **kwargs)
                return l + 0.5 * rho * c**2 + alpha * c # augmented lagrangian

            params_new, augm_new, hist_new = optimize_control(augm_loss, params, g,
                    lambd=0,
                    verbose=False, return_hist=True,
                    lr=lr, scheduler=scheduler, num_steps=num_steps,
                    device=device, save_params_every=save_params_every, save_loss_arr=save_loss_arr
                    )
            loss_new = hist_new["final_loss_control"]
            constr_new = hist_new["final_loss_cost"]

            hist.append(hist_new)

            with torch.no_grad():
                if constr_new > 0.25 * constr:
                    rho *= 10
                else:
                    break

        params_est, constr = params_new, constr_new
        alpha += rho * constr
        if constr <= constr_tol:
            flag_max_iter = False
            break
        return params_est, constr, hist_new
