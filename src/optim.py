from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm, trange
from .utils import *

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
        device=None, save_params_every=100, save_loss_arr=False,
        save_separate_losses=False,
        loss_1_name="loss_control", loss_2_name="loss_cost",
        loss_tol=1e-8,
        **kwargs):
    params = cl
    g.to(device)
    #print("Optimize for lambd={} on device={} and dtype={}".format(lambd, device, params.dtype))
    optimizer = torch.optim.Adam([{"params" : params}], lr=lr)
    if scheduler is not None:
        scheduler = scheduler(optimizer)
    hist = defaultdict(list)
    i_last = 0
    loss_prev = None
    pbar = tqdm(range(num_steps), disable=not verbose)
    for i in pbar:
        i_last = i
        params, loss = update(loss_fn, optimizer, params, g, lambd=lambd, **kwargs)
        #print("params = ", params)
        #print("loss = ", loss)
        with torch.no_grad():
            hist["loss"].append(loss.detach().cpu().numpy())
            hist["i_iter"].append(i)
            if (i % save_params_every == 0) or (i == num_steps - 1):
                hist["saved_at"].append(i)
                hist["params"].append(params.detach().cpu().numpy())
                hist["params_sm"].append(torch.sigmoid(params).detach().cpu().numpy())

                if save_loss_arr:
                    losses = compute_value(loss_fn, params, g, lambd=lambd, as_separate=True, as_array=True, **kwargs)
                    hist["{}_arr".format(loss_1_name)].append(losses[0].detach().cpu().numpy())
                    hist["{}_arr".format(loss_2_name)].append(losses[1].detach().cpu().numpy())
                if save_separate_losses:
                    losses = compute_value(loss_fn, params, g, lambd=lambd, as_separate=True, as_array=False, **kwargs)
                    hist[loss_1_name].append(losses[0].detach().cpu().numpy())
                    hist[loss_2_name].append(losses[1].detach().cpu().numpy())

        if scheduler is not None:
            scheduler.step()
            hist["lr"].append(scheduler.get_lr()[0])
        else:
            hist["lr"] = lr

        with torch.no_grad():
            if (loss_prev is not None) and (torch.abs(loss - loss_prev) < loss_tol):
                if verbose:
                    print("breaking optimization at step {} with loss: {}".format(i, loss))
                break
            else:
                loss_prev = loss

            pbar.set_postfix({'loss': loss.detach().cpu().numpy()})


    with torch.no_grad():
        hist["final_iter"] = i_last
        hist["final_params_sm"] = torch.sigmoid(params).detach().cpu().numpy()
        losses = compute_value(loss_fn, params, g, lambd=lambd, as_separate=True, as_array=True, **kwargs)
        hist["final_{}_arr".format(loss_1_name)] = losses[0].detach().cpu().numpy()
        hist["final_{}_arr".format(loss_2_name)] = losses[1].detach().cpu().numpy()
        losses = compute_value(loss_fn, params, g, lambd=lambd, as_separate=True, as_array=False, **kwargs)
        hist["final_{}".format(loss_1_name)]= losses[0].detach().cpu().numpy()
        hist["final_{}".format(loss_2_name)] = losses[1].detach().cpu().numpy()
    ret = (params, loss)
    if return_hist:
        ret += (dict(hist), )
    return ret


def constraint_optimize_control(
        loss_fns, cl, g, budget,
        verbose=False, return_hist=False,
        lr=0.1, scheduler=None,
        max_iter=100, num_steps=10000,
        device=None, save_params_every=100, save_loss_arr=False,
        constr_tol = 1e-8,
        loss_tol = 1e-8,
        loss_1_name="loss_control", loss_2_name="loss_cost",
        **kwargs
        ):
    params = cl
    rho, alpha, constr, constr_new = 1.0, 0.0, float("Inf"), float("Inf")
    flag_max_iter = True

    hist = defaultdict(list)

    step_nr = -1

    for i in range(max_iter):
        while rho < 1e+20:
            # optimize the actual loss
            step_nr += 1
            def augm_loss(*loss_args, as_separate=False, **loss_kwargs):
                l, c = loss_fns(*loss_args, as_separate=True, **loss_kwargs)
                if loss_kwargs.get("as_array", False):
                    sm, tm = loss_kwargs.get("source_mask", None), loss_kwargs.get("target_mask", None)
                    if sm is not None:
                        l = pad_from_mask(l, tm, ttype='torch')
                    if tm is not None:
                        c = pad_from_mask(c, sm, ttype='torch')
                c_l = c - budget
                augm_l = l + 0.5 * rho * c_l**2 + alpha * c_l # augmented lagrangian
                # print("terms:", l.detach().cpu().numpy(), (0.5*rho*c**2).detach().cpu().numpy(), (alpha*c).detach().cpu().numpy())
                # print("c = ", c)
                if as_separate:
                    return augm_l, c_l
                else:
                    return augm_l

            params, augm_new, hist_new = optimize_control(augm_loss, params, g,
                    lambd=1,
                    verbose=verbose, return_hist=True,
                    lr=lr, scheduler=scheduler, num_steps=num_steps,
                    device=device, save_params_every=save_params_every, save_loss_arr=save_loss_arr,
                    loss_1_name="loss_augm", loss_2_name="loss_costr",
                    loss_tol=loss_tol,
                    **kwargs
                    )

            loss_new = hist_new["final_loss_augm"]
            constr_new = hist_new["final_loss_costr"]

            with torch.no_grad():
                # print(kwargs)
                losses_orig_new = compute_value(loss_fns, params, g, lambd=1, as_separate=True, as_array=False, **kwargs)


            hist[loss_1_name].append(losses_orig_new[0].detach().cpu().numpy())
            hist[loss_2_name].append(losses_orig_new[1].detach().cpu().numpy())
            hist["loss_augm"].append(loss_new)
            hist["i_contr_iter"].append(i)
            hist["step_nr"].append(step_nr)
            hist["rho"].append(rho)
            hist["alpha"].append(alpha)
            hist["constr"].append(constr_new)
            hist["tot_cost"].append(constr_new+budget)
            hist["final_iter"].append(hist_new["final_iter"])

            hist["hist_optim"].append(hist_new)
            # print("iter:",hist["final_iter"][-1], " augm:", loss_new, "constr_new:", constr_new, "constr:", constr, "tot_cost:", hist["tot_cost"][-1])
            # print("loss_control:", hist["loss_control"][-1], "loss_cost:", hist["loss_cost"][-1])
            if np.abs(constr_new) > 0.25 * np.abs(constr):
                rho *= 10
                print("Increasing rho to: 10**", np.log10(rho))
            else:
                print("Break: ", constr_new, constr)
                break

        constr = constr_new
        alpha += rho * constr
        if np.abs(constr) <= constr_tol:
            flag_max_iter = False
            break


    with torch.no_grad():
        hist["final_iter"] = step_nr
        hist["final_params_sm"] = torch.sigmoid(params).detach().cpu().numpy()
        losses = compute_value(loss_fns, params, g, lambd=1, as_separate=True, as_array=True, **kwargs)
        hist["final_{}_arr".format(loss_1_name)]= losses[0].detach().cpu().numpy()
        hist["final_{}_arr".format(loss_2_name)] = losses[1].detach().cpu().numpy()
        losses = compute_value(loss_fns, params, g, lambd=1, as_separate=True, as_array=False, **kwargs)
        hist["final_{}".format(loss_1_name)]= losses[0].detach().cpu().numpy()
        hist["final_{}".format(loss_2_name)] = losses[1].detach().cpu().numpy()
        ret = (params, losses, constr)
        if return_hist:
            ret += (dict(hist), )
        return ret



    return params_est, constr, hist
