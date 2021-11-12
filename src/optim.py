from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm, trange
from .utils import *
from .loss import compute_control_from_loss
import time

def get_cl_init(N, loc=-7, scale=1e-4, vals=None, device=None):
    if vals is not None:
        assert N == len(vals)
        cl_normal = torch.tensor(vals)
    else:
        cl_normal = torch.normal(mean=loc, std=scale, size=(N,))

    #cl[0] = 0
    cl = cl_normal.clone().detach().requires_grad_(True)
    cl = cl.to(device)
    #cl = torch.tensor(cl_normal.clone().detach(), requires_grad=True, device=device, dtype=dtype)
    return cl

def compute_total_shares(cl, g, source_mask=None): # cl is |S|, while ol is N

    cl_soft = torch.sigmoid(cl) # !

    shares_in_network = g.total_shares_in_network
    # root nodes are taken to be fully controllable
    shares_in_network[g.identify_uncontrollable()] = 1.0
    if source_mask is not None:
        assert sum(source_mask) == cl.shape[0], "mask not matching parameters"
        shares_in_network = shares_in_network[source_mask] # mask size S


    # core
    ol = shares_in_network*cl_soft # ol is 0 for no external ownership!
    assert torch.min(ol) >= 0 and torch.max(ol) <= 1, "strange: {} -- {}".format(torch.min(ol), torch.max(ol))

    # size S to N
    ol = pad_from_mask(ol, source_mask, ttype='torch')
    assert ol.shape[0] == g.number_of_nodes, "should be N size"

    return ol

def compute_value(fn, cl, g, *args, **kwargs):
    ol = compute_total_shares(cl, g, source_mask=kwargs.get("source_mask"))
    return fn(ol, g, *args, **kwargs)

def compute_value_and_grad(fn, cl, g, *args, **kwargs):
    value = compute_value(fn, cl, g, *args, **kwargs)
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
        es_wait=1,
        **kwargs):
    params = cl
    g.to(device)
    #print("Optimize for lambd={} on device={} and dtype={}".format(lambd, device, params.dtype))
    optimizer = torch.optim.Adam([{"params" : params}], lr=lr)
    if scheduler is not None:
        scheduler = scheduler(optimizer)
    hist = defaultdict(list)
    target_mask = kwargs.get("target_mask", None)
    source_mask = kwargs.get("source_mask", None)
    i_last = 0
    loss_prev = None
    pbar = tqdm(range(num_steps), disable=not verbose)
    n_wait = 0
    start = time.time()
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
                hist["ol"].append(compute_total_shares(cl, g, source_mask=source_mask).detach().cpu().numpy())

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
                n_wait += 1
                if n_wait >= es_wait:
                    if verbose:
                        print("breaking optimization at step {} with loss: {}".format(i, loss))
                    break
                else:
                    if verbose:
                        print("waiting, but no improvement: {}".format(n_wait))
            else:
                n_wait = 0 # reset, must be subsequent
            loss_prev = loss
            pbar.set_postfix({'loss': loss.detach().cpu().numpy()})
    end = time.time()

    with torch.no_grad():
        print("computing hist info")
        hist["time"] = end - start  
        hist["final_iter"] = i_last
        hist["final_params"] = params.detach().cpu().numpy()
        hist["final_params_sm"] = torch.sigmoid(params).detach().cpu().numpy()
        hist["final_ol"] = compute_total_shares(cl, g, source_mask=source_mask).detach().cpu().numpy()
        hist["total_shares_in_network"] = g.total_shares_in_network.detach().cpu().numpy()
        losses = compute_value(loss_fn, params, g, lambd=lambd, as_separate=True, as_array=True, **kwargs)
        hist["final_{}_arr".format(loss_1_name)] = losses[0].detach().cpu().numpy()
        hist["final_{}_arr".format(loss_2_name)] = losses[1].detach().cpu().numpy()
        losses = compute_value(loss_fn, params, g, lambd=lambd, as_separate=True, as_array=False, **kwargs)
        hist["final_{}".format(loss_1_name)]= losses[0].detach().cpu().numpy()
        hist["final_{}".format(loss_2_name)] = losses[1].detach().cpu().numpy()
        hist["final_control"] = compute_control_from_loss(losses[0].detach().cpu().numpy(), g, target_mask)
        hist["final_control_shares"] = compute_control_from_loss(losses[0].detach().cpu().numpy(), g, target_mask, normalize='shares')
        hist["final_control_nodes"] = compute_control_from_loss(losses[0].detach().cpu().numpy(), g, target_mask, normalize='nodes')
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
        es_wait=1,
        **kwargs
        ):
    params = cl
    rho, alpha, constr, constr_new = 1.0, 0.0, float("Inf"), float("Inf")
    flag_max_iter = True

    hist = defaultdict(list)

    step_nr = -1

    for i in range(max_iter):
        print("Starting with iter i={}".format(i))
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
                #h_constr = torch.abs(c - budget)
                h_constr = c - budget # can only be smaller than or equal to zero
                # if c > budget, which is not allowed: h_constr is positive, and only then contribute to loss
                h_constr = torch.clamp(h_constr/budget, min=0)
                h_constr = torch.abs(h_constr) # just a check
                augm_lagr = l + 0.5 * rho * h_constr**2 + alpha * h_constr # augmented lagrangian
                # print("terms:", l.detach().cpu().numpy(), (0.5*rho*c**2).detach().cpu().numpy(), (alpha*c).detach().cpu().numpy())
                # print("c = ", c)
                #print("l, h:", l, h_constr)
                if as_separate:
                    return augm_lagr, h_constr#*0
                else:
                    return augm_lagr

            params, augm_new, hist_new = optimize_control(augm_loss, params, g,
                    lambd=1,
                    verbose=verbose, return_hist=True,
                    lr=lr if step_nr == 0 else lr/10, scheduler=scheduler, num_steps=num_steps,
                    device=device, save_params_every=save_params_every, save_loss_arr=save_loss_arr,
                    loss_1_name="loss_augm", loss_2_name="loss_costr",
                    loss_tol=loss_tol,
                    es_wait=es_wait,
                    **kwargs
                    )

            loss_new = hist_new["final_loss_augm"]
            constr_new = hist_new["final_loss_costr"]

            with torch.no_grad():
                # print(kwargs)
                losses_orig_new = compute_value(loss_fns, params, g, lambd=1, as_separate=True, as_array=False, **kwargs)


            hist[loss_1_name].append(losses_orig_new[0].detach().cpu().numpy())
            hist[loss_2_name].append(losses_orig_new[1].detach().cpu().numpy())
            print("current required value (budget={}): {}".format(budget, losses_orig_new[1]))
            hist["loss_augm"].append(loss_new)
            hist["i_contr_iter"].append(i)
            hist["step_nr"].append(step_nr)
            hist["rho"].append(rho)
            hist["alpha"].append(alpha)
            hist["constr"].append(constr_new)
            hist["tot_cost"].append(constr_new+budget)
            hist["final_iter"].append(hist_new["final_iter"])
            hist["total_shares_in_network"] = hist_new["total_shares_in_network"]
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
        alpha += np.abs(rho * constr) # just to be sure, should go up!
        if np.abs(constr) <= constr_tol:
            flag_max_iter = False
            break


    with torch.no_grad():
        print("computing final hist info")
        hist["final_iter"] = step_nr
        hist["final_params"] = params.detach().cpu().numpy()
        hist["final_params_sm"] = torch.sigmoid(params).detach().cpu().numpy()
        hist["final_ol"] = compute_total_shares(params, g, source_mask=kwargs.get("source_mask", None)).detach().cpu().numpy()
        losses = compute_value(loss_fns, params, g, lambd=1, as_separate=True, as_array=True, **kwargs)
        hist["final_{}_arr".format(loss_1_name)]= losses[0].detach().cpu().numpy()
        hist["final_{}_arr".format(loss_2_name)] = losses[1].detach().cpu().numpy()
        losses = compute_value(loss_fns, params, g, lambd=1, as_separate=True, as_array=False, **kwargs)
        hist["final_{}".format(loss_1_name)]= losses[0].detach().cpu().numpy()
        hist["final_{}".format(loss_2_name)] = losses[1].detach().cpu().numpy()
        hist["final_control"] = hist_new["final_control"]
        hist["final_control_shares"] = hist_new["final_control_shares"]
        hist["final_control_nodes"] = hist_new["final_control_nodes"]
        ret = (params, losses, constr)
        if return_hist:
            ret += (dict(hist), )
        return ret



    return params_est, constr, hist
