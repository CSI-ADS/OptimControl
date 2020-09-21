import torch
import numpy as np
from .network import adjust_for_external_ownership
from .vitali import *

#
# def shapley(C, control_cutoff):
#

def make_control_cutoff(C, control_cutoff):
    if control_cutoff is None:
        return C
    full_control = C >= control_cutoff
    C_adj = torch.zeros_like(C)
    is_full_control = full_control.sum(axis=1)
    C_adj[:, ~is_full_control] = C[:, ~is_full_control]
    C_adj[full_control] = 1.0
    # rest is zero
    return C_adj

def compute_control_with_external(cl, g, desc_mask=None, weight_control=False, control_cutoff=None):
    #print("tot shares", g.total_shares_in_network)
    C = adjust_for_external_ownership(cl, g, desc_mask=desc_mask)
    cl_resc = compute_total_shares(cl, g, desc_mask=desc_mask)
    cl_adjusted = torch.zeros((g.number_of_nodes,), device=cl_resc.device, dtype=cl_resc.dtype)
    cl_adjusted[desc_mask] = cl_resc # percent of available stock
    if control_cutoff is not None:
        C = make_control_cutoff(C, control_cutoff)
    dtilde = compute_control(cl_adjusted, C, desc=torch.arange(g.number_of_nodes)) # get the control of my node
    if control_cutoff:
        dtilde[dtilde >= control_cutoff] = 1.0
    control = dtilde*g.value if weight_control else dtilde # more control over expensive firms
    return control

def compute_total_shares(cl, g, desc_mask=None):
    shares_in_network = g.total_shares_in_network
    # root nodes are taken to be fully controllable
    shares_in_network[g.identify_uncontrollable()] = 1.0
    if desc_mask is not None:
        assert sum(desc_mask) == cl.shape[0], "mask not matching parameters"
        shares_in_network = shares_in_network[desc_mask]
    return shares_in_network*cl


def compute_control_loss(cl, g, as_array=False, desc_mask=None, weight_control=False, control_cutoff=None): # as low as possible
    """
        D : mask of what has something in what
        g : network containing control matrix
        cl : list of % of stocks that I control in each company
    """
    tot_control = compute_control_with_external(cl, g, desc_mask=desc_mask, weight_control=weight_control, control_cutoff=control_cutoff)
    tot_control = torch.clamp(tot_control, min=0, max=1).flatten() # numerical issues otherwise
    # if not as_array:
    #     tot_control = torch.sum(tot_control)
    assert tot_control.shape[0] == g.number_of_nodes, "below doesn't work"
    if weight_control:
        cost = g.value*(1.0 - tot_control)
    else:
        cost = 1.0 - tot_control

    if as_array:
        return cost
    else:
        return torch.sum(cost) # should sum up tot total number of nodes

def compute_owned_cost(cl, g, as_array=False, desc_mask=None, scale_by_total=True):
    if g.value is not None:
        if desc_mask is None:
            assert g.number_of_nodes == cl.shape[0], "cl should have size of nodes"
        else:
            assert g.number_of_nodes == desc_mask.shape[0], "desc_mask has wrong shape"
            assert cl.shape[0] == sum(desc_mask), "cl has wrong shape"

    owned_network_shares = compute_total_shares(cl, g, desc_mask=desc_mask)

    value = g.value
    if value is None:
        s_cost_per_comp = owned_network_shares
    else: # weighting !!!
        if desc_mask is not None:
            value = value[desc_mask]
        s_cost_per_comp = owned_network_shares*value

    if as_array:
        return s_cost_per_comp.flatten()
    else:
        return s_cost_per_comp.sum()

# def no_shares_cost(cl, g, cutoff=1e-8, desc_mask=None):
#     # if there are no shares available, put the cl to 0 by including a cost of (arbitrary) 1
#     frac_shares_in_network = g.total_shares_in_network
#     if desc_mask is not None:
#         frac_shares_in_network = frac_shares_in_network[desc_mask]
#     no_shares_in_network = (frac_shares_in_network < cutoff)
#     if torch.sum(no_shares_in_network) == 0:
#         return 0
#     cost = cl*no_shares_in_network # bring those to zero
#     return torch.sum(cost)

def compute_sparse_loss(cl, g, lambd=0.1, M=None, as_array=False, as_separate=False, desc_mask=None, weight_control=False, control_cutoff=None):
    assert torch.min(cl) >= 0 and torch.max(cl) <= 1, "strange: {} -- {}".format(torch.min(cl), torch.max(cl))
    if cl.shape[0] != g.number_of_nodes:
        assert desc_mask is not None, "must specify desc_mask when the cl is not the same as the number of nodes"
    c_loss = compute_control_loss(cl, g, as_array=as_array, desc_mask=desc_mask, weight_control=weight_control, control_cutoff=control_cutoff)
    if lambd > 0:
        s_loss = compute_owned_cost(cl, g, as_array=as_array, desc_mask=desc_mask)
    else:
        s_loss = torch.zeros_like(c_loss)

    # s_loss += no_shares_cost(cl, g, cutoff=1e-8, desc_mask=desc_mask)
    if as_separate:
        return c_loss, s_loss
    else:
        return c_loss + lambd*s_loss
#
# def find_lambda_scaling(g):
#     if g.value is None:
#         return 1.
#     total_cost = g.compute_total_value(only_network_shares=True, include_root_shares=True)
#     return total_cost
