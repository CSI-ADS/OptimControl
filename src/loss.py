import torch
import numpy as np
from .network import adjust_for_external_ownership
from .vitali import *

def compute_control_with_external(cl, g, desc_mask=None):
    #print("tot shares", g.total_shares_in_network)
    shares_in_network = g.total_shares_in_network
    # root nodes are taken to be fully controllable
    shares_in_network[g.identify_uncontrollable()] = 1.0

    if desc_mask is not None:
        assert sum(desc_mask) == cl.shape[0], "mask not matching parameters"
        shares_in_network = shares_in_network[desc_mask]

    C = adjust_for_external_ownership(cl, g, desc_mask=desc_mask)

    cl_adjusted = torch.zeros((g.number_of_nodes,), device=cl.device, dtype=cl.dtype)
    cl_adjusted[desc_mask] = cl*shares_in_network # percent of available stock
    dtilde = compute_control(cl_adjusted, C, desc=torch.arange(g.number_of_nodes)) # get the control of my node
    return dtilde

def compute_control_loss(cl, g, desc_mask=None): # as low as possible
    """
        D : mask of what has something in what
        g : network containing control matrix
        cl : list of % of stocks that I control in each company
    """
    dtilde = compute_control_with_external(cl, g, desc_mask=desc_mask)
    dtilde = torch.clamp(dtilde, min=0, max=1) # numerical issues otherwise
    tot_control = torch.sum(dtilde)
    return g.number_of_nodes - tot_control

def compute_owned_cost(cl, g, as_array=False, desc_mask=None):
    if g.value is not None:
        if desc_mask is None:
            assert g.number_of_nodes == cl.shape[0], "cl should have size of nodes"
        else:
            assert g.number_of_nodes == desc_mask.shape[0], "desc_mask has wrong shape"
            assert cl.shape[0] == sum(desc_mask), "cl has wrong shape"
    frac_shares_in_network = g.total_shares_in_network
    if desc_mask is not None:
        frac_shares_in_network = frac_shares_in_network[desc_mask]
    owned_network_shares = cl*frac_shares_in_network

    if g.value is None:
        s_cost_per_comp = owned_network_shares
    else: # weighting !!!
        value = g.value
        if desc_mask is not None:
            value = value[desc_mask]
        s_cost_per_comp = owned_network_shares*value

    if as_array:
        return s_cost_per_comp
    else:
        return s_cost_per_comp.sum()

def no_shares_cost(cl, g, cutoff=1e-8, desc_mask=None):
    # if there are no shares available, put the cl to 0 by including a cost of (arbitrary) 1
    frac_shares_in_network = g.total_shares_in_network
    if desc_mask is not None:
        frac_shares_in_network = frac_shares_in_network[desc_mask]
    no_shares_in_network = (frac_shares_in_network < cutoff)
    if torch.sum(no_shares_in_network) == 0:
        return 0
    cost = cl*no_shares_in_network # bring those to zero
    return torch.sum(cost)

def compute_sparse_loss(cl, g, lambd=0.1, M=None, as_separate=False, desc_mask=None):
    assert torch.min(cl) >= 0 and torch.max(cl) <= 1, "strange"
    if cl.shape[0] != g.number_of_nodes:
        assert desc_mask is not None, "must specify desc_mask when the cl is not the same as the number of nodes"
    c_loss = compute_control_loss(cl, g, desc_mask=desc_mask)
    s_loss = compute_owned_cost(cl, g, as_array=False, desc_mask=desc_mask)
    s_loss += no_shares_cost(cl, g, cutoff=1e-8, desc_mask=desc_mask)
    if as_separate:
        return c_loss, s_loss
    else:
        return c_loss + lambd*s_loss

def compute_sparse_loss_cache_vars(cl, g, lambd=0.1, as_separate=False, desc_mask=None):
    cl = torch.sigmoid(cl) # to keep numbers [0, 1] !!!
    return compute_sparse_loss(cl, g, lambd=lambd, as_separate=as_separate, desc_mask=desc_mask)
