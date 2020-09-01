import torch
import numpy as np
from .network import *
from .vitali import *

def compute_control_with_external(cl, g):
    cl_adjusted = cl*g.total_shares_in_network
    C = adjust_for_external_ownership(cl, g)
    dtilde = compute_control(cl_adjusted, C) # get the control of my node
    return dtilde

def compute_control_loss(cl, g): # as low as possible
    """
        D : mask of what has something in what
        g : network containing control matrix
        cl : list of % of stocks that I control in each company
    """
    dtilde = compute_control_with_external(cl, g)
    dtilde = torch.clamp(dtilde, min=0, max=1) # numerical issues otherwise
    tot_control = torch.sum(dtilde)
    return g.number_of_nodes - tot_control

def compute_owned_cost(cl, g, as_array=False):
    if g.value is not None:
        assert g.number_of_nodes == cl.shape, "cl should have size of nodes"
    frac_shares_in_network = g.total_shares_in_network
    owned_network_shares = cl*frac_shares_in_network

    if g.value is None:
        s_cost_per_comp = owned_network_shares
    else: # weighting !!!
        s_cost_per_comp = owned_network_shares*g.value

    if as_array:
        return s_cost_per_comp
    else:
        return s_cost_per_comp.sum()

def no_shares_cost(cl, g, cutoff=1e-8):
    # if there are no shares available, put the cl to 0 by including a cost of (arbitrary) 1
    frac_shares_in_network = g.total_shares_in_network
    no_shares_in_network = (frac_shares_in_network < cutoff)
    cost = cl*no_shares_in_network # bring those to zero
    return torch.sum(cost)

def compute_sparse_loss(cl, g, lambd=0.1, M=None, as_separate=False):
    assert torch.min(cl) >= 0 and torch.max(cl) <= 1, "strange"
    c_loss = compute_control_loss(cl, g)
    s_loss = compute_owned_cost(cl, g, as_array=False)
    s_loss += no_shares_cost(cl, g, cutoff=1e-8)
    if as_separate:
        return c_loss, s_loss
    else:
        return c_loss + lambd*s_loss

def compute_sparse_loss_cache_vars(cl, g, lambd=0.1, as_separate=False):
    cl = torch.sigmoid(cl) # to keep numbers [0, 1] !!!
    return compute_sparse_loss(cl, g, lambd=lambd, as_separate=as_separate)
