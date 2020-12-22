import torch
import numpy as np
from .vitali import *
from .utils import *
#
# def shapley(C, control_cutoff):
#
def adjust_for_external_ownership(ol, g):
    assert torch.min(ol) >= 0 and torch.max(ol) <= 1, "ol was outside bounds"
    C = g.ownership.clone()
    non_root_nodes = g.identify_controllable() # only non-roots should be adjusted
    tot_shares = reduce_from_mask(g.total_shares_in_network, non_root_nodes)
    perc_avail_shares = torch.clamp(reduce_from_mask(ol, non_root_nodes) / tot_shares, min=0, max=1) # to be certain we clip
    C[:,non_root_nodes] = C[:,non_root_nodes] * (1.0 - perc_avail_shares) # remains the same if we don't own (ol=0)
    return C

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

def compute_control_with_external(ol, g, source_mask=None, target_mask=None, weight_control=False, control_cutoff=None):

    C = adjust_for_external_ownership(ol, g)

    if control_cutoff is not None:
        C = make_control_cutoff(C, control_cutoff)

    #core
    control = compute_control(ol, C, reach=torch.arange(g.number_of_nodes)) # get the control of my node
    assert control.shape[0] == g.number_of_nodes
    if control_cutoff:
        control[control >= control_cutoff] = 1.0

    if weight_control:
        value = g.value
        assert value is not None, "cannot weight if we don't get value"
        control *= value

    if target_mask is not None:
        control = reduce_from_mask(control, target_mask)

    return control

def compute_control_loss(ol, g, as_array=False, source_mask=None, target_mask=None, weight_control=False, control_cutoff=None): # as low as possible
    """
        D : mask of what has something in what
        g : network containing control matrix
        ol : list of % of stocks that I control in each company
    """
    assert ol.shape[0] == g.number_of_nodes
    if source_mask is not None:
        assert source_mask.shape[0] == g.number_of_nodes, "source mask has incorrect shape"
    if target_mask is not None:
        assert target_mask.shape[0] == g.number_of_nodes, "target mask has incorrect shape"

    tot_control = compute_control_with_external(ol, g, source_mask=source_mask, target_mask=target_mask, weight_control=weight_control, control_cutoff=control_cutoff)
    tot_control = torch.clamp(tot_control, min=0, max=1).flatten() # numerical issues otherwise, shape is target set
    if target_mask is not None:
        assert tot_control.shape[0] == sum(target_mask)
    else:
        assert tot_control.shape[0] == g.number_of_nodes

    if weight_control:
        value = g.value
        assert value is not None, "value cannot be none if we need to weight with it"
        value = reduce_from_mask(value, source_mask)
        cost = value*(1.0 - tot_control)
    else:
        cost = 1.0 - tot_control

    if as_array:
        return cost
    else:
        return torch.sum(cost) # should sum up tot total number of nodes

def compute_owned_cost(ol, g, as_array=False, source_mask=None, scale_by_total=True):
    assert ol.shape[0] == g.number_of_nodes
    if source_mask is not None:
        assert source_mask.shape[0] == g.number_of_nodes

    value = g.value
    s_cost_per_comp = reduce_from_mask(ol, source_mask)

    if g.value is not None: # then weight
        value = reduce_from_mask(value, source_mask)
        s_cost_per_comp *= value

    if as_array:
        return s_cost_per_comp.flatten()
    else:
        return s_cost_per_comp.sum()

def compute_sparse_loss(
        ol, g, lambd=1,
        as_array=False, as_separate=False,
        source_mask=None, target_mask=None,
        weight_control=False, control_cutoff=None):

    c_loss = compute_control_loss(
        ol, g,
        as_array=as_array,
        source_mask=source_mask, target_mask=target_mask,
        weight_control=weight_control, control_cutoff=control_cutoff
        )
    s_loss = compute_owned_cost(ol, g, as_array=as_array, source_mask=source_mask)

    # s_loss += no_shares_cost(ol, g, cutoff=1e-8, source_mask=source_mask)
    if as_separate:
        return c_loss, s_loss
    else:
        return c_loss + lambd*s_loss
