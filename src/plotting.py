import numpy as np
import matplotlib.pyplot as plt
from .utils import *
from .optim import compute_total_shares
import copy

def plot_direct_control(
        g,
        ol,
        source_mask=None,
        target_mask=None,
        figsize=(20, 5),
        total_shares_in_network=None,
        filename=None
        ):
    nodelist = g.node_list.detach().cpu().numpy()
    if source_mask is not None:
        source_mask = source_mask.detach().cpu().numpy()
        nodelist = nodelist[source_mask]
    is_target = np.array([True]*ol.shape[0])
    if target_mask is not None:
        is_target = target_mask.detach().cpu().numpy()

    ol_plot = copy.copy(ol)
    total_shares_in_network_plot = copy.copy(total_shares_in_network)
    if source_mask is not None:
        ol_plot = ol_plot[source_mask]
        is_target = is_target[source_mask]
        if total_shares_in_network is not None:
            total_shares_in_network_plot = total_shares_in_network_plot[source_mask]
    idx = np.argsort(ol_plot)[::-1]
    node_names = np.array(["{}".format(x) for x in nodelist[idx]], dtype=object)
    print(total_shares_in_network_plot)
    ol_plot = ol_plot[idx]
    is_target = is_target[idx]
    if total_shares_in_network is not None:
        total_shares_in_network_plot = total_shares_in_network_plot[idx]
    plt.figure(figsize=figsize)
    plt.bar(node_names, ol_plot, color='blue', label=r'total shares bought $\mathbf{o}$')
    if total_shares_in_network is not None:
        plt.bar(node_names, total_shares_in_network_plot, label='available shares', color='black', fill=False)
    if target_mask is not None:
        plt.bar(node_names[is_target], ol_plot[is_target], label='target', color='red')
    plt.xlabel("company name")
    plt.ylabel("shares")
    plt.xticks(np.arange(len(node_names)), node_names, rotation='vertical')
    plt.title(r"$\mathbf{o} \in \mathcal{S}$")
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def plot_control_distr(
        g,
        loss_control_arr,
        ol=None,
        source_mask=None,
        target_mask=None,
        cutoff=None,
        figsize=(20, 5),
        total_shares_in_network=None,
        filename=None
        ):

    if cutoff is None:
        cutoff = -np.inf

    nodelist = g.node_list.detach().cpu().numpy()
    if target_mask is not None:
        target_mask = target_mask.detach().cpu().numpy()
        nodelist = nodelist[target_mask]
    if source_mask is not None:
        source_mask = source_mask.detach().cpu().numpy()

    plt.figure(figsize=figsize)
    idx = np.argsort(1-loss_control_arr)[::-1]
    idx = [i for i in idx if 1-loss_control_arr[i] >= cutoff]
    node_names = ["{}".format(x) for x in g.node_list.detach().cpu().numpy()[idx]]
    yval = 1-loss_control_arr[idx]
    plt.bar(node_names, yval, color='blue', label="total")
    if ol is not None:
        mask = np.ones((g.number_of_nodes), dtype=bool)
        if target_mask is not None:
            mask *= target_mask
        if source_mask is not None:
            mask *= source_mask
        ol_plot = reduce_from_mask(ol*mask, target_mask)
        plt.bar(node_names, ol_plot[idx], color='red', label='direct')
    if total_shares_in_network is not None:
        total_shares_in_network_plot = reduce_from_mask(total_shares_in_network*mask, target_mask)
        plt.bar(node_names, total_shares_in_network_plot[idx], color='black', label='available shares', fill=False)
    plt.xlabel("company name")
    plt.ylabel("control $c$")
    plt.xticks(np.arange(len(node_names)), node_names, rotation='vertical')
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    plt.show()
