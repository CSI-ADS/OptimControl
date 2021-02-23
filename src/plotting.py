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
        filename=None,
        max_show=100
        ):
    nodelist = g.node_list.detach().cpu().numpy()
    if source_mask is not None:
        source_mask = source_mask.detach().cpu().numpy()
    is_target = np.array([True]*ol.shape[0])
    if target_mask is not None:
        is_target = target_mask.detach().cpu().numpy()

    ol_plot = copy.copy(ol)
    total_shares_in_network_plot = copy.copy(total_shares_in_network)
    total_shares_in_network_plot[g.identify_uncontrollable()] = 1.0
    total_value_plot = copy.copy(g.value.detach().cpu().numpy())
    if source_mask is not None:
        nodelist = nodelist[source_mask]
        ol_plot = ol_plot[source_mask]
        is_target = is_target[source_mask]
        total_value_plot = total_value_plot[source_mask]
        if total_shares_in_network is not None:
            total_shares_in_network_plot = total_shares_in_network_plot[source_mask]
    idx = np.argsort(ol_plot)[::-1]
    node_names = np.array(["{}".format(x) for x in nodelist[idx]], dtype=object)
    ol_plot = ol_plot[idx]
    is_target = is_target[idx]
    total_value_plot = total_value_plot[idx]
    if total_shares_in_network is not None:
        total_shares_in_network_plot = total_shares_in_network_plot[idx]


    fig, axes = plt.subplots(figsize=figsize, nrows=2, ncols=1, sharex=True)
    fig.subplots_adjust(hspace=0.0)
    axes[1].bar(node_names[:max_show], ol_plot[:max_show], color='blue', label=r'total shares bought $\mathbf{o}$')
    if total_shares_in_network is not None:
        axes[1].bar(node_names[:max_show], total_shares_in_network_plot[:max_show], label='available shares', color='black', fill=False)
    if target_mask is not None:
        axes[1].bar(node_names[:max_show], ol_plot[:max_show]*is_target[:max_show], label='target', color='red')

    axes[0].bar(node_names[:max_show], total_value_plot[:max_show], color='black', fill=False, label="total")
    axes[0].bar(node_names[:max_show], total_value_plot[:max_show]*total_shares_in_network_plot[:max_show], color='black', label="available")
    axes[0].legend()
    #axes[0].set_yscale('log')
    axes[0].set_ylabel("value (\$1M)")

    axes[1].set_xlabel("company name")
    axes[1].set_ylabel("shares")
    # axes[1].xticks(rotation=90)
    plt.setp( axes[1].xaxis.get_majorticklabels(), rotation=90 )
    #plt.xticks(np.arange(len(node_names)), node_names, rotation='vertical')
    #plt.title(r"$\mathbf{o} \in \mathcal{S}$")
    axes[1].legend()
    if filename is not None:
        my_savefig(filename)
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
        filename=None,
        max_show=100
        ):

    if cutoff is None:
        cutoff = -np.inf
    if source_mask is not None:
        source_mask = source_mask.detach().cpu().numpy()

    node_names = g.node_list.detach().cpu().numpy()
    ol_plot = ol
    total_shares_in_network_plot = total_shares_in_network
    print("Nuncontrollable:", g.identify_uncontrollable())
    total_shares_in_network_plot[g.identify_uncontrollable()] = 1.0 #extra
    if target_mask is not None:
        target_mask = target_mask.detach().cpu().numpy()
        node_names = node_names[target_mask]
        if ol is not None:
            ol_plot = ol_plot[target_mask]
        if total_shares_in_network is not None:
            total_shares_in_network_plot = total_shares_in_network_plot[target_mask]
        if source_mask is not None:
            source_mask = source_mask[target_mask]
        target_mask = target_mask[target_mask]

    plt.figure(figsize=figsize)
    #print(loss_control_arr)
    control_arr = 1-loss_control_arr
    #print(len(loss_control_arr))
    idx = np.argsort(control_arr)[::-1]
    #print(idx)
    idx = [i for i in idx if control_arr[i] >= cutoff]
    print(idx)
    node_names = ["{}".format(x) for x in node_names[idx]]
    yval = 1-loss_control_arr[idx]
    if target_mask is not None:
        target_mask = target_mask[idx]
    if source_mask is not None:
        source_mask = source_mask[idx]
    if ol is not None:
        ol_plot = ol_plot[idx]

    #print(yval)
    plt.bar(node_names[:max_show], yval[:max_show], color='blue', label="total")
    if ol is not None:
        #print(len(ol), g.number_of_nodes)
        #mask = np.ones((g.number_of_nodes), dtype=bool)[idx]
        #if target_mask is not None:
        #    mask *= target_mask
        #if source_mask is not None:
        #    mask *= source_mask
        #ol_plot = reduce_from_mask(ol_plot*mask, target_mask)
        #print(len(ol_plot), len(mask), len(target_mask))
        #print(node_names[:max_show], ol_plot[:max_show])
        plt.bar(node_names[:max_show], ol_plot[:max_show], color='red', label='direct')
    if total_shares_in_network is not None:
        total_shares_in_network_plot = total_shares_in_network_plot[idx]
        #total_shares_in_network_plot = reduce_from_mask(total_shares_in_network*mask, target_mask)
        plt.bar(node_names[:max_show], total_shares_in_network_plot[:max_show], color='black', label='available shares', fill=False)
    plt.xlabel("company name")
    plt.ylabel("control $c$")
    plt.xticks(rotation=90)
    #plt.xticks(np.arange(len(node_names)), node_names, rotation='vertical')
    plt.legend()
    if filename is not None:
        my_savefig(filename)
    plt.show()
