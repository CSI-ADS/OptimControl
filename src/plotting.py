import numpy as np
import matplotlib.pyplot as plt
from .utils import *
from .optim import compute_total_shares


def plot_direct_control(
        g,
        ol,
        source_mask=None,
        target_mask=None,
        figsize=(20, 5)
        ):
    nodelist = g.node_list.detach().cpu().numpy()
    if source_mask is not None:
        source_mask = source_mask.detach().cpu().numpy()
        nodelist = nodelist[source_mask]
    is_target = np.array([True]*ol.shape[0])
    if target_mask is not None:
        is_target = target_mask.detach().cpu().numpy()

    ol_plot = ol
    if source_mask is not None:
        ol_plot = ol_plot[source_mask]
        is_target = is_target[source_mask]
    idx = np.argsort(ol_plot)[::-1]
    node_names = np.array(["{}".format(x) for x in nodelist[idx]], dtype=object)
    if source_mask is not None:
        ol_plot = ol_plot[idx]
        is_target = is_target[idx]
    plt.figure(figsize=figsize)
    plt.bar(node_names, ol_plot, label=None, color='blue')
    if target_mask is not None:
        plt.bar(node_names[is_target], ol_plot[is_target], label='target', color='red')
    plt.xlabel("company name")
    plt.ylabel("total shares bought")
    plt.xticks(np.arange(len(node_names)), node_names, rotation='vertical')
    plt.title("vector o in set S")
    plt.legend()
    plt.show()


def plot_control_distr(
        g,
        loss_control_arr,
        ol=None,
        source_mask=None,
        target_mask=None,
        cutoff=None,
        figsize=(20, 5)
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
    plt.xlabel("company name")
    plt.ylabel("total control")
    plt.xticks(np.arange(len(node_names)), node_names, rotation='vertical')
    plt.legend()
    plt.show()
