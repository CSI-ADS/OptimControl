# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import pandas as pd
from src.test_networks import *
from src.utils import reduce_from_mask, pad_from_mask
from src.plotting import *
import sys, os

# %load_ext autoreload
# %autoreload 2

device = 'cpu'
use_gpu = False
if use_gpu and torch.cuda.is_available():
    print("CUDA available")
    device = 'cuda'
print(device)

# +
NETWORK_NAME = [
    "PHARMA",
    "BIOTECH",
    "BIOTECH_PM",
    "VITALI",
    "SIMPLE_CYCLE",
    "SIMPLE_CHAIN",
    "SIMPLE_STAR"
][1]
print(NETWORK_NAME)

LIMIT_CONTROL=True

if NETWORK_NAME in ("VITALI", "SIMPLE_CHAIN", "SIMPLE_CYCLE", "SIMPLE_STAR"):
    G_nx = get_test_network(NETWORK_NAME)
    nx.adjacency_matrix(G_nx).todense()
    nx.draw(G_nx, pos=nx.spring_layout(G_nx), arrows=True, with_labels=True)
    nodes = list(G_nx.nodes())
    if NETWORK_NAME == "SIMPLE_STAR":
        values = {u:{"value":p["value"]} for u, p in G_nx.nodes(data=True)}
    else:
        values = {u:{"value":1} for u in G_nx.nodes()}
    edges = {(u,v):{"weight":p} for (u,v),p, in nx.get_edge_attributes(G_nx, "weight").items()}
else:
    edge_filename, value_filename = {
        "PHARMA" : ("edges_pharma_lc-3.csv", "value_pharma_lc.csv"),
        "BIOTECH" : ("edges_belgian_biotech_reach_nace.csv", "value_belgian_biotech_reach_nace.csv"),
        "BIOTECH_PM" : ("edges_belgian_biotech_plus1mil.csv", "value_belgian_biotech_plus1mil-2.csv"),
    }[NETWORK_NAME]
    
    
    
    A = pd.read_csv(os.path.join("data", edge_filename), index_col=0)
    #A = A.loc[A["ownership"]>1e-4,:] # make things a bit easier
    display(A.head(2))
    print(A.shape)
#     print(A.shape, A["ownership"].unique())
    V = pd.read_csv(os.path.join("data", value_filename), index_col=0)
    V.loc[:, "assets"] = V.loc[:, "assets"] / 10**6 # per 1M
    plt.hist(V["assets"].values)
    plt.ylabel("assets")
    plt.yscale('log')
#     plt.xscale('log')
    plt.show()
    display(V.head(2))
    print(V.shape)
    
    A = A.astype({"shareholder_id": int, "company_id": int})
    nodes = set(A["shareholder_id"].unique()).union(set(A["company_id"].unique()))
    print(len(nodes))
    values = {int(u):{"value":p} for u, p in V.set_index("company_id")["assets"].iteritems()}
    edges = {(int(u), int(v)):{"weight":p} for u, v, p in zip(A["shareholder_id"].values, A["company_id"].values, A["ownership"].values)}
    
    print(V.loc[V["assets"].idxmax(axis=0, skipna=True),:])
    
    if len(nodes) != len(V["company_id"].unique()):
        print("!"*10)
        print("WARNING: not the same amount of nodes in edges and nodes list:")
        print("in edges:", len(nodes))
        print("in nodes:", len(V["company_id"].unique()))
        print("intersection:", len(nodes.intersection(set(V["company_id"].unique()))))
        difference_sets = nodes.symmetric_difference(set(V["company_id"].unique()))
        display(V.loc[V["company_id"].isin(difference_sets),:].head(5))
    
    if LIMIT_CONTROL:
        NETWORK_NAME += "_sc"
        V_source = V.loc[~V["is_financial"],"company_id"].astype(int).unique()
        print("source:", V_source.shape)
        V_target = V.loc[V["is_biotech"],"company_id"].astype(int).unique()
        print("target:", V_target.shape)

print(NETWORK_NAME)
assert len(values)>0, "?"
assert len(edges)>0, "?"
# -

V

nodes

values

edges

G_tot = nx.DiGraph()
for u in nodes:
    if u in values.keys():
        G_tot.add_node(u, **values[u])
for (u, v), p in edges.items():
    if G_tot.has_node(u) and G_tot.has_node(v):
        G_tot.add_edge(u, v, **p)
print(G_tot.number_of_nodes())

# +
print("components:", [len(x) for x in nx.connected_components(G_tot.to_undirected())])
largest_cc = max(nx.connected_components(G_tot.to_undirected()), key=len)
print(len(largest_cc))
# nodes_to_remove = []
# for component in nx.connected_components(G_tot.to_undirected()):
#     if component == largest_cc:
#         continue
#     display(V.loc[V["company_id"].isin(component),:])


G = G_tot.subgraph(largest_cc).copy()
# largest_cc = list(G_tot.nodes())
# print(largest_cc)
# G = G_tot.subgraph(largest_cc).copy()
if G_tot.number_of_nodes() != G.number_of_nodes():
    print("WARNING: keeping {} out of {} nodes".format(G.number_of_nodes(), G_tot.number_of_nodes()))

G.number_of_nodes()

# +
#nx.draw(G, pos=nx.spring_layout(G), node_size=10)
# -

values

vals = pd.Series(pd.DataFrame(values).T["value"], name="val")
centr = pd.Series(nx.degree_centrality(G), name="centr")
in_centr = pd.Series(nx.in_degree_centrality(G), name="in_centr")
out_centr = pd.Series(nx.out_degree_centrality(G), name="out_centr")
betw_centr = pd.Series(nx.betweenness_centrality(G), name="betw_centr")
pr = pd.Series(nx.pagerank(G), name="pr")
node_nx_props = pd.concat([vals, centr, in_centr, out_centr, betw_centr, pr], axis=1)
#node_nx_props = node_nx_props.dropna()
node_nx_props

from numpy.polynomial.polynomial import polyfit
for col in node_nx_props.columns:
    if col == "val": continue
    x = node_nx_props["val"].values
    y = node_nx_props[col].values
    plt.scatter(x, y, s=1) 
    plt.xlabel("value")
    plt.ylabel(col)
    plt.xscale('log')
#     plt.yscale('log')
    plt.show()

# +
from src.network import *

print("Edge example:", [u for u in G.edges(data=True)][0])

g = Network( # ordered according to G.nodes()
        nx.adjacency_matrix(G, weight="weight"), 
        value=np.array([values[n]["value"] for n in G.nodes()]), 
        node_list=list(G.nodes()),
        dtype=torch.float
)

figsize=(10,10)

if g.number_of_nodes < 50:
    print("Weight values", np.unique(np.asarray(nx.adjacency_matrix(G, weight="weight").todense()).flatten()))

print(g.number_of_nodes, g.number_of_edges)
print("-"*100)
g.educated_value_guess()
print("ZEROS")
g.draw_zeros(figsize=figsize)
print("NANS")
g.draw_nans(figsize=figsize)
print("="*100)
if g.number_of_nodes < 10000:
    print("Value graph")
    g.draw(color_arr=g.value, figsize=figsize)
print("dropping nans")
# TODO: better handling of remaining nans: what are these?
g = g.dropna_vals()
# print("dropping uncontrollable")
#g = g.remove_uncontrollable()

print(g.number_of_nodes, g.number_of_edges)
if g.number_of_nodes < 10000:
    print("Value graph")
    g.draw(color_arr=g.value, figsize=figsize, filename="figs/{}.pdf".format(NETWORK_NAME), show_edge_values=True)
    
g = g.to(device)


# +
source_mask=None
target_mask=None


if LIMIT_CONTROL:
    print("limiting control")
    target_mask = make_mask_from_node_list(g, V_target.astype(int))
    g = g.remove_irrelevant(target_mask)
    target_mask = make_mask_from_node_list(g, V_target.astype(int)) # redo!
    source_mask = make_mask_from_node_list(g, V_source.astype(int))
    print("({}) sources {}, target {}".format(g.number_of_nodes, sum(source_mask), sum(target_mask)))
    print(g.number_of_nodes, g.number_of_edges, source_mask.shape, target_mask.shape)
    
    g.draw(color_arr=g.value, figsize=figsize, filename="figs/{}.pdf".format(NETWORK_NAME), show_edge_values=True, 
          source_mask=source_mask, target_mask=target_mask)
# -

plt.hist(g.compute_total_value(only_network_shares=True, include_root_shares=True).detach().cpu().numpy())
plt.hist(g.compute_total_value(only_network_shares=True, include_root_shares=True, sel_mask=source_mask).detach().cpu().numpy())
plt.yscale('log')
plt.ylabel("euro")
plt.show()

# +
from collections import defaultdict
from src.optim import *
from src.vitali import *
from src.loss import *
from src.network import *

number_of_sources = g.number_of_nodes if source_mask is None else sum(source_mask)
print(g.number_of_nodes, number_of_sources)
cl = get_cl_init(number_of_sources, device=device)
cl_soft = torch.sigmoid(cl)
print(torch.min(cl_soft), torch.max(cl_soft))
init_lr = 0.1
decay = 0.1
max_steps = 2
lambd=0.1
weight_control = False
control_cutoff = None
use_schedule = False
lr = init_lr
scheduler = None
if use_schedule: # make new copy
    scheduler = lambda opt : torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[7500, 9500], gamma=decay)

_,_, hist = optimize_control(compute_sparse_loss, cl, g, 
                             lambd=lambd, return_hist=True, verbose=True, 
                             save_loss_arr=True,
                             save_params_every=500,
                             num_steps=max_steps, lr=lr, scheduler=scheduler,
                             device=device, 
                             weight_control=weight_control,
                             control_cutoff=control_cutoff,
                             loss_tol=1e-5,
                             save_separate_losses=True,
                             source_mask=source_mask,
                             target_mask=target_mask
                            )
# -

if source_mask is not None:
    print(torch.sum(g.compute_total_value(only_network_shares=True, include_root_shares=True)))
Vtot = g.compute_total_value(only_network_shares=True, include_root_shares=True, sel_mask=source_mask)
print(torch.sum(Vtot))
#print(Vtot)

# +
# import torch.autograd.profiler as profiler
# with profiler.profile(record_shapes=True) as prof:
#     with profiler.record_function("model_inference"):
#         _,_, hist = optimize_control(compute_sparse_loss, cl, g, 
#                                      lambd=1, return_hist=True, verbose=True, 
#                                      save_loss_arr=True,
#                                      num_steps=max_steps, lr=lr, scheduler=scheduler,
#                                      device=device, desc_mask=None
#                                     )

# prof.export_chrome_trace("trace.json")

# +
import matplotlib.pyplot as plt


plt.plot(hist["lr"])
plt.yscale('log')
plt.xlabel("step")
plt.ylabel("lr")
plt.show()

plt.plot(hist["loss"])
plt.yscale('log')
plt.xlabel("step")
plt.ylabel("loss")
plt.show()


plt.plot(hist["loss_control"])
# plt.plot(hist["loss_control_arr"], color='black')
plt.yscale('log')
plt.xlabel("step")
plt.ylabel("loss_control")
plt.show()

plt.plot(hist["loss_cost"])
plt.yscale('log')
plt.xlabel("step")
plt.ylabel("loss_cost")
plt.show()



# +
num_params = hist["params_sm"][0].shape[0]
print(num_params)

plt.figure()
for loss_name, loss_arr in {k:v for k,v in hist.items() if (k.startswith("loss_") and k.endswith("arr"))}.items():
    print(loss_name)
    for i in range(loss_arr[0].shape[0]):
        loss_tr = lambda x : x
        if loss_name == "loss_control":
            loss_tr = lambda x : -x+1
        plt.plot([loss_tr(loss_arr[t][i]) for t in range(len(loss_arr))], label=str(i))
    plt.xlabel("step")
    if loss_name == "loss_control":
        plt.ylabel("control over i")
    else:
        plt.ylabel("cost of buying i")
#     plt.yscale('log')
    if num_params <= 15:
        plt.legend()
    plt.savefig("figs/{}_{}_{}_ifo_step.pdf".format(NETWORK_NAME, lambd, loss_name))
    plt.show()


plt.figure()
for i in range(num_params):
    plt.plot([hist["params_sm"][t][i] for t in range(len(hist["params_sm"]))], label=str(i))
plt.xlabel("step")
plt.ylabel("parameter (sigmoid) = % bought of !available! company i stock")
plt.yscale('log')
if num_params <= 15:
    plt.legend()
plt.savefig("figs/{}_{}_sigmoid_ifo_step.pdf".format(NETWORK_NAME, lambd))
plt.show()

plt.figure()
for i in range(num_params):
    plt.plot([hist["ol"][t][i] for t in range(len(hist["ol"]))], label=str(i))
plt.xlabel("step")
plt.ylabel("o = % bought of company i")
plt.yscale('log')
if num_params <= 15:
    plt.legend()
plt.savefig("figs/{}_{}_totshares_ifo_step.pdf".format(NETWORK_NAME, lambd))
plt.show()
# -


hist

hist["ol"][0].shape

# +
plot_direct_control(
        g,
        hist["final_ol"],
        source_mask=source_mask,
        target_mask=target_mask,
        figsize=(20, 5)
        )

plot_control_distr(
        g,
        hist["final_loss_control_arr"],
        ol=hist["final_ol"],
        source_mask=source_mask,
        target_mask=target_mask,
        cutoff=None,
        figsize=(20, 5)
        )
# -

if num_params <= 1000:
    figsize=(10,10)
    print("direct direct control o")
    direct_control=hist["final_ol"]
    g.draw(external_ownership=direct_control, vmin=0, vmax=1, figsize=figsize)

    print("cost of buying")
    final_cost=pad_from_mask(hist["final_loss_cost_arr"], source_mask)
    g.draw(color_arr=final_cost, vmin=0, vmax=1, figsize=figsize)
    
    print("total/propagated control")
    final_control = pad_from_mask(1-hist["final_loss_control_arr"], target_mask)
    #print(final_control)
    g.draw(color_arr=final_control, vmin=0, vmax=1, figsize=figsize)
    
    print("size = total control, color = cost")
    #print(hist["final_loss_cost"].shape)
    g.draw(color_arr=final_cost, size_arr=final_control, vmin=0, vmax=1, figsize=figsize, filename="figs/{}_{}_size_control_color_cost.pdf".format(NETWORK_NAME, lambd),
          target_mask=target_mask)

g.device
device

# +
# cl_test = torch.tensor([-100, 100, 100, 100, 100, 100], device=g.device, dtype=torch.float)
# cl_test = torch.sigmoid(cl_test)
# print(cl_test)
# print(compute_control_with_external(cl_test, g))
# g.draw(external_ownership=cl_test)
# -

plt.bar(np.arange(num_params), hist["final_params_sm"])
plt.xlabel("parameter (sigmoid)")
plt.ylabel("value after optimization")
# plt.xlim(-1,10)
plt.show()

# +
cs = []
ss = []
param_result = []
lambd_range = np.logspace(-10, 2, num=15)#np.logspace(-2, 1, num=10)
# lambd_range = np.linspace(0, 1, num=20)#np.logspace(-2, 1, num=10)
print("lambdas to evaluate:", lambd_range)
number_of_sources = g.number_of_nodes if source_mask is None else sum(source_mask)
    
for lambd in lambd_range:
    cl = get_cl_init(number_of_sources, device=device)

    loss_fn = compute_sparse_loss
    cl, cost, hist = optimize_control(loss_fn, cl, g, lambd=lambd, return_hist=True, save_params_every=10000,
                                lr=lr, num_steps=max_steps, verbose=True, device=device, weight_control=weight_control,
                                     control_cutoff=control_cutoff, source_mask=source_mask, target_mask=target_mask, 
                                      loss_tol=1e-8)
    # get some stats
    with torch.no_grad():
        c, s = compute_value(loss_fn, cl, g, lambd=lambd, as_separate=True, source_mask=source_mask, target_mask=target_mask)
        print("lambd={}, c={}, s={}".format(lambd, c.detach().cpu().numpy(), s.detach().cpu().numpy()))
        param_result.append(cl.detach().cpu())
        cs.append(c.detach().cpu())
        ss.append(s.detach().cpu())
        #print(lambd, param_result[-1], c, s)
        if num_params <= 15:
            tot_shares = g.total_shares_in_network.detach().cpu().numpy()
            tot_shares[g.identify_uncontrollable().detach().cpu().numpy()] = 1
            print(tot_shares*hist["final_params_sm"])
            g.draw(external_ownership=tot_shares*hist["final_params_sm"])
        
with torch.no_grad():
    cs = np.array([x.numpy() for x in cs])
    ss = np.array([x.numpy() for x in ss])
    param_result = torch.sigmoid(torch.stack(param_result)).numpy()

# +
fig, ax1 = plt.subplots()

if weight_control:
    raise NotImplemented("todo")
#     y_control = (torch.sum(g.value).detach().cpu().numpy()-cs)/torch.sum(g.value).detach().cpu().numpy()
    #print(y_control)
else:
    number_of_target_nodes = int(sum(target_mask) if target_mask is not None else g.number_of_nodes) 
    print("targets:", number_of_target_nodes)
    y_control = (number_of_target_nodes-cs)#/g.number_of_nodes
    
ax1.plot(lambd_range, y_control, label="control", color='blue')
ax1.scatter(lambd_range, y_control, color='blue')
ax1.set_ylabel("control", color='blue')
ax1.set_xscale('log')
ax1.set_yscale('log')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.plot(lambd_range, ss, label="total budget", color='red')
ax2.scatter(lambd_range, ss, color='red')
ax2.set_ylabel("total budget", color='red')
ax2.set_yscale('log')
ax1.set_xlabel("lambda")
plt.legend()

plt.savefig("figs/{}_lambda_curve.pdf".format(NETWORK_NAME))
plt.show()


plt.plot(ss, y_control)
plt.xlabel("total budget")
plt.ylabel("total control")
plt.xscale('log')
plt.yscale('log')
plt.savefig("figs/{}_control_vs_cost_curve.pdf".format(NETWORK_NAME))

plt.show()

# +
# node_nx_props.index = node_nx_props.index.astype(int)
# nodes = [(n, i) for i, n in enumerate(g.nodes.detach().cpu().numpy().astype(int)) if n in node_nx_props.index]
# nodes, node_idx = zip(*nodes)
# node_idx = np.array(node_idx).flatten()
# props = node_nx_props.loc[nodes,:]
# for col in props.columns:
#     props_col = props[col].sort_values()
#     plt.figure(figsize=(10,5))
#     for lambd, params in zip(lambd_range, param_result):
#         plt.scatter(props[col].values, params[node_idx], label="{:.2f}".format(lambd), alpha=0.1)
#     plt.xlabel(col)
#     plt.ylabel("c")
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.show()    

# +
# plt.figure()
# for lambd, params in zip(lambd_range, param_result):
#     if lambd == 0: continue
#     p = params[node_idx]
#     #p = p[p < 0.25]
#     plt.hist(p, label="{:.2f}".format(lambd), alpha=0.1, bins=100)
# plt.ylabel("c")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.show()
# -

# # Constraint

source_mask

g.draw(
    color_arr=g.value, 
    size_arr=None,
    figsize=figsize, 
    filename="figs/{}_overview.pdf".format(NETWORK_NAME), 
    show_edge_values=True, 
    source_mask=source_mask, 
    target_mask=target_mask,
    colorbar_scale='log',
    colorbar_text='value (€1M)',
    rescale_color=False,
    rescale_size=True
)

# +
from collections import defaultdict
from src.optim import *
from src.vitali import *
from src.loss import *
from src.network import *

print(g.number_of_nodes)
budget = 10000
number_of_sources = g.number_of_nodes if source_mask is None else sum(source_mask)    
cl = get_cl_init(number_of_sources, device=device)
print(number_of_sources)
cl_soft = torch.sigmoid(cl)
print(torch.min(cl_soft), torch.max(cl_soft))
init_lr = 0.01
decay = 0.1
num_steps = 1000
max_iter = 2
use_schedule = False
lr = init_lr
scheduler = None
if use_schedule: # make new copy
    scheduler = lambda opt : torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[7500, 9500], gamma=decay)

print(torch.sum(g.compute_total_value()))
    
# _,_, hist = optimize_control(compute_sparse_loss, cl, g, 
#                              lambd=1, return_hist=True, verbose=True, 
#                              save_loss_arr=True,
#                              num_steps=max_steps, lr=lr, scheduler=scheduler,
#                              device=device, source_mask=None, target_mask=None,
#                              weight_control=weight_control,
#                              control_cutoff=control_cutoff
#                             )

param_est, loss_vals, constr_vals, hist_all = constraint_optimize_control(
        compute_sparse_loss, cl, g, budget,
        verbose=True, return_hist=True,
        lr=init_lr, scheduler=scheduler,
        max_iter=max_iter, num_steps=num_steps,
        device=device, save_params_every=10000, save_loss_arr=False,
        constr_tol=1e-8,
        loss_tol = 1e-8,
        source_mask=source_mask, target_mask=target_mask,
        weight_control=weight_control,
        control_cutoff=control_cutoff
        )

# +
plt.scatter(hist_all["step_nr"], hist_all["loss_control"])
plt.xlabel("i")
plt.ylabel("loss_contr")
plt.show()

plt.scatter(hist_all["step_nr"], hist_all["loss_cost"])
plt.xlabel("i")
plt.ylabel("loss_cost")
plt.show()

plt.scatter(hist_all["step_nr"], hist_all["loss_augm"])
plt.xlabel("i")
plt.ylabel("augm")
plt.show()

plt.scatter(hist_all["step_nr"], hist_all["constr"])
plt.xlabel("i")
plt.ylabel("constr")
plt.show()

plt.scatter(hist_all["step_nr"], hist_all["rho"])
plt.xlabel("i")
plt.ylabel("rho")
plt.show()
# +
print("size = total control, color = cost")
#print(hist["final_loss_cost"].shape)
final_cost = pad_from_mask(hist_all["final_loss_cost_arr"], source_mask)
final_control = pad_from_mask(1-hist_all["final_loss_control_arr"], target_mask)
     
g.draw(color_arr=final_cost, vmin=0, vmax=1, figsize=figsize, 
       filename="figs/{}_size_control_color_cost_budget_{}.pdf".format(NETWORK_NAME, budget),
      target_mask=target_mask, source_mask=source_mask)


# +
plot_direct_control(
        g,
        hist_all["final_ol"],
        source_mask=source_mask,
        target_mask=target_mask,
        figsize=(20, 5)
        )

plot_control_distr(
        g,
        hist_all["final_loss_control_arr"],
        ol=hist_all["final_ol"],
        source_mask=source_mask,
        target_mask=target_mask,
        cutoff=None,
        figsize=(20, 5)
        )
# -

import pickle
with open('dump.pickle', 'wb') as handle:
    pickle.dump((param_est, loss_vals, constr_vals, hist_all, target_mask, source_mask), handle)


