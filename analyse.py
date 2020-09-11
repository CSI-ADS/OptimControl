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

# %load_ext autoreload
# %autoreload 2

device = 'cpu'
if torch.cuda.is_available():
    print("CUDA available")
    device = 'cuda'
print(device)

# +
NETWORK_NAME = [
    "PHARMA",
    "VITALI",
    "SIMPLE_CYCLE",
    "SIMPLE_CHAIN"
][0]
print(NETWORK_NAME)

if NETWORK_NAME in ("VITALI", "SIMPLE_CHAIN", "SIMPLE_CYCLE"):
    G_nx = get_test_network(NETWORK_NAME)
    nx.adjacency_matrix(G_nx).todense()
    nx.draw(G_nx, pos=nx.spring_layout(G_nx), arrows=True, with_labels=True)
    nodes = list(G_nx.nodes())
    values = {u:{"value":1} for u in G_nx.nodes()}
    edges = {(u,v):{"weight":p} for (u,v),p, in nx.get_edge_attributes(G_nx, "weight").items()}
else:
    A = pd.read_csv("data/edges_pharma_lc-3.csv", index_col=0)
    #A = A.loc[A["ownership"]>1e-4,:] # make things a bit easier
    display(A.head(2))
    print(A.shape, A["ownership"].unique())
    V = pd.read_csv("data/value_pharma_lc.csv", index_col=0)
    display(V.head(2))
    print(V.shape)
    
    A = A.astype({"shareholder_id": int, "company_id": int})
    nodes = set(A["shareholder_id"].unique()).union(set(A["company_id"].unique()))
    print(len(nodes))
    values = {int(u):{"value":p} for u, p in V.set_index("company_id")["assets"].iteritems()}
    edges = {(int(u), int(v)):{"weight":p} for u, v, p in zip(A["shareholder_id"].values, A["company_id"].values, A["ownership"].values)}
# -

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

largest_cc = max(nx.connected_components(G_tot.to_undirected()), key=len)
print(len(largest_cc))
G = G_tot.subgraph(largest_cc).copy()

# +
#nx.draw(G, pos=nx.spring_layout(G), node_size=10)
# -

vals = pd.Series(pd.DataFrame(values).T["value"], name="val")
centr = pd.Series(nx.degree_centrality(G), name="centr")
in_centr = pd.Series(nx.in_degree_centrality(G), name="in_centr")
out_centr = pd.Series(nx.out_degree_centrality(G), name="out_centr")
betw_centr = pd.Series(nx.betweenness_centrality(G), name="betw_centr")
pr = pd.Series(nx.pagerank(G), name="pr")
node_nx_props = pd.concat([vals, centr, in_centr, out_centr, betw_centr, pr], axis=1)
node_nx_props = node_nx_props.dropna()
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
print("Weight values", np.unique(np.asarray(nx.adjacency_matrix(G, weight="weight").todense()).flatten()))

g = Network( # ordered according to G.nodes()
        nx.adjacency_matrix(G, weight="weight"), 
        value=np.array([values[n]["value"] for n in G.nodes()]), 
        node_list=list(G.nodes()),
        dtype=torch.float
)

print(g.number_of_nodes, g.number_of_edges)
# if g.number_of_nodes < 10000:
#     plt.figure(figsize=(20,20))
#     g.draw(color_arr=g.value)
g.educated_value_guess()
print("dropping nans")
g = g.dropna_vals()
print("dropping uncontrollable")
#g = g.remove_uncontrollable()

print(g.number_of_nodes, g.number_of_edges)
if g.number_of_nodes < 10000:
    plt.figure(figsize=(20,20))
    g.draw(color_arr=g.value)


# +
from collections import defaultdict
from src.optim import *
from src.vitali import *
from src.loss import *

cl = get_cl_init(g.number_of_nodes, device=device)
lr = 0.1
max_steps = 10000
_,_, hist = optimize_control(compute_sparse_loss, cl, g, 
                              lambd=1, return_hist=True, verbose=True, 
                             save_loss_arr=True,
                             num_steps=max_steps, lr=lr,
                              device=device, desc_mask=None
                            )
# -

print(g.total_value, torch.sum(g.total_shares_in_network))

# +
import matplotlib.pyplot as plt

plt.plot(hist["loss"])
plt.yscale('log')
plt.xlabel("step")
plt.ylabel("loss")
plt.show()
# -

g.A.todense()

# +
num_params = hist["params_sm"][0].shape[0]

plt.figure()
for loss_name, loss_arr in {k:v for k,v in hist.items() if k.startswith("loss_")}.items():
    for i in range(loss_arr[0].shape[0]):
        loss_tr = lambda x : x
        if loss_name == "loss_control":
            loss_tr = lambda x : -x+g.number_of_nodes
        plt.plot([loss_tr(loss_arr[t][i]) for t in range(len(loss_arr))], label=str(i))
    plt.xlabel("step")
    if loss_name == "loss_control":
        plt.ylabel("control over i")
    else:
        plt.ylabel("cost of buying i")
    # plt.yscale('log')
    if num_params <= 15:
        plt.legend()
    plt.show()


plt.figure()
for i in range(num_params):
    plt.plot([hist["params_sm"][t][i] for t in range(len(hist["params_sm"]))], label=str(i))
plt.xlabel("step")
plt.ylabel("parameter (sigmoid) = % bought of available company i stock")
plt.yscale('log')
if num_params <= 15:
    plt.legend()
plt.show()

plt.figure()
tot_shares = g.total_shares_in_network.detach().cpu().numpy()
tot_shares[g.identify_uncontrollable().detach().cpu().numpy()] = 1
num_params = hist["params_sm"][0].shape[0]
for i in range(num_params):
    plt.plot([hist["params_sm"][t][i]*tot_shares[i] for t in range(len(hist["params_sm"]))], label=str(i))
plt.xlabel("step")
plt.ylabel("parameter (sigmoid) = % bought of company i")
plt.yscale('log')
if num_params <= 15:
    plt.legend()
plt.show()

# -

if num_params <= 15:
    print("control sigmoid")
    print(hist["final_params_sm"])
    g.draw(external_ownership=hist["final_params_sm"], vmin=0, vmax=1)

    print("direct control weighted by shares")
    tot_shares = g.total_shares_in_network.detach().cpu().numpy()
    tot_shares[g.identify_uncontrollable().detach().cpu().numpy()] = 1
    print(tot_shares*hist["final_params_sm"])
    g.draw(external_ownership=tot_shares*hist["final_params_sm"], vmin=0, vmax=1)

    print("cost of buying")
    final_cost = hist["final_loss_cost"]
    print(final_cost)
    g.draw(color_arr=final_cost, vmin=0, vmax=1)
    
    print("propagated control")
    final_control = g.number_of_nodes-hist["final_loss_control"]
    print(final_control)
    g.draw(color_arr=final_control, vmin=0, vmax=1)
    
    print("color = control, size = cost")
    print(hist["final_loss_cost"].shape)
    g.draw(size_arr=final_cost, color_arr=final_control, vmin=0, vmax=1)

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
plt.show()

# +
cs = []
ss = []
param_result = []
#lambd_range = np.logspace(-1, 0, num=20)#np.logspace(-2, 1, num=10)
lambd_range = np.linspace(0, 2, num=40)#np.logspace(-2, 1, num=10)
print("lambdas to evaluate:", lambd_range)
for lambd in lambd_range:
    cl = get_cl_init(g.number_of_nodes, device=device)
    loss_fn = compute_sparse_loss
    cl, cost, hist = optimize_control(loss_fn, cl, g, lambd=lambd, return_hist=True, 
                                lr=lr, num_steps=max_steps, verbose=True, device=device)
    
    # get some stats
    with torch.no_grad():
        c, s = compute_sparse_loss(cl, g, lambd=lambd, as_separate=True)
        param_result.append(cl.detach().cpu())
        cs.append(c.detach().cpu())
        ss.append(s.detach().cpu())
        print(lambd, param_result[-1], c, s)
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

ax1.plot(lambd_range, (g.number_of_nodes-cs)/g.number_of_nodes, label="control", color='blue')
ax1.set_ylabel("control")
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.plot(lambd_range, ss, label="total budget", color='red')
ax1.set_ylabel("total budget")

#plt.legend()
ax1.set_xlabel("lambda")
plt.show()
# -

node_nx_props.index = node_nx_props.index.astype(int)
nodes = [(n, i) for i, n in enumerate(g.nodes.detach().cpu().numpy().astype(int)) if n in node_nx_props.index]
nodes, node_idx = zip(*nodes)
node_idx = np.array(node_idx).flatten()
props = node_nx_props.loc[nodes,:]
for col in props.columns:
    props_col = props[col].sort_values()
    plt.figure(figsize=(10,5))
    for lambd, params in zip(lambd_range, param_result):
        plt.scatter(props[col].values, params[node_idx], label="{:.2f}".format(lambd), alpha=0.1)
    plt.xlabel(col)
    plt.ylabel("c")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()    

plt.figure()
for lambd, params in zip(lambd_range, param_result):
    if lambd == 0: continue
    p = params[node_idx]
    #p = p[p < 0.25]
    plt.hist(p, label="{:.2f}".format(lambd), alpha=0.1, bins=100)
plt.ylabel("c")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

g.to('cpu')






