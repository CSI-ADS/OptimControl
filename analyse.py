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

# %load_ext autoreload
# %autoreload 2

device = 'cpu'
if torch.cuda.is_available():
    print("CUDA available")
    device = 'cuda'
print(device)
# -

A = pd.read_csv("data/edges_pharma_lc-3.csv", index_col=0)
#A = A.loc[A["ownership"]>1e-4,:] # make things a bit easier
display(A.head(2))
print(A.shape, A["ownership"].unique())
V = pd.read_csv("data/value_pharma_lc.csv", index_col=0)
display(V.head(2))
print(V.shape)

nodes = set(A["shareholder_id"].unique()).union(set(A["company_id"].unique()))
len(nodes)

values = {u:{"value":p} for u, p in V.set_index("company_id")["assets"].iteritems()}
values

edges = {(u, v):{"weight":p} for u, v, p in zip(A["shareholder_id"].values, A["company_id"].values, A["ownership"].values)}
edges

G_tot = nx.DiGraph()
for u in nodes:
    if u in values.keys():
        G_tot.add_node(u, **values[u])
for (u, v), p in edges.items():
    if G_tot.has_node(u) and G_tot.has_node(v):
        G_tot.add_edge(u, v, **p)

largest_cc = max(nx.connected_components(G_tot.to_undirected()), key=len)
len(largest_cc)
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
g.educated_value_guess()
g = g.remove_uncontrollable()

# +
from collections import defaultdict
from src.optim import *
from src.vitali import *
from src.loss import *


cl = get_cl_init(g.number_of_nodes, device=device)
print(len(cl), g.number_of_nodes)
lr = 0.001
max_steps = 10000
_, _, hist = optimize_control(compute_sparse_loss_cache_vars, cl, g, 
                              lambd=0, return_hist=True, verbose=True, num_steps=max_steps, lr=lr,
                              device=device)
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

plt.figure()
num_params = hist["params_sm"][0].shape[0]
for i in range(num_params):
    plt.plot([hist["params_sm"][t][i] for t in range(len(hist["params_sm"]))], label=str(i))
plt.xlabel("step")
plt.ylabel("parameter (sigmoid) = % bought of company i")
plt.yscale('log')
#plt.legend()
plt.show()

plt.bar(np.arange(num_params), hist["final_params_sm"])
plt.xlabel("parameter (sigmoid)")
plt.ylabel("value after optimization")
plt.show()

# +
cs = []
ss = []
param_result = []
lambd_range = np.linspace(0, 5, num=20)#np.logspace(-2, 1, num=10)
for lambd in lambd_range:
    cl = get_cl_init(g.number_of_nodes, device=device)
    loss_fn = compute_sparse_loss_cache_vars
    cl, cost = optimize_control(loss_fn, cl, g, lambd=lambd, return_hist=False, 
                                lr=lr, num_steps=max_steps, verbose=True, device=device)
    
    # get some stats
    with torch.no_grad():
        c, s = compute_sparse_loss_cache_vars(cl, g, lambd=lambd, as_separate=True)
        param_result.append(cl.detach().cpu())
        cs.append(c.detach().cpu())
        ss.append(s.detach().cpu())
        #print(lambd, c, s)
        
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

nodes = g.nodes
props = node_nx_props.loc[nodes,:]
for col in props.columns:
    props_col = props[col].sort_values()
    plt.figure(figsize=(10,5))
    for lambd, params in zip(lambd_range, param_result):
        plt.plot(props_col.values, params, label="{:.2f}".format(lambd))
    plt.xlabel(col)
    plt.ylabel("c")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()




