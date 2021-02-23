# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
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
from src.utils import reduce_from_mask, pad_from_mask, my_savefig
from src.plotting import *
import sys, os


np.random.seed(seed=33)

# %load_ext autoreload
# %autoreload 2

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
## for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

torch.set_default_dtype(torch.float64)

device = 'cpu'
use_gpu = False
if use_gpu and torch.cuda.is_available():
    print("CUDA available")
    device = 'cuda'
print(device)

# -

COUNTRY_CODES = """
0	NL	0
1	HR	1
2	FR	2
3	IE	3
4	RO	4
5	GB	5
6	LV	6
7	BE	7
8	DK	8
9	PT	9
10	SK	10
11	SI	11
12	BG	12
13	CZ	13
14	AT	14
15	PL	15
16	SE	16
17	HU	17
18	NO	18
19	DE	19
20	IT	20
21	ES	21
22	FI	22
23	LT	23
24	EE	24
25	GR	25
"""
COUNTRY_CODES = COUNTRY_CODES.strip().split("\n")
COUNTRY_CODES = [l.strip().split()[1:] for l in COUNTRY_CODES]
COUNTRY_CODES = pd.DataFrame(COUNTRY_CODES, columns=["countryisocode", "country_index"])
COUNTRY_CODES["country_index"] = COUNTRY_CODES["country_index"].astype(int)
COUNTRY_CODES = COUNTRY_CODES.set_index("country_index")
COUNTRY_CODES = COUNTRY_CODES.to_dict()["countryisocode"]
COUNTRY_CODES

# +
NETWORK_NAME = [
    "PHARMA",
    "BIOTECH",
    "BIOTECH_PM",
    "BIOTECH_SEC",
    "VITALI",
    "SIMPLE_CYCLE",
    "SIMPLE_CHAIN",
    "SIMPLE_STAR"
][3]
print(NETWORK_NAME)

# set to false for most examples
LIMIT_CONTROL=True
assert LIMIT_CONTROL
# WARNING: DONT CHANGE THIS, NEVER CHECKED WItHOUT SOUrCE MASK ETC
LAYOUT = nx.spring_layout
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
    LAYOUT = lambda x: nx.nx_pydot.pydot_layout(x, prog='twopi', root=None)
    V_target = np.array(nodes)
    V_source = np.array(nodes)
else:
    edge_filename, value_filename = {
        "PHARMA" : ("edges_pharma_lc-3.csv", "value_pharma_lc.csv"),
        "BIOTECH" : ("edges_belgian_biotech_reach_nace.csv", "value_belgian_biotech_reach_nace.csv"),
        "BIOTECH_PM" : ("edges_belgian_biotech_plus1mil.csv", "value_belgian_biotech_plus1mil-2.csv"),
        "BIOTECH_SEC" : ("edges_GB_biotech_reach_lc_nace_w_assets.csv", "value_GB_biotech_reach_lc_nace_w_assets.csv")
    }[NETWORK_NAME]
    
    
    
    A = pd.read_csv(os.path.join("data", edge_filename), index_col=0)
    #A = A.loc[A["ownership"]>1e-4,:] # make things a bit easier
    display(A.head(2))
    print(A.shape)
#     print(A.shape, A["ownership"].unique())
    V = pd.read_csv(os.path.join("data", value_filename), index_col=0)
    V.loc[:,"country"] = V.loc[:,"country"].astype(int)
    V["country_simple"] = V.loc[:,"country"].map(lambda x : "GB" if int(x)==5 else "notGB")
    V.loc[:,"country"] = V.loc[:, "country"].map(lambda x : COUNTRY_CODES[int(x)])
    display(V.head(2))
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
        
    if NETWORK_NAME == "BIOTECH_SEC":
        print("reducing network BIOTECH_SEC")
        
    
    if LIMIT_CONTROL:
        # sorry about the name, should be 'source is ...'
#         NETWORK_NAME += "_sc_targetisallother"
        NETWORK_NAME += "_sc_targetisNONGB"
#         NETWORK_NAME += "_sc_targetisall"
        #V_source = V.loc[~V["is_financial"],"company_id"].astype(int).unique()
        #print("source:", V_source.shape)
        #V_target = V.loc[V["is_biotech"],"company_id"].astype(int).unique()
        #print("target:", V_target.shape)
        # let's try something else!
        
        #source_sel = V.loc[:,"country_simple"] != "GB"
        #source_sel = V.index
        # always remains the same:
        target_sel = V.loc[:,"is_biotech"].astype(bool) & (V.loc[:,"country_simple"] == "GB")
        if NETWORK_NAME.endswith("targetisNONGB"):
            source_sel = V.loc[:,"country_simple"] != "GB" # non GB
        elif NETWORK_NAME.endswith("targetisall"):
            source_sel = V.index
        elif NETWORK_NAME.endswith("targetisallother"):
            source_sel = ~target_sel
        else:
            raise ValueError("unknown network name source")
        V_source = V.loc[source_sel,"company_id"].astype(int).unique()
        print("source:", V_source.shape)
        V_target = V.loc[target_sel,"company_id"].astype(int).unique()
        print("target:", V_target.shape)

print(NETWORK_NAME)
assert len(values)>0, "?"
assert len(edges)>0, "?"
# -

V_source

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
    plt.ylabel(col.replace("_","-"))
    plt.xscale('log')
#     plt.yscale('log')
    plt.show()

# +
from src.network import *

print("Edge example:", [u for u in G.edges(data=True)][0])

g = Network( # ordered according to G.nodes()
        nx.adjacency_matrix(G, weight="weight"), 
        value=np.array([values[n]["value"] for n in G.nodes()]), 
        node_list=list(G.nodes())
)

figsize=(10,10)

if g.number_of_nodes < 50:
    print("Weight values", np.unique(np.asarray(nx.adjacency_matrix(G, weight="weight").todense()).flatten()))

print(g.number_of_nodes, g.number_of_edges)
print("-"*100)
g.educated_value_guess()
print("ZEROS")
g.draw_zeros(figsize=figsize, layout=LAYOUT)
print("NANS")
g.draw_nans(figsize=figsize, layout=LAYOUT)
print("="*100)
if g.number_of_nodes < 10000:
    print("Value graph")
    g.draw(color_arr=g.value, figsize=figsize, layout=LAYOUT)
print("dropping nans")
# TODO: better handling of remaining nans: what are these?
g = g.dropna_vals()
# print("dropping uncontrollable")
#g = g.remove_uncontrollable()

print(g.number_of_nodes, g.number_of_edges)
if g.number_of_nodes < 10000:
    print("Value graph")
    g.draw(color_arr=g.value, figsize=figsize, filename="figs/{}.pdf".format(NETWORK_NAME), show_edge_values=True, layout=LAYOUT)
    
g = g.to(device)
# -


g.V

g.node_list

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
          source_mask=source_mask, target_mask=target_mask, layout=LAYOUT)
    print("Number of source nodes:", torch.sum(source_mask))
    print("Number of target nodes:", torch.sum(target_mask))
# -

plt.hist(g.compute_total_value(only_network_shares=True, include_root_shares=True).detach().cpu().numpy())
plt.hist(g.compute_total_value(only_network_shares=True, include_root_shares=True, sel_mask=source_mask).detach().cpu().numpy())
plt.yscale('log')
plt.ylabel("euro")
plt.show()

# save data
with open("data_network_{}.pickle".format(NETWORK_NAME), 'wb') as f:
    pickle.dump((source_mask, target_mask, g, NETWORK_NAME), f)

# +
from collections import defaultdict
from src.optim import *
from src.vitali import *
from src.loss import *
from src.network import *

number_of_sources = g.number_of_nodes if source_mask is None else sum(source_mask)
print(g.number_of_nodes, number_of_sources)
cl = get_cl_init(number_of_sources, device=device, loc=-8)
cl_soft = torch.sigmoid(cl)
print(torch.min(cl_soft), torch.max(cl_soft))
init_lr = 1#0.01
decay = 0.1
max_steps = 3000
lambd = 1#1 #0#0.75#0.75#1.38949549e-02 #1.93069773e-01#0#1.00000000e+01#5.17947468e-02 #1e-3
print("lambda = ",lambd)
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
                             loss_tol=1e-8,
                             save_separate_losses=True,
                             source_mask=source_mask,
                             target_mask=target_mask
                            )
# -

print(hist["time"], g.number_of_nodes)

if source_mask is not None:
    print(torch.sum(g.compute_total_value(only_network_shares=True, include_root_shares=True)))
Vtot = g.compute_total_value(only_network_shares=True, include_root_shares=True, sel_mask=source_mask)
print(torch.sum(Vtot))

hist["final_control"]

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
plt.ylabel("loss control")
plt.show()

plt.plot(hist["loss_cost"])
plt.yscale('log')
plt.xlabel("step")
plt.ylabel("loss cost")
plt.show()

# -


plt.plot(hist["final_ol"])

# +
num_params = hist["params_sm"][0].shape[0]
print(num_params)

plt.figure()
for loss_name, loss_arr in {k:v for k,v in hist.items() if (k.startswith("loss_") and k.endswith("arr"))}.items():
    print(loss_name)
    for i in range(loss_arr[0].shape[0]):
        loss_tr = lambda x : x
        if loss_name == "loss_control_arr":
            loss_tr = lambda x : -x+1
        plt.plot([loss_tr(loss_arr[t][i]) for t in range(len(loss_arr))], label=str(i))
    plt.xlabel("step")
    if loss_name == "loss_control_arr":
        plt.ylabel(r"control over i")
    else:
        plt.ylabel(r"cost of buying i")
        plt.yscale('log')
    if num_params <= 15:
        plt.legend()
    my_savefig("figs/{}_{}_{}_ifo_step.pdf".format(NETWORK_NAME, lambd, loss_name))
    plt.show()


plt.figure()
for i in range(num_params):
    plt.plot([hist["params_sm"][t][i] for t in range(len(hist["params_sm"]))], label=str(i))
plt.xlabel("step")
plt.ylabel(r"parameter (sigmoid) = percent bought of !available! company i stock")
plt.yscale('log')
if num_params <= 15:
    plt.legend()
my_savefig("figs/{}_{}_sigmoid_ifo_step.pdf".format(NETWORK_NAME, lambd))
plt.show()

plt.figure()
for i in range(num_params):
    plt.plot([hist["ol"][t][i] for t in range(len(hist["ol"]))], label=str(i))
plt.xlabel("step")
plt.ylabel(r"o = percent bought of company i")
plt.yscale('log')
if num_params <= 15:
    plt.legend()
my_savefig("figs/{}_{}_totshares_ifo_step.pdf".format(NETWORK_NAME, lambd))
plt.show()
# -


hist

hist["ol"][0].shape

g.total_shares_in_network

sum(source_mask)

# +
#print(source_mask, target_mask)

plot_direct_control(
        g,
        hist["final_ol"],
        source_mask=source_mask,
        target_mask=target_mask,
        figsize=(15, 5),
        total_shares_in_network=hist["total_shares_in_network"],
        filename="figs/o_distr_{}_{}.pdf".format(NETWORK_NAME, lambd)
        )

plot_control_distr(
        g,
        hist["final_loss_control_arr"],
        ol=hist["final_ol"],
        source_mask=source_mask,
        target_mask=target_mask,
        cutoff=None,
        figsize=(15, 2.5),
        total_shares_in_network=hist["total_shares_in_network"],
        filename="figs/control_distr_{}_{}.pdf".format(NETWORK_NAME, lambd)
        )
# -

g.V

if num_params <= 1000:
    kwargs = {"show_edge_values":False, "show_node_names":False}
    figsize=(10,10)
    print("direct direct control o")
    direct_control=hist["final_ol"]
    g.draw(external_ownership=direct_control, vmin=0, vmax=1, figsize=figsize, layout=LAYOUT, **kwargs)

    print("cost of buying")
    final_cost=pad_from_mask(hist["final_loss_cost_arr"], source_mask)
    g.draw(color_arr=final_cost, vmin=0, vmax=1, figsize=figsize, layout=LAYOUT, **kwargs)
    
    print("total/propagated control")
    final_control = pad_from_mask(1-hist["final_loss_control_arr"], target_mask)
    #print(final_control)
    g.draw(color_arr=final_control, vmin=0, vmax=1, figsize=figsize, layout=LAYOUT, **kwargs)
    
    print("size = total control, color = cost")
    print(hist["final_loss_cost"].shape)
    g.draw(color_arr=final_cost, size_arr=final_control, vmin=0, vmax=1, figsize=figsize, filename="figs/{}_{}_size_control_color_cost.pdf".format(NETWORK_NAME, lambd),
          target_mask=target_mask, layout=LAYOUT, **kwargs)
#     g.draw(color_arr=final_cost, size_arr=final_control, vmin=0, vmax=1, figsize=figsize, filename="figs/{}_{}_size_control_color_cost.pdf".format(NETWORK_NAME, lambd),
#       layout=LAYOUT, **kwargs)

sum(hist["final_loss_control_arr"])

sum(target_mask)

print(
    "Fraction of control: ", float(hist["final_control"])
)


# +

def make_groupby_network(g, V, ol, target_mask, column="country", show=True, ax=None):
    node_list = list(g.node_list.numpy())
    #print(node_list)
    bar_list = []
    ol = torch.Tensor(ol)
    control = compute_control_with_external(ol, g, as_matrix=True)
    tot_control = control[:, target_mask].sum()
    print("total_control = ", tot_control)
    for k, v in V.groupby(column):
        print(k, v.shape)
        if v.shape[0] == 0: continue
        sel_NL = list(v["company_id"].values.astype(int))
        sel_NL = [x for x in sel_NL if x in node_list]
        sel_NL_idx = g.get_indices(sel_NL)
        if len(sel_NL) == 0: continue
        control_value = control[sel_NL_idx,:].sum(axis=0)
        control_value = control_value[target_mask]
        control_value = control_value.sum() / tot_control
        print("control_value = ", control_value)
        bar_list.append((k, control_value))
        # do the aggregation
    #print(new_NL)
    bar_list = dict(bar_list)
    bar_list = {k: v for k, v in sorted(bar_list.items(), key=lambda item: -item[1])}
    if ax is None:
        fig, ax = plt.subplots()    
    ax.bar(list(bar_list.keys()), list(bar_list.values()))
    ax.set_yscale('log')
    ax.set_ylabel("control on GB biotech")
    ax.set_xlabel("country")
    ax.set_ylim(0.01, tot_control)
    if show:
        plt.show()


print(NETWORK_NAME)
if "BIOTECH_SEC_sc" in NETWORK_NAME:
    make_groupby_network(g, V, hist["final_ol"], target_mask)     
# -

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

cs = []
ss = []
control_values = []
param_result = []
hists = []
if NETWORK_NAME == 'SIMPLE_STAR':
    lambd_range = [ 0, 0.50, 0.6, 0.75, 1, 1.1]
else:
#     lambd_range = [0] + list(np.logspace(-3, 2, num=20))
    lambd_range = [0] + np.logspace(-3, -1, num=20) # source non GB
#     lambd_range = np.logspace(-4, 2, num=5)#np.logspace(-2, 1, num=10)
# lambd_range = np.linspace(0, 1, num=20)#np.logspace(-2, 1, num=10)
print("lambdas to evaluate:", lambd_range)
number_of_sources = g.number_of_nodes if source_mask is None else sum(source_mask)
max_steps = 3000
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
        control_values.append(hist["final_control"])
        #print(lambd, param_result[-1], c, s)
        if num_params <= 15:
            tot_shares = g.total_shares_in_network.detach().cpu().numpy()
            tot_shares[g.identify_uncontrollable().detach().cpu().numpy()] = 1
            print(tot_shares*hist["final_params_sm"])
            g.draw(external_ownership=tot_shares*hist["final_params_sm"])
        hists.append(hist)
with torch.no_grad():
    cs = np.array([x.numpy() for x in cs])
    ss = np.array([x.numpy() for x in ss])
    param_result = torch.sigmoid(torch.stack(param_result)).numpy()

# +
fig, ax1 = plt.subplots()
print(lambd_range)
if weight_control:
    raise NotImplemented("todo")
#     y_control = (torch.sum(g.value).detach().cpu().numpy()-cs)/torch.sum(g.value).detach().cpu().numpy()
    #print(y_control)
else:
    #number_of_target_nodes = int(sum(target_mask) if target_mask is not None else g.number_of_nodes) 
    #print("targets:", number_of_target_nodes)
    #y_control = (number_of_target_nodes-cs)#/g.number_of_nodes
    y_control = control_values

ax1.plot(lambd_range, y_control, label="control", color='blue')
ax1.scatter(lambd_range, y_control, color='blue')
ax1.set_ylabel("control", color='blue')
ax1.set_xscale('log')
ax1.set_yscale('log')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.plot(lambd_range, ss, label="total budget", color='red')
ax2.scatter(lambd_range, ss, color='red')
ax2.set_ylabel("total budget (€1M)", color='red')
ax2.set_yscale('log')
ax1.set_xlabel("lambda")
plt.legend()

my_savefig("figs/{}_lambda_curve_bis.pdf".format(NETWORK_NAME))
plt.show()

with open("data_for_{}_lambda_curve.pickle".format(NETWORK_NAME), 'wb') as f:
    pickle.dump((g, y_control, ss, lambd_range, NETWORK_NAME, hists, ss, cs), f)
plt.plot(y_control, ss, color='blue', lw=2)
plt.scatter(y_control, ss, color='blue')
plt.ylabel("cost (€1M)")
plt.xlabel("control")
for i in range(len(lambd_range)):
    print(i)
    #if i not in (4,8,9,10): continue
    plt.text(y_control[i]+0.5, ss[i], r"   $\lambda={:.2f}$   ".format(lambd_range[i]), va='top', ha='right' if i < len(lambd_range)/2-2 else 'left')
# plt.xscale('log')
plt.yscale('log')
my_savefig("figs/{}_lambda_curve.pdf".format(NETWORK_NAME))
plt.show()

plt.plot(ss, y_control, color='blue', lw=2)
plt.xlabel("cost (€1M)")
plt.ylabel("control")
plt.xscale('log')
for i in range(len(lambd_range)):
    plt.text(ss[i], y_control[i], r"   $\lambda={:.2f}$   ".format(lambd_range[i]), va='top', ha='right' if i < len(lambd_range)/2-2 else 'left')
#plt.yscale('log')
my_savefig("figs/{}_control_vs_cost_curve.pdf".format(NETWORK_NAME))

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
#     source_mask=source_mask, 
#     target_mask=target_mask,
    colorbar_scale='log',
    colorbar_text='value (€1M)',
    rescale_color=False,
    rescale_size=True,
    layout=LAYOUT
)

print(NETWORK_NAME)

# +
from collections import defaultdict
from src.optim import *
from src.vitali import *
from src.loss import *
from src.network import *

print(NETWORK_NAME)
print(g.number_of_nodes)
#budget = 10000
# budget = 100000
number_of_sources = g.number_of_nodes if source_mask is None else sum(source_mask)    
source_values = g.compute_total_value(only_network_shares=True, include_root_shares=True, sel_mask=source_mask)[source_mask]
target_values = g.compute_total_value(only_network_shares=True, include_root_shares=True, sel_mask=target_mask)[target_mask]
total_source_value = source_values.sum()
total_target_value = target_values.sum()
print("total value of the targets:", total_target_value)
budget = float(total_target_value) #1000
print("target value:", g.compute_total_value(only_network_shares=True, include_root_shares=True, sel_mask=target_mask).sum())
print("with budget:", budget)
source_value_ratio = budget/total_source_value
print("need ratio:", source_value_ratio)
guess = torch.clamp(source_value_ratio/source_values, min=1e-8, max=1-1e-8)
guess = torch.log(guess/(1-guess))
#print("guess:", guess)
cl = get_cl_init(number_of_sources, device=device, loc=-10, vals=guess)
print(number_of_sources)
cl_soft = torch.sigmoid(cl)
print(torch.min(cl_soft), torch.max(cl_soft))
# init_lr = 0.001
init_lr = 1
decay = 0.1
num_steps = 3000
# num_steps = 100
max_iter = 20
use_schedule = True#False
lr = init_lr
scheduler = None
if use_schedule: # make new copy
    scheduler = lambda opt : torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[2000], gamma=decay)

#clip_value = 0.0001
#cl.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    
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
        loss_tol = 1e-6,
        source_mask=source_mask, target_mask=target_mask,
        es_wait=10
        )
print("DONE!")

# +
#sum(g.identify_uncontrollable())
# -

print("max control:", sum(target_mask), sum(g.total_shares_in_network*target_mask), sum(target_mask) - sum(g.total_shares_in_network*target_mask))

# +
plt.scatter(hist_all["step_nr"], hist_all["loss_control"])
plt.xlabel("i")
plt.ylabel("loss contr")
plt.show()

plt.scatter(hist_all["step_nr"], hist_all["loss_cost"])
plt.xlabel("i")
plt.ylabel("loss cost")
plt.show()

# plt.scatter(hist_all["step_nr"], np.sum(hist_all["loss_cost"]))
# plt.xlabel("i")
# plt.ylabel("loss cost")
# plt.show()

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
     
g.draw(color_arr=final_control, figsize=figsize, 
       filename="figs/{}_size_control_color_cost_budget_{}.pdf".format(NETWORK_NAME, budget),
      target_mask=target_mask, source_mask=source_mask, colorbar_text="control", layout=LAYOUT)
# -


plt.hist(hist_all["final_ol"], bins=100)
plt.show()

g.value

print(sum(final_cost))
print(sum(final_control))

# +
#g.compute_total_value(only_network_shares=True, include_root_shares=True, sel_mask=target_mask)[target_mask]

# +
#g.node_list.detach().cpu().numpy()[target_mask]

# +
#plt.hist(g.value, bins=100)
#plt.xlim(0,1000)
#plt.show()

#print(g.compute_total_value(only_network_shares=True, include_root_shares=True, sel_mask=target_mask))

plt.bar(
    ["{}".format(x) for x in g.node_list.detach().cpu().numpy()[target_mask]], 
    g.compute_total_value(only_network_shares=True, include_root_shares=True, sel_mask=target_mask)[target_mask]
)
plt.yscale('log')
plt.show()


# log-scaled bins
bins = np.logspace(np.log(min(g.value)), np.log(max(g.value)), 50)
#print(bins)
widths = (bins[1:] - bins[:-1])

# Calculate histogram
hist = np.histogram(g.value, bins=bins)
# normalize by bin width
hist_norm = hist[0]/widths

# plot it!
plt.bar(bins[:-1], hist_norm, widths)
plt.xscale('log')
plt.yscale('log')
plt.show()
# -

#len(hist_all["final_ol"])
#len(source_mask)
pad_from_mask(hist_all["final_loss_control_arr"], target_mask)

hist_all["final_ol"]

# +
plot_direct_control(
        g,
#         pad_from_mask(hist_all["final_loss_cost_arr"], source_mask),
        hist_all["final_ol"],
        source_mask=source_mask,
        target_mask=target_mask,
        figsize=(20, 5),
        total_shares_in_network=hist_all["total_shares_in_network"],
        filename="figs/o_distr_{}_budget{}.pdf".format(NETWORK_NAME, budget)
        )

plot_control_distr(
        g,
        hist_all["final_loss_control_arr"],
        ol=hist_all["final_ol"],
        source_mask=source_mask,
        target_mask=target_mask,
        cutoff=None,
        figsize=(20, 5),
        total_shares_in_network=hist_all["total_shares_in_network"],
        filename="figs/control_distr_{}_budget{}.pdf".format(NETWORK_NAME, budget)
        )
# -

import pickle
with open('dump_network{}_budget{}.pickle'.format(NETWORK_NAME, budget), 'wb') as handle:
    pickle.dump((g, V, param_est, loss_vals, constr_vals, hist_all, target_mask, source_mask), handle)

sum(source_mask & target_mask)

reduce_from_mask(hist_all["final_ol"], target_mask)

reduce_from_mask(hist_all["final_ol"], source_mask)

reduce_from_mask(hist_all["final_ol"], source_mask & target_mask)

print(NETWORK_NAME)
if "BIOTECH_SEC_sc" in NETWORK_NAME:
    print(sum(1-hist_all["final_loss_control_arr"]))
    make_groupby_network(g, V, hist_all["final_ol"], target_mask)     

#






