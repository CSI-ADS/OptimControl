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

# %load_ext autoreload
# %autoreload 2

device = 'cpu'
if torch.cuda.is_available():
    print("CUDA available")
    device = 'cuda'
print(device)
# -

# # Create example network with some values

# +
from src.network import *

N = 1000
G = nx.fast_gnp_random_graph(N, 0.2, directed=True)
value = torch.Tensor(list(nx.degree_centrality(G).values()))
print(value)
for i, j, _ in G.edges(data=True):
    if (i,j) not in G.edges().keys():
        continue
    G.edges[i,j]['weight'] = np.random.uniform(high=1/G.number_of_nodes())
A = nx.adjacency_matrix(G).todense()
assert np.min(A) >= 0 and np.max(A) <= 1, "{} | {}".format(np.min(A), np.max(A))


# normalize columns
#print("Au",A)
# A = normalize_ownership(torch.from_numpy(A))
# G = nx.from_numpy_array(A.numpy())

#nx.draw(G)
plt.matshow(A)
plt.colorbar()
plt.show()
print("A = ",A)
g = Network(A, dtype=torch.float)

print("normalizing")
C = normalize_ownership(g)
g = Network(C, dtype=torch.float)

#D = g.D
C = g.C
print("C = ", C, C.dtype, C.device)
#nx.draw(G)
#desc = np.arange(0, D.shape[0])

# +
""" Example of how to compute the loss function"""

from src.optim import *
from src.vitali import *
from src.loss import *

# initialize the stocks that I have (logits)
print(g.number_of_nodes)
""" 
    Note: cl is not the actual percentage, 
    it's rather a value between -+\infty that is rescaled later 
    using sigmoid to get something between 0 and 1
"""
cl = get_cl_init(g.number_of_nodes, device=device, dtype=torch.float)
print(cl.dtype, cl.device)
g.to(device)
print(compute_sparse_loss_cache_vars(cl, g, lambd=0.1))
print(cl, cl.dtype, cl.device)
print("-"*10)
print(compute_value_and_grad(compute_sparse_loss_cache_vars, cl, g, lambd=0.1))

# +
""" Example of control optimization with lambda=0, so no cost of stocks"""
from collections import defaultdict
from src.vitali import *

cl = get_cl_init(g.number_of_nodes, device=device)
_, _, hist = optimize_control(compute_sparse_loss_cache_vars, cl, g, lambd=0, return_hist=True, verbose=True, num_steps=100,
                             device=device)

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

# # Optimization for multiple lambda: balancing cost with control

# +
cs = []
ss = []
param_result = []
lambd_range = np.linspace(0, 5, num=20)#np.logspace(-2, 1, num=10)
for lambd in lambd_range:
    cl = get_cl_init(g.number_of_nodes, device=device)
    loss_fn = compute_sparse_loss_cache_vars
    cl, cost = optimize_control(loss_fn, cl, g, lambd=lambd, return_hist=False, lr=0.1, num_steps=1000, verbose=True, device=device)
    
    
    # get some stats
    with torch.no_grad():
        c, s = compute_sparse_loss_cache_vars(cl, g, lambd=lambd, as_separate=True)
        param_result.append(cl.detach().cpu())
        cs.append(c.detach().cpu())
        ss.append(s.detach().cpu())
        print(lambd, c, s)
        
with torch.no_grad():
    cs = np.array([x.numpy() for x in cs])
    ss = np.array([x.numpy() for x in ss])
    param_result = torch.sigmoid(torch.stack(param_result)).numpy()
# -

plt.scatter(x=(g.number_of_nodes-cs)/g.number_of_nodes, y=ss)
plt.xlabel("Total control")
plt.ylabel("Total cost (budget)")
plt.show()

plt.plot(lambd_range, (g.number_of_nodes-cs)/g.number_of_nodes, label="control")
plt.plot(lambd_range, ss, label="total budget")
plt.legend()
plt.xlabel("lambda")
plt.show()

# +
if g.value:
    plt.matshow(g.value.reshape(1, -1))
    plt.colorbar()
    plt.show()

plt.matshow(param_result)
plt.colorbar()
plt.xlabel("parameters (% bought)")
plt.ylabel("lambda nr")
plt.show()
# -








