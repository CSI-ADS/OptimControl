# Network control by a constrained external agent as a continuous optimization problem

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<!-- [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/1234.56789) -->

Code for the paper on "Network control by a constrained external agent as a continuous optimization problem", Nys et al. (2021)

## How to use

To run the example in analyse.py
* install anaconda
* create the conda environment from environment.yml
* src folder contains the code
* analyse.py reproduces some of the figures in the paper
* open analyse.py with jupyter notebook (install jupytext to open the .py files as jupyter notebooks)


## Core

Note: figures will be dumped into the figs/ folder.

### Networks

The core object in the code is defined in network.py. It should be initialized as (schematically)
```python
from src.network import Network
g = Network(
        adjacency_matrix,
        value=company_value_array,
        node_list=node_id_list
)
```
It implements useful functions such as `Network.educated_value_guess` to impute missing values. It also implements plotting functions to visualize, for example, remaining nans: `Network.draw_zeros`.

For targeted optimization (as well as source optimization), one must construct a mask
```python
from src.network import make_mask_from_node_list
target_mask = make_mask_from_node_list(g, node_id_list)
```
These can also be visualized using the kwargs `source_mask` and `source_mask` in `Network.draw`.

Some example networks are implemented in `src.test_networks.py`.


### Optimization and loss function

The core of the optimization algorithms are implemented in `loss.py` and `optim.py`. The most important functions begin
```python
from src.optim import optimize_control
from src.loss import compute_sparse_loss
params, loss, hist = optimize_control(compute_sparse_loss, pu, g, budget, return_hist=False)
```
for uncontrolled optimization, and
```python
from src.optim import constraint_optimize_control
from src.loss import compute_sparse_loss
params, loss, hist = constraint_optimize_control(compute_sparse_loss, pu, g, budget, return_hist=False)
```
for controlled optimization. Here, `compute_sparse_loss` implements the loss functions in our paper.
We capture a large amount of statistics during the optimization and store it in `hist`.
These functions have many adjustable parameters, such as
* Verbosity level (amount of print output) - verbose
* To store and return the history - return_hist
* The learning rate - lr
* Number of optimization steps - num_steps
* And many more.

### Backbone algorithm

We implement Vitali et al as a backbone algorithm (see paper for details and citation).
One can compute the control with the vitali algorithm

```python
from src.vitali import compute_control

c = compute_control(o, B)
```
where B is the adjusted ownership matrix for nodes reachable from node x. Here, c represents the control of x over all nodes in the network.

### Automatic differentiation

As long as you procede as above, we handle the automatic differentiation for you with the following functions in `optim.py`:

```python
from src.optim import compute_value, compute_value_and_grad, update

def compute_value(fn, cl, g, *args, **kwargs):
    # this converts some continuous unbounded variables to bounded ones
    ol = compute_total_shares(cl, g, source_mask=kwargs.get("source_mask"))
    return fn(ol, g, *args, **kwargs)

def compute_value_and_grad(fn, cl, g, *args, **kwargs):
    value = compute_value(fn, cl, g, *args, **kwargs)
    value.backward()
    grads = cl.grad
    return value, grads

def update(loss_fn, optimizer, params, *args, **kwargs):
    optimizer.zero_grad()
    params = get_params(optimizer)
    cost, grads = compute_value_and_grad(loss_fn, params, *args, **kwargs)
    optimizer.step()
    return params, cost
```
Here, `cl` corresponds to `p_u` in our paper, and is converted into `o` through `compute_total_shares`.
More generally, the argument `loss_fn` can be a custom loss function, as long as it is differentiable and written in PyTorch.

### Plotting results

Results can be visualized with the drawing functions in `src.network.py`.

The produced plots should look like, for example (see SM):

<!-- ![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true) -->
![alt text](https://github.com/CSI-ADS/OptimControl/blob/online/figs/example_fig.png?raw=true)

Furthermore, `src.plotting.py` also implements `plot_direct_control` and `plot_control_distr` based on the obtained graph and control vector `o`.

## GPU

This code runs smoothly on a GPU! Just put the network (see below) and initial tensors on the right device.
You can use the following code to determine the devices that are available
```python
import torch
device = 'cpu'
if torch.cuda.is_available():
    print("CUDA available")
    device = 'cuda'
print(device)
```
This defaults to a CPU device, and only switches to a GPU is one is available. Notice that the code speeds up significantly on a GPU.
To smoothly move the network to the GPU, we've made the following possible:
```
g = g.to(device)
```
where `g` is a `Network` defined earlier. Notice, however, that the current implementation uses dense matrices, since PyTorch's sparse matrices are still under active development. We are also expecting that `jax` will soon become a relevant tool for this.

## Cite

```python
@article{nys2021network,
  title={Network control by a constrained external agent as a continuous optimization problem},
  author={Nys, Jannes and Heuvel, Milan van den and Schoors, Koen and Merlevede, Bruno},
  journal={arXiv preprint arXiv:2108.10298},
  year={2021}
}
```
