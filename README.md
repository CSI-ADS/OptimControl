# Optim

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

## Cite

```python
@article{nys2021network,
  title={Network control by a constrained external agent as a continuous optimization problem},
  author={Nys, Jannes and Heuvel, Milan van den and Schoors, Koen and Merlevede, Bruno},
  journal={arXiv preprint arXiv:2108.10298},
  year={2021}
}
```
