import torch
import numpy as np
import networkx as nx
import scipy
import matplotlib.pyplot as plt

class Network:

    def __init__(self, A, value=None, node_list=None, dtype=torch.float):
        A = scipy.sparse.csr_matrix(A)
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        if isinstance(node_list, list):
            node_list = torch.from_numpy(np.array(node_list))
        if isinstance(node_list, np.ndarray):
            node_list = torch.from_numpy(node_list)
        assert value is None or isinstance(value, torch.Tensor), "use torch, found {}".format(type(value))
        if value is not None:
            value = value.to(dtype)
        self.N = A.shape[0]
        self.set_control_matrix(A, dtype=dtype)
        self.V = value # value of each node
        self.node_list = node_list
        self.A = A
        #self.set_reachable(A)

    @classmethod
    def from_nx(cls, G, value_key="value", ownership_key="ownership", default_value=1, **kwargs):
        A = nx.adjacency_matrix(G)
        nodes = dict(G.nodes(data=value_key, default=default_value))
        node_list = list(nodes.keys())
        value = list(nodes.values())
        return Network(A, value=value, node_list=node_list, **kwargs)


    # def set_reachable(self, A, tol=1e-8):
    #     D = torch.abs(torch.matrix_power(A, self.N))
    #     D = D - torch.diag(torch.diag(D))
    #     self.D = D > tol

    def set_control_matrix(self, A, dtype=torch.float):
        if scipy.sparse.issparse(A):
            A = torch.from_numpy(A.todense())
        if isinstance(A, np.ndarray):
            A = torch.from_numpy(A)
        self.C = A.to(dtype)
        #self.C.fill_diagonal_(1)

    @property
    def ownership(self):
        return self.C

    @property
    def value(self):
        return self.V

    @property
    def number_of_nodes(self):
        return self.N

    @property
    def number_of_edges(self):
        return self.A.count_nonzero()

    def compute_total_value(self, only_network_shares=True, include_root_shares=True, sel_mask=None):
        if self.value is None:
            return None
        V = self.value.clone()
        if only_network_shares:
            shares = self.total_shares_in_network
            if include_root_shares:
                contr = self.identify_controllable()
                V[contr] *= shares[contr] # keep the rest at full value
            else:
                V *= shares # roots are put to 0
        if sel_mask is not None:
            V[~sel_mask] = 0
        return V

    @property
    def nodes(self):
        return self.node_list if self.node_list is not None else torch.arange(self.N)

    @property
    def total_shares_in_network(self):
        return self.C.sum(axis=0)

    @property
    def device(self):
        return self.C.device

    def to(self, device):
        self.C = self.C.to(device)
        if self.V is not None:
            self.V = self.V.to(device)
        return self

    @property
    def number_of_controllable(self):
        return sum(self.identify_controllable())

    def identify_controllable(self):
        Ctot = self.total_shares_in_network
        contr = Ctot > 1e-8
        return contr

    # func 1
    def identify_uncontrollable(self):
        return ~self.identify_controllable()

    def network_selection(self, sel):
        print("Keeping {} out of {}".format(sum(sel), self.number_of_nodes))
        A = self.A[sel,:][:,sel]
        V = None if self.value is None else self.value[sel]
        node_list = None if self.node_list is None else self.node_list[sel]
        dtype = self.C.dtype
        return Network(A, value=V, node_list=node_list, dtype=dtype)

    # func 2
    def remove_uncontrollable(self):
        assert False, "better not to do this, since we generate new root nodes by dropping, better to determine the desc"
        contr = self.identify_controllable()
        tot_contr = sum(contr)
        if tot_contr == 0:
            return self
        return self.network_selection(contr)

    def educated_value_guess(self):
        nan_idx = torch.isnan(self.V)
        children_idx = self.A[nan_idx,:].tocsr().nonzero()
        vnew = np.zeros((sum(nan_idx),)) # stores values for the nans
        for i, u in enumerate(np.where(nan_idx)[0]):
            idx = children_idx[1][children_idx[0] == i]
            children_vals = self.V[idx].detach().cpu().numpy()
            ownership = np.array(self.A[u, idx].todense()).flatten()
            vnew[i] = np.sum(children_vals*ownership) # this is a guess
        self.V[nan_idx] = torch.tensor(vnew).to(self.V.dtype)

    def dropna_vals(self):
        assert self.V is not None, "cannot drop nans if no values are given"
        sel = ~torch.isnan(self.V)
        if sum(sel) == self.number_of_nodes:
            return self
        return self.network_selection(sel)

    def draw_nans(self, **kwargs):
        if self.V is None:
            print("no values to draw")
            return
        nans = torch.isnan(self.V).float().detach().cpu().numpy()
        self.draw(color_arr=nans, rescale=False, scale_size=False, **kwargs)

    def draw_zeros(self, **kwargs):
        if self.V is None:
            print("no values to draw")
            return
        zeros = (self.V == 0).float().detach().cpu().numpy()
        self.draw(color_arr=zeros, rescale=False, scale_size=False, **kwargs)

    def draw(self, external_ownership=None, color_arr=None, size_arr=None, edge_arr=None, rescale=True,
            scale_size=True, scale_color=True, scale_edge=True, show_edge_values=True, **kwargs):
        node_list = dict(zip(np.arange(self.number_of_nodes), self.nodes.detach().cpu().numpy()))
        V = self.V.detach().cpu().numpy()
            #nx.set_node_attributes(G, values, name="value")
            #nx.set_node_attributes(G, eo, name="external_ownership")
        #pos=nx.spring_layout(G)

        node_color = None #default
        if scale_color:
            if external_ownership is not None:
                eo = dict(zip(np.arange(self.number_of_nodes), external_ownership))
                node_color = np.array(list(eo.values()))
            if color_arr is not None:
                # print(color_arr)
                assert len(color_arr) == self.number_of_nodes
                node_color = np.array(color_arr)
            if rescale:
                node_color = node_color / np.nanmax([1, np.nanmax(node_color)])
                node_color = np.clip(node_color, 0.01, None)
        #print(node_color)

        node_size = None # default
        if scale_size:
            if V is not None:
                values = dict(zip(np.arange(self.number_of_nodes), V))
                node_size = np.array(list(values.values()))
            if size_arr is not None:
                assert len(size_arr) == self.number_of_nodes
                node_size = np.array(size_arr)
            if rescale:
                node_size = node_size / np.nanmax([1, np.nanmax(node_size)])
                node_size = np.clip(node_size, 0.01, None)
                node_size *= 300

        edge_width = None
        if scale_edge:
            weights = dict(self.A.copy().todok().items())
            edge_width = np.array(list(weights.values()))
            if edge_arr is not None:
                edge_width = np.array(edge_arr)


        #print(node_size)
        with_labels = self.number_of_nodes < 50
        node_labels = node_list if with_labels else None

        draw_nx_graph(
            self.A,
            with_labels=with_labels,
            node_labels=node_labels,
            node_color=node_color,
            node_size=node_size,
            edge_width=edge_width,
            show_edge_values=show_edge_values,
            **kwargs
            )

def draw_nx_graph(
        A,
        with_labels=True, node_labels=None,
        node_color=None,
        node_size=None,
        edge_width=None,
        show_edge_values=True,
        show=True,
        figsize=(20,20),
        filename=None,
        **kwargs):
    # nx defaults
    if node_color is None:
        node_color = "#1f78b4"
    if node_size is None:
        node_size = 300
    if edge_width is None:
        edge_width = 1.0

    # networkx
    G = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph)
    pos = nx.nx_pydot.pydot_layout(G, prog='neato', root=None)
    plt.figure(figsize=figsize, frameon=False)
    nx.draw_networkx(
        G,
        pos=pos, arrows=True, with_labels=with_labels,
        labels=node_labels,
        node_color=node_color,
        node_size=node_size,
        width=edge_width*2,
        **kwargs
        )
    number_of_edges = A.count_nonzero()
    if show_edge_values and number_of_edges < 50:
        edge_labels = nx.get_edge_attributes(G, "weight")
        edge_labels = {k:"{:.2f}".format(v) for k,v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    cut = 1.05
    xmax= cut*max(xx for xx,yy in pos.values())
    ymax= cut*max(yy for xx,yy in pos.values())
    plt.xlim(0,xmax)
    plt.ylim(0,ymax)
    plt.margins(0,0)


    if filename is not None:
        plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0)
    if show:
        plt.show()

def make_mask_from_node_list(g, node_list, **kwargs):
    l = set(node_list)
    return torch.tensor([(x in l) for x in g.node_list.detach().cpu().numpy()], **kwargs)

def make_mask(g, idx, **kwargs):
    mask = torch.zeros((g.number_of_nodes,), **kwargs)
    return mask.scatter_(0, torch.tensor(idx), 1)

def normalize_ownership(g):
    C = g.ownership
    N = C.shape[0]
    Csum = C.sum(axis=0)
    Csum_zero = (Csum == 0).to(C.dtype)
    Csum = Csum + Csum_zero
    C = C / Csum.reshape((1, N))
    return C


def adjust_for_external_ownership(cl_desc, g, source_mask=None):
    assert torch.min(cl_desc) >= 0 and torch.max(cl_desc) <= 1, "cl was outside bounds"
#     N = C.shape[0]
    C = g.ownership.clone()
    # cl_desc might be too short
    if source_mask is not None:
        cl = torch.zeros((g.number_of_nodes,), device=cl_desc.device, dtype=cl_desc.dtype)
        cl[source_mask] = cl_desc
    else:
        cl = cl_desc

    non_root_nodes = g.identify_controllable()
    # C
    #print(C)
    # renormalize
    # assumption: we will take shares from other proportional to the amount of shares
    # e.g. 10% of shares, is 10% from all owners
    #print("C = ", C)
    #print(non_root_nodes)
    #print(cl)
    #print("Cnr = ", C[:,non_root_nodes], (1.0 - cl[non_root_nodes]))
    C[:,non_root_nodes] = C[:,non_root_nodes] * (1.0 - cl[non_root_nodes])
    #print("C = ", C)
#     C = torch.cat((cl.reshape(1, N), C), axis=0)
#     Csum = C.sum(axis=0)
#     #numerical issues
#     Csum_zero = (Csum == 0).double()
#     Csum = Csum + Csum_zero
#     #print(Csum)
#     C = C / Csum.reshape((1, N))
#     C = C[1:,:]
    return C
