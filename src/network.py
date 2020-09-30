import torch
import numpy as np
import networkx as nx
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math
from .utils import *

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

    def identify_relevant(self, target_mask):
        G = nx.DiGraph(self.A)
        targets = np.where(target_mask)[0]
        anc = set(targets)
        for n in targets:
            anc = anc.union(nx.ancestors(G, n))
        anc = list(anc)
        print("anc = ", anc)
        mask = idx_to_mask(self.N, anc)
        print("mask = ", mask)
        return mask

    def remove_irrelevant(self, target_mask):
        if target_mask is None:
            return self
        anc = self.identify_relevant(target_mask)
        if sum(anc) == self.N:
            return self
        return self.network_selection(anc)


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

    def draw(self, external_ownership=None, color_arr=None, size_arr=None, edge_arr=None,
            rescale_size=True, rescale_color=False,
            scale_size=True, scale_color=True,
            scale_edge=True, show_edge_values=True, source_mask=None,
             target_mask=None, exclusion_mask=None, colorbar=True, colorbar_text='Cost', colorbar_scale=None, **kwargs):
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
            if rescale_color and node_color is not None:
                node_color = node_color / np.nanmax([1, np.nanmax(node_color)])
                node_color = np.clip(node_color, 0.01, None)
        # print('color:' ,node_color)

        node_size = None # default
        if scale_size:
            if V is not None:
                values = dict(zip(np.arange(self.number_of_nodes), V))
                node_size = np.array(list(values.values()))
            if size_arr is not None:
                assert len(size_arr) == self.number_of_nodes
                node_size = np.array(size_arr)
            if rescale_size and node_size is not None:
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

        if node_color is None:
            colorbar=False
            colorbar_text=None

        ax = draw_nx_graph(
            self.A,
            with_labels=with_labels,
            node_labels=node_labels,
            node_color=node_color,
            node_size=node_size,
            edge_width=edge_width,
            show_edge_values=show_edge_values,
            source_mask=source_mask,
            target_mask=target_mask,
            exclusion_mask=exclusion_mask,
            colorbar=colorbar,
            colorbar_text=colorbar_text,
            colorbar_scale=colorbar_scale,
            **kwargs
            )

        return ax

# +
def draw_nx_graph(
        A,
        with_labels=True, node_labels=None,
        node_color=None,
        node_size=None,
        edge_width=None,
        show_edge_values=True,
        source_mask=None,
        target_mask=None,
        exclusion_mask=None,
        show=True,
        figsize=(20,20),
        filename=None,
        colorbar=False,
        colorbar_text='Cost',
        cmap=plt.cm.jet,
        colorbar_scale=None,
        **kwargs):
    # nx defaults
    if node_color is None:
        node_color = "#613613"#"#1f78b4"
    if node_size is None:
        node_size = 300
    else:
        node_size = np.interp(node_size, (node_size.min(), node_size.max()), (100, 600))
    if edge_width is None:
        edge_width = 1.0
    else:
        edge_width = np.interp(edge_width, (edge_width.min(), edge_width.max()), (0.5, 3))

    number_of_nodes = A.shape[0]

    # networkx
    fig, ax = plt.subplots(figsize=figsize, frameon=False)

    G = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph)
    pos = nx.nx_pydot.pydot_layout(G, prog='twopi', root=None)

    vmin, vmax = kwargs.pop("vmin", None), kwargs.pop("vmax", None)
    if vmin is None:
        vmin = None if isinstance(node_color, str) else min(node_color)
    if vmax is None:
        vmax = None if isinstance(node_color, str) else max(node_color)

    node_types = combine_masks(number_of_nodes, target_mask=target_mask, source_mask=source_mask, exclusion_mask=exclusion_mask)
    if node_types is not None:
        #edge_origins, edge_targets = zip(*G.edges)
        #print(edge_width)
        for node_type in ['none','source','both','target']:
            ix= node_types == node_type
            nodelist = np.where(node_types == node_type)[0]
            #print('nodelist',nodelist)
            #edge_index = find_edges_incident_on_nodelist(G, nodelist)
            #print(edge_index)
            if node_type == 'source':
                nx.draw_networkx_nodes(G, pos, nodelist=nodelist, label=node_type,node_size=node_size[ix]#,node_shape='*'
                                       ,node_color=node_color[ix],cmap=cmap,vmin=vmin,vmax=vmax,edgecolors='r',linewidths=1, **kwargs)
                #edgelist = G.edges(nbunch=nodelist)
                #nx.draw_networkx_edges(G,pos,edgelist=edgelist,width=edge_width[edge_index], arrowsize=1)
            elif node_type == 'target':
                nx.draw_networkx_nodes(G, pos, nodelist=nodelist, label=node_type,node_size=node_size[ix]*1.5,node_shape='*'
                                       ,node_color=node_color[ix],cmap=cmap,vmin=vmin,vmax=vmax,edgecolors='b',linewidths=1, **kwargs)
                #edgelist = G.edges(nbunch=nodelist)
                #nx.draw_networkx_edges(G,pos,edgelist=edgelist,width=edge_width[edge_index], arrowsize=1)
            elif node_type == 'both':
                nx.draw_networkx_nodes(G, pos, nodelist=nodelist, label=node_type,node_size=node_size[ix]*1.5,node_shape='*'
                                       ,node_color=node_color[ix],cmap=cmap,vmin=vmin,vmax=vmax,edgecolors='g',linewidths=1, **kwargs)
                #edgelist = G.edges(nbunch=nodelist)
                #nx.draw_networkx_edges(G,pos,edgelist=edgelist,width=edge_width[edge_index])
            else:
                nx.draw_networkx_nodes(G, pos, nodelist=nodelist,node_size=node_size[ix],
                                       node_color='#C0C0C0',alpha=1,cmap=cmap,vmin=vmin,vmax=vmax, **kwargs)
                #edgelist = G.edges(nbunch=nodelist)
                #nx.draw_networkx_edges(G,pos,edgelist=edgelist,width=edge_width[edge_index], arrowsize=1)#, alpha=0.2)
        nx.draw_networkx_edges(G,pos,width=edge_width)#, arrowsize=1)
        #legend = ax.legend(prop={'size': 25})


    if (node_types is None) and (exclusion_mask is not None):
        nodelist = np.where(exclusion_mask==True)[0]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist,node_size=node_size[exclusion_mask]
                                       ,node_color='#C0C0C0',cmap=cmap,vmin=vmin,vmax=vmax,linewidths=1, **kwargs)
        nodelist = np.where(exclusion_mask==False)[0]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist,node_size=node_size[exclusion_mask==False]
                                       ,node_color=node_color[exclusion_mask==False],cmap=cmap,vmin=vmin,vmax=vmax,linewidths=1, **kwargs)

        nx.draw_networkx_edges(G,pos,width=edge_width)

    if (node_types is None) and (exclusion_mask is None):
        nx.draw_networkx(
                G,
                pos=pos, arrows=True,
                with_labels=with_labels,
                labels=node_labels,
                node_color=node_color,
                cmap=cmap,
                node_size=node_size,
                width=edge_width,
                vmin=vmin,
                vmax=vmax,
                **kwargs
                )

    number_of_edges = A.count_nonzero()
    if show_edge_values and number_of_edges < 50:
        edge_labels = nx.get_edge_attributes(G, "weight")
        edge_labels = {k:"{:.2f}".format(v) for k,v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)


    #if exclusion_mask is not None:
    print(vmin, vmax)
    if colorbar_scale is None:
        colorbar_scale = plt.Normalize(vmin=vmin, vmax=vmax)
    elif colorbar_scale == 'log':
        colorbar_scale = LogNorm(vmin=vmin, vmax=vmax)
    else:
        raise ValueError("scale not implemented")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=colorbar_scale)
    sm._A = []
    if colorbar:
        cbar = plt.colorbar(sm, orientation="vertical", pad=-0.05, shrink=0.5)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(colorbar_text, rotation=270, fontsize=16)
    fig.patch.set_visible(False)
    ax.axis('off')

#     cut_min = 0.9
#     cut_max = 1.1
#     xmin= cut_min*min(xx for xx,yy in pos.values())
#     ymin= cut_min*min(yy for xx,yy in pos.values())
#     xmax= cut_max*max(xx for xx,yy in pos.values())
#     ymax= cut_max*max(yy for xx,yy in pos.values())
#     plt.xlim(xmin,xmax)
#     plt.ylim(ymin,ymax)
#     plt.margins(0,0)

    if filename is not None:
        plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0)
    if show:
        plt.show()

    return ax


# -

def find_edges_incident_on_nodelist(G, nodelist):
    edge_index = []
    for node in nodelist:
        for i,edge in enumerate(G.edges):
            if node in edge:
                edge_index.append(i)
    return np.sort(np.unique(edge_index))


def make_mask_from_node_list(g, node_list, **kwargs):
    l = set(node_list)
    return torch.tensor([(x in l) for x in g.node_list.detach().cpu().numpy()], **kwargs)

def combine_masks(
        number_of_nodes,
        source_mask=None,
        target_mask=None,
        exclusion_mask=None,
        target=r'target',
        source=r'source',
        both=r'both',
        no=r'none'
        ):
    if source_mask is None and target_mask is None:
        return None
    node_type = np.full((number_of_nodes,), no, dtype=object)
    if source_mask is not None:
        node_type[source_mask] = source
    if target_mask is not None:
        node_type[target_mask] = target
    if (source_mask is not None) and (target_mask is not None):
        node_type[source_mask & target_mask] = both
    if exclusion_mask is not None:
        node_type[exclusion_mask] = no
    return node_type


def make_color_arr(
        number_of_nodes,
        source_mask=None,
        target_mask=None,
        target_color=r'#ff7f0e',
        source_color=r'#17becf',
        both_color=r'#808080',
        no_color=r'none'
        ):
    if source_mask is None and target_mask is None:
        return None, None
    edge_colors = np.full((number_of_nodes,), no_color)
    cdict={}
    if source_mask is not None:
        edge_colors[source_mask] = source_color
        cdict[source_color[:4]] = 'source'
    if target_mask is not None:
        edge_colors[target_mask] = target_color
        cdict[target_color[:4]] = 'target'
    if (source_mask is not None) and (target_mask is not None):
        edge_colors[source_mask & target_mask] = both_color
        cdict[both_color[:4]] = 'source & target'
    return edge_colors, cdict

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
