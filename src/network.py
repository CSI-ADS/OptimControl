import torch
import numpy as np
import networkx as nx
import scipy


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
    def total_value(self):
        if self.V is None:
            return None
        else:
            return torch.sum(self.V*self.total_shares_in_network)

    @property
    def nodes(self):
        return self.node_list if self.node_list is not None else torch.arange(self.N)

    @property
    def total_shares_in_network(self):
        return self.C.sum(axis=0)

    def to(self, device):
        self.C = self.C.to(device)
        if self.V is not None:
            self.V = self.V.to(device)
        return self

    def identify_uncontrollable(self):
        Ctot = self.total_shares_in_network
        contr = Ctot > 1e-8
        return contr

    def remove_uncontrollable(self):
        contr = self.identify_uncontrollable()
        tot_contr = sum(contr)
        print("Keeping {} out of {}".format(tot_contr, self.number_of_nodes))
        if tot_contr == self.number_of_nodes:
            return self
        A = self.A[contr,:][:,contr]
        V = None if self.V is None else self.V[contr]
        node_list = None if self.node_list is None else self.node_list[contr]
        dtype = self.C.dtype
        return Network(A, value=V, node_list=node_list, dtype=dtype)


def normalize_ownership(g):
    C = g.ownership
    N = C.shape[0]
    Csum = C.sum(axis=0)
    Csum_zero = (Csum == 0).to(C.dtype)
    Csum = Csum + Csum_zero
    C = C / Csum.reshape((1, N))
    return C


def adjust_for_external_ownership(cl, g):
#     N = C.shape[0]
    C = g.ownership
    # C
    #print(C)
    # renormalize
    # assumption: we will take shares from other proportional to the amount of shares
    # e.g. 10% of shares, is 10% from all owners
    C = C * (1.0 - cl)
#     C = torch.cat((cl.reshape(1, N), C), axis=0)
#     Csum = C.sum(axis=0)
#     #numerical issues
#     Csum_zero = (Csum == 0).double()
#     Csum = Csum + Csum_zero
#     #print(Csum)
#     C = C / Csum.reshape((1, N))
#     C = C[1:,:]
    return C
