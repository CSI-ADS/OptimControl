import torch
import numpy as np
import networkx as nx

class Network:

    def __init__(self, A, value=None, dtype=torch.float):
        if isinstance(A, np.ndarray):
            A = torch.from_numpy(A)
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        assert isinstance(A, torch.Tensor), "use torch"
        assert value is None or isinstance(value, torch.Tensor), "use torch"
        A = A.to(dtype)
        if value:
            value = value.to(dtype)
        self.N = A.shape[0]
        self.set_control_matrix(A)
        self.V = value # value of each node
        #self.set_reachable(A)

    # def set_reachable(self, A, tol=1e-8):
    #     D = torch.abs(torch.matrix_power(A, self.N))
    #     D = D - torch.diag(torch.diag(D))
    #     self.D = D > tol

    def set_control_matrix(self, A):
        self.C = A
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
    def total_shares_in_network(self):
        return self.C.sum(axis=0)

    def to(self, device):
        self.C = self.C.to(device)
        if self.V:
            self.V = self.V.to(device)
        return self

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
