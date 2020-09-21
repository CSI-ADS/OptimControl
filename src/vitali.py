import numpy as np
import torch

def get_reachable(cl):
    return torch.where(cl)[0] # exact sparseness would help

def get_Bsub(C, reach):
    return C[reach,:][:, reach]

def get_d(cl, reach):
    """ FIXME, here, cl has a clear definition"""
    return cl[reach].reshape(1, -1) # row vector

def get_Isub(reach, device=None):
    return torch.eye(len(reach), device=device)

def get_dtilde(cl, C, reach):
    Nreach = len(reach)

    if Nreach == 0:
        return None
    #print("C=",C)
    Bsub = get_Bsub(C, reach)
    #print("Bsub=", Bsub)

    d = get_d(cl, reach)

#     print("d = ", d)
#     print("Bsub = ", Bsub)
#     print("reach=", reach)

    Isub = get_Isub(reach, device=Bsub.device)
    to_invert = Isub - Bsub
#     print("TO",to_invert)

    inverted = torch.inverse(to_invert)
#     print("INV", inverted)
    # TODO: adjust by p value
    dtilde = d.matmul(inverted)
#     print("DTILDE", dtilde)
    return dtilde

def compute_control(cl, C, reach=None, control_cutoff=None):
    if reach is None:
        reach = get_reachable(cl)
    assert len(reach) > 0, "no reachendents"
    if control_cutoff:
        C = make_control_cutoff(C, control_cutoff)
    #print(C, reach)
    dtilde = get_dtilde(cl, C, reach)
    return dtilde
