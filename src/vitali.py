import numpy as np
import torch

def get_descendants(cl):
    return torch.where(cl)[0] # exact sparseness would help

def get_Bsub(C, desc):
    return C[desc,:][:, desc]

def get_d(cl, desc):
    """ FIXME, here, cl has a clear definition"""
    return cl[desc].reshape(1, -1) # row vector

def get_Isub(desc, device=None):
    return torch.eye(len(desc), device=device)

def get_dtilde(cl, C, desc):
    Ndesc = len(desc)

    if Ndesc == 0:
        return None
    #print("C=",C)
    Bsub = get_Bsub(C, desc)
    #print("Bsub=", Bsub)

    d = get_d(cl, desc)

#     print("d = ", d)
#     print("Bsub = ", Bsub)
#     print("desc=", desc)

    Isub = get_Isub(desc, device=Bsub.device)
    to_invert = Isub - Bsub
#     print("TO",to_invert)

    inverted = torch.inverse(to_invert)
#     print("INV", inverted)
    # TODO: adjust by p value
    dtilde = d.matmul(inverted)
#     print("DTILDE", dtilde)
    return dtilde

def compute_control(cl, C, desc=None, control_cutoff=None):
    if desc is None:
        desc = get_descendants(cl)
    assert len(desc) > 0, "no descendents"
    if control_cutoff:
        C = make_control_cutoff(C, control_cutoff)
    #print(C, desc)
    dtilde = get_dtilde(cl, C, desc)
    return dtilde
