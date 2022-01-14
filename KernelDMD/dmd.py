from scipy.sparse.linalg import eigsh, aslinearoperator
import torch
import numpy 

def _get_scaled_SVD_right_eigvectors(X, num_modes, backend):
    if backend =='torch':
        Sigma, V = torch.linalg.eigh(X)
        Sigma_r, V_r = Sigma[-num_modes:], V[:,-num_modes:]
        return V_r@torch.diag(torch.sqrt(Sigma_r)**-1)
    elif backend =='keops':
        Sigma_r, V_r = eigsh(aslinearoperator(X), k = num_modes)
        return V_r@numpy.diag(numpy.sqrt(Sigma_r)**-1)
    else:
        raise ValueError("Supported backends are 'torch' or 'keops'")

def _DMD(trajectory, kernel, num_modes, backend):
    """DMD utility function

    Args:
        trajectory (array): [observations, features]
        kernel (kernel object)
        num_modes (int): number of modes to compute
    """
    Vhat_r = _get_scaled_SVD_right_eigvectors(kernel(trajectory[:-1], backend=backend), num_modes, backend)
    Ahat = Vhat_r.T @(kernel(trajectory[:-1],trajectory[1:], backend=backend)@Vhat_r)
    if backend =='torch':
        evals, evecs = torch.linalg.eig(Ahat)
        Vhat_r = Vhat_r.cfloat() #convert to Complex type
    elif backend =='keops':
        evals, evecs = numpy.linalg.eig(Ahat)
    else:
        raise ValueError("Supported backends are 'torch' or 'keops'")    
    return evals, Vhat_r@evecs

def DMD_large_scale(trajectory, kernel, num_modes):
    """DMD using Lanczos iterations. Useful for large problems

    Args:
        trajectory (array): [observations, features]
        kernel (kernel object)
        num_modes (int): number of modes to compute
    """
    return _DMD(trajectory, kernel, num_modes, 'keops')

def DMD(trajectory, kernel, num_modes):
    """DMD using truncated SVD

    Args:
        trajectory (array): [observations, features]
        kernel (kernel object)
        num_modes (int): number of modes to compute
    """
    return _DMD(trajectory, kernel, num_modes, 'torch')