import logging

import numpy as np
import torch

log = logging.getLogger(__name__)


def directed_hausdorff_distance(pred: np.ndarray, reference: np.ndarray):
    """One-sided hausdorff distance between sets."""
    pred = np.asanyarray(pred)
    reference = np.asanyarray(reference)
    assert pred.ndim == 1
    assert reference.ndim == 1

    distances = np.zeros((pred.shape[0], reference.shape[0]), dtype=np.float64)
    for pred_idx, pred_pt in enumerate(pred):
        for reference_idx, reference_pt in enumerate(reference):
            distances[pred_idx, reference_idx] = np.abs(pred_pt - reference_pt)
    hausdorff_dist = np.max(np.min(distances, axis=1))
    return hausdorff_dist


def correlation_score(cov_x, cov_y, cov_xy):
    """ Computes the correlation score between two random multi-dimensional variables x and y.

    Maximizing this score is equivalent to maximizing the correlation between x and y

    The correlation score is defined as:
        P := ( ||cov_x^1/2 @ cov_xy @ cov_y^1/2 ||_HS )^2
    Args:
        cov_x:  (batch, features, features) or (features, features)
        cov_y:  (batch, features, features) or (features, features)
        cov_xy: (batch, features, features) or (features, features)

    Returns:
        Correlation Score value: (batch, 1) or (1,) depending on the input shape.
    """
    M_X = torch.linalg.lstsq(cov_x, cov_xy).solution
    M_Y = torch.linalg.lstsq(cov_y, cov_xy.transpose(-1, -2)).solution
    M_X_times_M_Y = torch.einsum('...ij,...jk->...ik', M_X, M_Y)
    # Compute the trace of the (potentially batch) of matrices.
    S = M_X_times_M_Y.diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)

    return S


def spectral_score(cov_x, cov_y, cov_xy):
    """ Computes the spectral score of the covariance matrices. This is a looser bound on the correlation score, but a
    more numerically stable metric, since it avoids computing the inverse of the covariance matrices.

    The spectral score is defined as:
        S := (||cov_xy||_HS)^2 / (||cov_x||_2 / ||cov_y||_2 )
    Args:
        cov_x: (..., features, features) or (features, features)
        cov_y: (..., features, features) or (features, features)
        cov_xy: (..., features, features) or (features, features)

    Returns:
        Score value: (..., 1) or (1,) depending on the input shape.
    """
    score = torch.linalg.matrix_norm(cov_xy, ord='fro', dim=(-2, -1)) ** 2  # == ||cov_xy|| 2, HS
    score = score / torch.linalg.matrix_norm(cov_x, ord=2, dim=(-2, -1))  # ||cov_xy|| 2, HS / ||cov_x||
    score = score / torch.linalg.matrix_norm(cov_y, ord=2, dim=(-2, -1))  # ||cov_xy|| 2, HS / (||cov_x|| * ||cov_y||)
    return score


def vectorized_spectral_scores(covYXdt: torch.Tensor,
                               covX: torch.Tensor,
                               covY: torch.Tensor,
                               run_checks: bool = False):
    """ Compute the spectral and correlation scores using the cross-covariance operators between distinct time steps.

    Args:
        covYXdt: (T, |Y|, |X|) Tensor containing all the Cross-Covariance
         empirical operators two multivariate random variables X and Y. Such that CovYX(dt-1) = Cov(y_i+dt, x_i) for
         all i in [0, time_horizon - dt], dt in [1, T].
        covX: (|X|, |X|) CovX = Cov(x_i, x_i) ∀ i in Integers
        covY: (|Y|, |Y|) CovY = Cov(y_i, y_i) ∀ i in Integers
        run_checks: (bool) Whether to print debug information on the scores computed. Defaults to False.
    Returns:
        spectral_scores: (T,) Tensor containing the spectral score between time steps separated
         apart by a shift of `dt` [steps/time]. That is:
            spectral_score[dt - 1] = avg(||Cov(x_i, x'_i+dt)||_HS^2/(||Cov(x_i, x_i)||_2*||Cov(x'_i+dt, x'_i+dt)||_2))
             | ∀ i in [0, time_horizon - dt], dt in [1, min(time_horizon - i, window_size)]
    """
    dimX, dimY = covYXdt.shape[-1], covYXdt.shape[-2]
    assert covYXdt.ndim == 3 and dimX == covYXdt.shape[-1] and dimY == covYXdt.shape[-2], \
        f"CovYXdt{covYXdt.shape}. Expected shape (T, |Y|, |X|)"
    assert covX.shape[-1] == covX.shape[-2], f"CovX:{covX.shape}. Expected shape (|X|, |X|)"
    assert covY.shape[-1] == covY.shape[-2], f"CovY:{covY.shape}. Expected shape (|Y|, |Y|)"

    norm_covX = torch.linalg.matrix_norm(covX, ord=2)  # norm_Cov_t = ||Cov(x, x)||_2
    norm_covY = torch.linalg.matrix_norm(covY, ord=2)  # norm_Cov_t = ||Cov(x', x')||_2
    norms_CCov = torch.linalg.matrix_norm(covYXdt, ord='fro', dim=(-2, -1)) # ||Cov(y_i+dt, x_i)||_HS for all i

    scores = norms_CCov ** 2 / (norm_covX * norm_covY)

    if run_checks:  # Check vectorized operations are equivalent to sequential operations
        for idx in range(len(scores)):
            dt = idx + 1
            exp = scores[idx]
            real = spectral_score(cov_x=covX, cov_y=covY, cov_xy=covYXdt[dt - 1])
            assert torch.allclose(exp, real, atol=1e-5), f"Spectral scores do not match {exp}!={real}"

    return scores


def vectorized_correlation_scores(covYXdt: torch.Tensor,
                                  covX: torch.Tensor,
                                  covY: torch.Tensor,
                                  run_checks: bool = False):
    """ Compute the correlation scores using the cross-covariance operators between distinct time steps.

    Args:
        covYXdt: (T, |Y|, |X|) Tensor containing all the Cross-Covariance
         empirical operators two multivariate random variables X and Y. Such that CovYX(dt-1) = Cov(y_i+dt, x_i) for
         all i in [0, time_horizon - dt], dt in [1, T].
        covX: (|X|, |X|) CovX = Cov(x_i, x_i) ∀ i in Integers
        covY: (|Y|, |Y|) CovY = Cov(y_i, y_i) ∀ i in Integers
        run_checks: (bool) Whether to print debug information on the scores computed. Defaults to False.
    Returns:
        corr_scores: (T,) Tensor containing the correlation scores between time steps separated
         apart by a shift of `dt` [steps/time]. That is:
            corr_score[dt - 1] = avg(||Cov(x_i, x_i)^-1 Cov(x_i, x'_i+dt) Cov(x'_i+dt, x'_i+dt)^-1||_HS^2)
             | ∀ i in [0, time_horizon - dt], dt in [1, min(time_horizon - i, window_size)]
    """
    dimX, dimY = covYXdt.shape[-1], covYXdt.shape[-2]
    assert covYXdt.ndim == 3 and dimX == covYXdt.shape[-1] and dimY == covYXdt.shape[-2], \
        f"CovYXdt{covYXdt.shape}. Expected shape (T, |Y|, |X|)"
    assert covX.shape[-1] == covX.shape[-2], f"CovX:{covX.shape}. Expected shape (|X|, |X|)"
    assert covY.shape[-1] == covY.shape[-2], f"CovY:{covY.shape}. Expected shape (|Y|, |Y|)"

    time_horizon = covYXdt.shape[0]
    # Unstable but fast in backward pass  !!!!
    # cov_xy_flat = covXY[idx_i, idx_j]
    # covX_inv_CovXY = torch.einsum("ab,tbc->tac", torch.linalg.pinv(covX, hermitian=True),
    #                               cov_xy_flat)  # covX.pinv() @ covXY
    # covX_inv_CovXY_covY_inv = torch.einsum("tac,cd->tad", covX_inv_CovXY,
    #                                        torch.linalg.pinv(covY, hermitian=True))  # covX.pinv() @ covXY @ covY
    # scores = torch.linalg.norm(covX_inv_CovXY_covY_inv, ord='fro', dim=(-2, -1)) ** 2

    # Stable but freaking slow in backward pass  !!!!
    scores = correlation_score(cov_x=covX.expand(time_horizon, -1, -1),
                               cov_y=covY.expand(time_horizon, -1, -1),
                               cov_xy=covYXdt)

    if run_checks:  # Check vectorized operations are equivalent to sequential operations
        for idx in range(len(scores)):
            dt = idx + 1
            exp = scores[idx]
            real = correlation_score(cov_x=covX, cov_y=covY, cov_xy=covYXdt[dt - 1])
            rel_error = torch.abs(exp - real) / torch.abs(real)
            assert rel_error <= 0.1, f"Correlation scores do not match {exp}!={real}"

    return scores

def chapman_kolmogorov_regularization(covYXdt: torch.Tensor, run_checks: bool = False):
    """ Compute the Chapman-Kolmogorov regularization using the cross-covariance operators between distinct time steps.

    This regularization aims at exploitation Markov Assumption of a linear dynamical system. Specifically it computes:
    ||Cov(y_t+dt, x_t) - (Cov(y_t+1, x_t) Cov(y_t+2, x_t+1) ... Cov(y_t+dt, x_t+dt-1))||_HS
    ∀ t in [0, pred_horizon-2], and dt in [2, pred_horizon].

    This regularization aims at enforcing the semi-group property of the Markov process.

    Args:
        covYXdt: (T, |Y|, |X|) Tensor containing all the Cross-Covariance
         empirical operators two multivariate random variables X and Y. Such that CovYX(dt-1) = Cov(y_i+dt, x_i) for
         all i in [0, time_horizon - dt], dt in [1, T].
        run_checks: (bool) Whether to print debug information on the CK scores computed. Defaults to False.
    Returns:

    """
    T, dimY, dimX = covYXdt.shape
    assert covYXdt.ndim == 3 and dimX == covYXdt.shape[-1] and dimY == covYXdt.shape[-2], \
        f"CovYXdt{covYXdt.shape}. Expected shape (T, |Y|, |X|)"


    # For efficiency (to avoid for loops), we compute the CK regularization only of powers of the original
    # cross-covariance matrices per dt. Understanding that:
    # Cov(y_i+2,x_i) = Cov(y_i+1, x_i)^2   < == > covYXdt[3] = covYXdt[0] @ covYXdt[0]
    # Cov(y_i+4,x_i) = Cov(y_i+2, x_i)^2   < == > covYXdt[5] = covYXdt[1] @ covYXdt[1]
    # Cov(y_i+6,x_i) = Cov(y_i+3, x_i)^2   < == > covYXdt[7] = covYXdt[2] @ covYXdt[2]


    max_dt = T if T % 2 == 0 else T - 1
    covYXdt_squared = torch.einsum("tij,tjk->tik", covYXdt[:(max_dt // 2) + 1], covYXdt[:(max_dt // 2) + 1])

    dt_idx = torch.arange(1, T//2 + 1, device=covYXdt.device)
    target_idx_dt = 2 * (dt_idx)

    # Cov(y_i+H*2, x_i) = Cov(y_i+H, x_i)^2 < == > covYXdt[2*H-1] = covYXdt[H-1] @ covYXdt[H-1]
    CK_err = covYXdt[target_idx_dt - 1] - covYXdt_squared[dt_idx - 1]

    CK_err_norm = torch.linalg.norm(CK_err, ord='fro', dim=(-2, -1))

    return CK_err_norm
