import logging
from typing import Optional

import numpy as np
import torch
from escnn.group import Representation

from kooplearn._src.linalg import batch_matrix_sqrt_inv
from kooplearn.nn.functional import vectorized_cross_cov

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


def vectorized_spectral_scores(covXY: torch.Tensor,
                               covX: torch.Tensor,
                               covY: torch.Tensor,
                               run_checks: bool = False):
    """ Compute the spectral and correlation scores using the cross-covariance operators between distinct time steps.

    Args:
        covXY: (time_horizon, time_horizon, state_dim, state_dim) Tensor containing all the Cross-Covariance
         empirical operators between the states in the main trajectory and states in the auxiliary trajectory.
         Each entry of the tensor is a (state_dim, state_dim) covariance estimate.
         Such that CCov(i,j) = Cov(x_i, x'_j) ∀ i, j in [0, time_horizon], j >= i.
        covX: (state_dim, state_dim) Tensor containing all the Covariance empirical operators between time
         steps on the main state space. Cov(i) = Cov(x_i, x_i) ∀ i in [0, time_horizon]
        covY: (state_dim, state_dim) Tensor containing all the Covariance empirical operators between
         time steps on the auxiliary state space. Cov_prime(i) = Cov(x'_i, x'_i) ∀ i in [0, time_horizon]
        window_size: (int) Maximum window length to compute the spectral score. Defaults to None, in which case the
         score is computed for all window_size > j >= i
        run_checks: (bool) Whether to print debug information on the scores computed. Defaults to False.
    Returns:
        spectral_scores: (time_horizon - 1) Tensor containing the average spectral score between time steps separated
         apart by a shift of `dt` [steps/time]. That is:
            spectral_score[dt - 1] = avg(||Cov(x_i, x'_i+dt)||_HS^2/(||Cov(x_i, x_i)||_2*||Cov(x'_i+dt, x'_i+dt)||_2))
             | ∀ i in [0, time_horizon - dt], dt in [1, min(time_horizon - i, window_size)]
    """
    assert (len(covXY.shape) == 4 and covXY.shape[0] == covXY.shape[1] and covXY.shape[2] == covXY.shape[3]), \
        f"CCov:{covXY.shape}. Expected Cov_t_dt of shape (T, T, state_dim, state_dim)"
    assert len(covX.shape) == 2 and covX.shape[-1] == covX.shape[-2], \
        f"Cov:{covX.shape}. Expected Cov of shape (state_dim, state_dim)"
    assert covX.shape == covY.shape, f"Expected Cov_prime of shape (state_dim, state_dim)"

    time_horizon = covXY.shape[0]

    # Compute the norm of the diagonal of the covariance matrices is a single parallel operation.
    norm_covX = torch.linalg.matrix_norm(covX, ord=2)  # norm_Cov_t = ||Cov(x, x)||_2
    norm_covY = torch.linalg.matrix_norm(covY, ord=2)  # norm_Cov_t = ||Cov(x', x')||_2

    # Since we are only interested in the terms ||Cov(x_i, x'_j)||_HS  s.t j > i, we compute the upper triangular part
    # the norms of the "upper triangular" part of the (time, time, ...) axis of the CCov matrix.
    # Get indices for the upper triangular part, including the diagonal
    # TODO: this introduces unnecessary copies, we should avoid them
    idx_i, idx_j = torch.triu_indices(time_horizon, time_horizon, device=covXY.device)
    # Instead of computing the norm for (time x time) matrices, instead compute the (time(time-1)/2) upper triangular
    norms_CCov = torch.linalg.matrix_norm(covXY[idx_i, idx_j], ord='fro', dim=(-2, -1))

    scores = norms_CCov ** 2 / (norm_covX * norm_covY)

    if run_checks:  # Check vectorized operations are equivalent to sequential operations
        for i, j, score_ij, norm in zip(idx_i, idx_j, scores, norms_CCov):
            exp = score_ij
            real = spectral_score(cov_x=covX, cov_y=covY, cov_xy=covXY[i, j])
            assert torch.allclose(exp, real, atol=1e-5), f"Spectral scores do not match {exp}!={real}"
            if i > 1: break

    return scores.mean()


def vectorized_correlation_scores(covXY: torch.Tensor,
                                  covX: torch.Tensor,
                                  covY: torch.Tensor,
                                  window_size: Optional[int] = None,
                                  run_checks: bool = False):
    """ Compute the correlation scores using the cross-covariance operators between distinct time steps.

    Args:
        covXY: (time_horizon, time_horizon, state_dim, state_dim) Tensor containing all the Cross-Covariance
         empirical operators between the states in the main trajectory and states in the auxiliary trajectory.
         Each entry of the tensor is a (state_dim, state_dim) covariance estimate.
         Such that CCov(i,j) = Cov(x_i, x'_j) ∀ i, j in [0, time_horizon], j >= i.
        covX: (state_dim, state_dim) Tensor containing all the Covariance empirical operators between time
         steps on the main state space. Cov(i) = Cov(x_i, x_i) ∀ i in [0, time_horizon]
        covY: (state_dim, state_dim) Tensor containing all the Covariance empirical operators between
         time steps on the auxiliary state space. Cov_prime(i) = Cov(x'_i, x'_i) ∀ i in [0, time_horizon]
        window_size: (int) Maximum window length to compute the spectral score. Defaults to None, in which case the
         score is computed for all window_size > j >= i
        run_checks: (bool) Whether to print debug information on the scores computed. Defaults to False.
    Returns:
        corr_scores: (time_horizon - 1) Tensor containing the correlation scores between time steps separated
         apart by a shift of `dt` [steps/time]. That is:
            corr_score[dt - 1] = avg(||Cov(x_i, x_i)^-1 Cov(x_i, x'_i+dt) Cov(x'_i+dt, x'_i+dt)^-1||_HS^2)
             | ∀ i in [0, time_horizon - dt], dt in [1, min(time_horizon - i, window_size)]
    """
    assert (len(covXY.shape) == 4 and covXY.shape[0] == covXY.shape[1] and covXY.shape[2] == covXY.shape[3]), \
        f"CCov:{covXY.shape}. Expected Cov_t_dt of shape (T, T, state_dim, state_dim)"
    assert len(covX.shape) == 2 and covX.shape[-1] == covX.shape[-2], \
        f"Cov:{covX.shape}. Expected Cov of shape (state_dim, state_dim)"
    assert covX.shape == covY.shape, f"Expected Cov_prime of shape (state_dim, state_dim)"

    time_horizon = covXY.shape[0]

    # Since we are only interested in the terms Cov(x_i, x'_j)  s.t j > i, we compute the upper triangular part
    # the norms of the "upper triangular" part of the (time, time, ...) axis of the CCov matrix.
    # Get indices for the upper triangular part, including the diagonal
    idx_i, idx_j = torch.triu_indices(time_horizon, time_horizon, device=covXY.device)

    # # Mildly stable and fast in backward pass !!!!
    # cov_inv_sqrt, cond_num_Cov = batch_matrix_sqrt_inv(torch.stack([covX, covY]), run_checks=run_checks)
    # cov_X_inv_sqrt = cov_inv_sqrt[0]
    # cov_Y_inv_sqrt = cov_inv_sqrt[1]
    #
    # covX_inv_CovXY = torch.einsum("ab,tbc->tac", cov_X_inv_sqrt, covXY[idx_i, idx_j])  # covX.pinv() @ covXY
    # covX_inv_CovXY_covY_inv = torch.einsum("tac,cd->tad", covX_inv_CovXY, cov_Y_inv_sqrt)  # covX.pinv() @ covXY @ covY
    # scores = torch.linalg.norm(covX_inv_CovXY_covY_inv, ord='fro', dim=(-2, -1)) ** 2

    # Unstable but fast in backward pass  !!!!
    cov_xy_flat = covXY[idx_i, idx_j]
    covX_inv_CovXY = torch.einsum("ab,tbc->tac", torch.linalg.pinv(covX, hermitian=True),  cov_xy_flat)  # covX.pinv() @ covXY
    covX_inv_CovXY_covY_inv = torch.einsum("tac,cd->tad", covX_inv_CovXY, torch.linalg.pinv(covY, hermitian=True))  # covX.pinv() @ covXY @ covY
    scores = torch.linalg.norm(covX_inv_CovXY_covY_inv, ord='fro', dim=(-2, -1)) ** 2

    # Stable but freaking slow in backward pass  !!!!
    # scores = correlation_score(cov_x=covX.expand(len(idx_i), -1, -1),
    #                            cov_y=covY.expand(len(idx_i), -1, -1),
    #                            cov_xy=covXY[idx_i, idx_j])

    if run_checks:  # Check vectorized operations are equivalent to sequential operations
        for i, j, score_ij in zip(idx_i, idx_j, scores):
            cov_xi_yj = covXY[i, j]
            exp = score_ij
            real = correlation_score(cov_x=covX, cov_y=covY, cov_xy=cov_xi_yj)
            # Least squares introduces some numerical instability, so we allow for a small tolerance
            rel_error = torch.abs(exp - real) / real
            assert rel_error <= 1e-1, \
                f"Relative error between vectorized and sequential computation of correlation is {rel_error}"

    return scores.mean()


def obs_state_space_metrics(obs_state_traj: torch.Tensor,
                            obs_state_traj_aux: Optional[torch.Tensor] = None,
                            representation: Optional[Representation] = None,
                            max_ck_window_length: int = 2):
    """ Compute the metrics of an observable space with expected linear dynamics.

    This function computes the metrics of an observable space with expected linear dynamics. Specifically,
    Args:
        obs_state_traj (batch, time_horizon, obs_state_dim): trajectory of states
        obs_state_traj_aux (batch, time_horizon, obs_state_dim): Auxiliary trajectory of states
        representation: Symmetry representation on the observable space. If provided, the empirical covariance and
            cross-covariance operators will be improved using the group average trick
        max_ck_window_length: Maximum window length to compute the Chapman-Kolmogorov regularization term.
        ck_w: Weight of the Chapman-Kolmogorov regularization term.
    Returns:
        Dictionary containing:
        - spectral_score: (time_horizon - 1) Tensor containing the average spectral score between time steps
        separated
         apart by a shift of `dt` [steps/time]. That is:
            spectral_score[dt - 1] = avg(||Cov(x_i, x'_i+dt)||_HS^2/(||Cov(x_i, x_i)||_2*||Cov(x'_i+dt, x'_i+dt)||_2))
             | ∀ i in [0, time_horizon - dt], dt in [1, min(time_horizon - i, window_size)]
        - corr_score: (time_horizon - 1) Tensor containing the correlation scores between time steps separated
         apart by a shift of `dt` [steps/time]. That is:
            corr_score[dt - 1] = avg(||Cov(x_i, x_i)^-1 Cov(x_i, x'_i+dt) Cov(x'_i+dt, x'_i+dt)^-1||_HS^2)
             | ∀ i in [0, time_horizon - dt], dt in [1, min(time_horizon - i, window_size)]
        - orth_reg: (time_horizon) Tensor containing the orthonormality regularization term for each time step.
         That is: orth_reg[t] = || Cov(t,t) - I ||_2
        - ck_reg: (time_horizon - 1,) Average CK error per `dt` time steps. That is:
            ck_error[dt - 2] = avg(|| Cov(t, t+dt) - Cov(t, t+1) Cov(t+1, t+2) ... Cov(t+dt-1, t+dt) ||) |
            ∀ t in [0, time_horizon - 2], dt in [2, min(time_horizon - 2, ck_window_length)]
        - cov_cond_num: (float) Average condition number of the Covariance matrices.
    """
    debug = log.level == logging.DEBUG  # TODO: remove default debug
    # Compute the empirical covariance and cross-covariance operators, ensuring that operators are equivariant.
    # CCov[i, j] := Cov(x_i, x'_j)     | i,j in [time_horizon], j > i  # Upper triangular tensor
    # Cov[t] := Cov(x_t, x_t)          | t in [time_horizon]
    # Cov_prime[t] := Cov(x'_t, x'_t)  | t in [time_horizon]
    CCov, Cov, Cov_prime = vectorized_cov_cross_cov(X_contexts=obs_state_traj,
                                                    Y_contexts=obs_state_traj_aux,
                                                    cov_window_size=max_ck_window_length,
                                                    representation=representation,
                                                    run_checks=debug)

    # Compute Cov(x_t)^-1/2 and Cov(x'_t)^-1/2 in a single parallel operation.
    Cov_inv_sqrt, cond_num_Cov = batch_matrix_sqrt_inv(Cov, run_checks=debug)
    Cov_prime_inv_sqrt, conv_num_Cov_prime = batch_matrix_sqrt_inv(Cov_prime, run_checks=debug)

    # Orthonormality regularization terms for ALL time steps in horizon
    # reg_orthonormal[t] = || Cov(x_i, x_i) - I || | t in [0, pred_horizon]
    orthonormality_Cov = regularization_orthonormality(Cov)
    orthonormality_Cov_prime = regularization_orthonormality(Cov_prime)
    reg_orthonormal = (orthonormality_Cov + orthonormality_Cov_prime) / 2.0

    cond_num_Cov = torch.cat([cond_num_Cov, conv_num_Cov_prime], dim=0).mean()

    # Compute the Correlation, Spectral and for ALL time steps in horizon.
    # spectral_scores[dt - 1] := ||Cov(t, t+dt)||^2_HS / (||Cov(t)|| ||Cov(t+d)||) | dt in [1, time_horizon)
    # corr_scores[dt - 1] := ||Cov(t)^-1 Cov(t, t+dt) Cov(t+d)^-1||^2_HS | dt in [1, time_horizon)
    spectral_scores, corr_scores = vectorized_spectral_correlation_scores(CCov=CCov,
                                                                          Cov=Cov, Cov_prime=Cov_prime,
                                                                          Cov_sqrt_inv=Cov_inv_sqrt,
                                                                          Cov_prime_sqrt_inv=Cov_prime_inv_sqrt,
                                                                          run_checks=debug)
    if debug:
        assert (corr_scores > spectral_scores).all(), "Correlation scores should be upper bound of spectral scores"

    # Compute the Chapman-Kolmogorov regularization scores for all possible step transitions. In return, we get:
    # ck_regularization[i,j] = || Cov(i, j) - ( Cov(i, i+1), ... Cov(j-1, j) ) ||_2  | j >= i + 2
    ck_regularization = chapman_kolmogorov_regularization(CCov=CCov,  # Cov=Cov, Cov_prime=Cov_prime,
                                                          ck_window_length=max_ck_window_length,
                                                          debug=debug)

    return dict(orth_reg=reg_orthonormal,
                ck_reg=ck_regularization,
                spectral_score=spectral_scores,
                corr_score=corr_scores,
                cov_cond_num=cond_num_Cov,
                # projection_score_t=torch.nanmean(projection_score_t, dim=0, keepdim=True),  # (batch, time)
                # spectral_score_t=torch.nanmean(spectral_score_t, dim=0, keepdim=True)       # (batch, time)
                )
