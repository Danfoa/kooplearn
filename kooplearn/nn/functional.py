from collections import namedtuple

from escnn.group import Representation
from kooplearn._src.check_deps import check_torch_deps
from kooplearn._src.metrics import vectorized_correlation_scores, vectorized_spectral_scores
from kooplearn.data import TensorContextDataset

check_torch_deps()
from typing import Optional  # noqa: E402

import torch  # noqa: E402


def sqrtmh(A: torch.Tensor):
    # Credits to
    """Compute the square root of a Symmetric or Hermitian positive definite matrix or batch of matrices. Credits to
    `https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228
    <https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228>`_."""
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH


def covariance(X: torch.Tensor, Y: Optional[torch.Tensor] = None, center: bool = True):
    """Covariance matrix

    Args:
        X (torch.Tensor): Input covariates of shape ``(samples, |X|)``.
        Y (Optional[torch.Tensor], optional): Output covariates of shape ``(samples, |Y|)``. Defaults to None.
        center (bool, optional): Whether to compute centered covariances. Defaults to True.

    Returns:
        torch.Tensor: variance matrix of shape Var(X)=Cov(X,X) in R^(|X|, |X|) if  Y is None. Else, cross-covariance
        matrix Cov(X,Y) in R^(|X|, |Y|).
    """
    assert X.ndim == 2
    cov_norm = torch.rsqrt(torch.tensor(X.shape[0]))
    if Y is None:
        _X = cov_norm * X
        if center:
            _X = _X - _X.mean(dim=0, keepdim=True)
        return torch.mm(_X.T, _X)
    else:
        assert Y.ndim == 2
        _X = cov_norm * X
        _Y = cov_norm * Y
        if center:
            _X = _X - _X.mean(dim=0, keepdim=True)
            _Y = _Y - _Y.mean(dim=0, keepdim=True)
        return torch.mm(_Y.T, _X)


def vamp_score(X, Y, schatten_norm: int = 2, center_covariances: bool = True):
    """Variational Approach for learning Markov Processes (VAMP) score by :footcite:t:`Wu2019`.

    Args:
        X (torch.Tensor): Covariates for the initial time steps.
        Y (torch.Tensor): Covariates for the evolved time steps.
        schatten_norm (int, optional): Computes the VAMP-p score with ``p = schatten_norm``. Defaults to 2.
        center_covariances (bool, optional): Use centered covariances to compute the VAMP score. Defaults to True.

    Raises:
        NotImplementedError: If ``schatten_norm`` is not 1 or 2.

    """
    cov_X, cov_Y, cov_YX = (
        covariance(X, center=center_covariances),
        covariance(Y, center=center_covariances),
        covariance(X, Y, center=center_covariances),
        )
    if schatten_norm == 2:
        # Using least squares in place of pinv for numerical stability
        M_X = torch.linalg.lstsq(cov_X, cov_YX).solution
        M_Y = torch.linalg.lstsq(cov_Y, cov_YX.T).solution
        return torch.trace(M_X @ M_Y)
    elif schatten_norm == 1:
        sqrt_cov_X = sqrtmh(cov_X)
        sqrt_cov_Y = sqrtmh(cov_Y)
        M = torch.linalg.multi_dot(
            [
                torch.linalg.pinv(sqrt_cov_X, hermitian=True),
                cov_YX,
                torch.linalg.pinv(sqrt_cov_Y, hermitian=True),
                ]
            )
        return torch.linalg.matrix_norm(M, "nuc")
    else:
        raise NotImplementedError(f"Schatten norm {schatten_norm} not implemented")


def deepprojection_score(
        X,
        Y,
        relaxed: bool = True,
        metric_deformation: float = 1.0,
        center_covariances: bool = True,
        ):
    """Deep Projection score by :footcite:t:`Kostic2023DPNets`.

    TODO: Add equations and explanation to docstring. Mention unrelaxed score is simply correlation.

    Args:
        X (torch.Tensor): Covariates for the initial time steps.
        Y (torch.Tensor): Covariates for the evolved time steps.
        relaxed (bool, optional): Whether to use the relaxed (more numerically stable) or the full deep-projection
        loss. Defaults to True.
        metric_deformation (float, optional): Strength of the metric metric deformation loss: Defaults to 1.0.
        center_covariances (bool, optional): Use centered covariances to compute the VAMP score. Defaults to True.

    """
    cov_X, cov_Y, cov_YX = (
        covariance(X, center=center_covariances),
        covariance(Y, center=center_covariances),
        covariance(X, Y, center=center_covariances),
        )
    R_X = log_fro_metric_deformation_loss(cov_X)
    R_Y = log_fro_metric_deformation_loss(cov_Y)
    if relaxed:
        S = (torch.linalg.matrix_norm(cov_YX, ord="fro") ** 2 /
             (torch.linalg.matrix_norm(cov_X, ord=2) * torch.linalg.matrix_norm(cov_Y, ord=2)))
    else:
        M_X = torch.linalg.lstsq(cov_X, cov_YX).solution
        M_Y = torch.linalg.lstsq(cov_Y, cov_YX.T).solution
        S = torch.trace(M_X @ M_Y)
    return S - (0.5 * metric_deformation * (R_X + R_Y))


def log_fro_metric_deformation_loss(cov: torch.tensor):
    """Logarithmic + Frobenious metric deformation loss as used in :footcite:t:`Kostic2023DPNets`, defined as
    :math:`{{\\rm Tr}}(C^{2} - C -\ln(C))` .

    Args:
        cov (torch.tensor): A symmetric positive-definite matrix of shape (s, s) or a batch of matrices of shape (b,
        s, s).
    Returns:
        torch.tensor: The metric deformation loss of shape (1,) or (b,) depending on the input shape.
    """
    eps = torch.finfo(cov.dtype).eps * cov.shape[-1]
    vals_x = torch.linalg.eigvalsh(cov)
    vals_x = torch.where(vals_x > eps, vals_x, eps * torch.ones_like(vals_x))
    loss = torch.mean(vals_x * (vals_x - 1.0) - torch.log(vals_x), dim=-1)
    return loss


def extract_evolved_states(state_traj: torch.Tensor,
                           state_traj_prime: Optional[torch.Tensor] = None,
                           dt: int = 1) -> (torch.Tensor, torch.Tensor):
    """
    This function takes a trajectory of states x_t in a time window t in [0, time_horizon] (steps) and returns two
    tensors X and X' of shape (batch * (time_horizon - dt), state_dim) containing the states at time x_k and x_k+dt,
    respectively.

    Args:
        state_traj: of shape (batch, time_horizon, state_dim) trajectory of states.
        state_traj_prime: of shape (batch, time_horizon, state_dim) trajectory of states.
        dt: Number of steps to draw the evolved states. Defaults to 1.
    Returns:
        X: a view of `state_traj` of shape (batch * (time_horizon - dt), state_dim) containing the states at time x_k.
        X_prime: a view of `state_traj` (or `state_traj_prime` if provided) of shape (batch * (time_horizon - dt),
         state_dim) containing the states at time x'_k+dt.
    """
    assert len(state_traj.shape) == 3, f"state_traj: {state_traj.shape}. Expected (batch, time_horizon, state_dim)"
    assert state_traj_prime is None or state_traj_prime.shape == state_traj.shape
    num_samples, time_horizon, state_dim = state_traj.shape
    pred_horizon = time_horizon - 1
    assert 0 < dt <= pred_horizon, f"dt: {dt}. Expected 0 < dt <= {pred_horizon}"

    X = state_traj[:, :-dt, :].view(-1, state_dim)
    if state_traj_prime is None:
        X_prime = state_traj[:, dt:, :].view(-1, state_dim)
    else:
        X_prime = state_traj_prime[:, dt:, :].view(-1, state_dim)

    return X, X_prime


def vectorized_cov_cross_cov(X_contexts: torch.Tensor,
                             Y_contexts: Optional[torch.Tensor] = None,
                             run_checks: bool = False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """ Compute empirical estimations of the cross-covariance operators.

    This function takes data from two random variables X and Y in the form of a trajectory of states in context window
    shape (n_samples, time_horizon, state_dim) there x_t, y_t are the states at time t in [0, time_horizon].

    It computes the cross covariance operators CovXYdt := Cov(x_i, y_i+dt) for all dt in [1, time_horizon]. Since the
    random variables are assumed to come from a time-homogenous process Cov(x_i, y_i+dt) := Cov(x_i+k, y_i+k+dt) for any
    integer k. That is, the covariance operators are dependent only on the time difference dt between states.

    Args:
        X_contexts: (batch, time_horizon, |X|) trajectory of states of random variable X
        Y_contexts: (batch, time_horizon, |Y|) trajectory of states of random variable Y.
        run_checks: (bool) Run sanity checks to ensure the empirical estimates are correct. Default to False.
    Returns:
        CovYXdt: a tensor of shape (min(time_horizon-1, max_dts), |Y|, |X|) containing the Cross-Covariances
         per time difference dt. Such that CCov[dt-1] := Cov(y_i+dt, x_i) ∀ i dt in [0, time_horizon].
    """
    assert len(X_contexts.shape) == 3, f"state_traj: {X_contexts.shape}. Expected (batch, time_horizon, |X|)"

    num_samples, time_horizon, _ = X_contexts.shape

    x_t = X_contexts[:, [0], :]  # (batch_size, 1, |X|)
    y_t_dt = Y_contexts[:, 1:time_horizon, :]  # (batch_size, max_dts, |Y|)

    # Compute cross-covariance in single batched/parallel operation
    CovYXdt = torch.einsum('bty,box->tyx', y_t_dt, x_t) / num_samples

    if run_checks:  # Sanity checks. These should be true by construction
        from kooplearn.nn.functional import covariance
        for dt in range(1, time_horizon):
            x_0 = X_contexts[:, 0, :]
            y_dt = Y_contexts[:, dt, :]
            CovYXdt_true = covariance(X=x_0, Y=y_dt, center=False)
            assert torch.allclose(CovYXdt[dt - 1], CovYXdt_true, rtol=1e-5, atol=1e-5), \
                f"Max error {torch.max(torch.abs(CovYXdt[dt - 1] - CovYXdt_true))}"

    return CovYXdt


def latent_space_metrics(
        Z_contexts: TensorContextDataset,
        Z_prime_contexts: TensorContextDataset,
        center_covariances: bool = False,
        G_rep_Z: Optional[torch.Tensor] = None,
        grad_correlation_score=True,
        grad_relaxed_score=True,
        run_checks=False):
    batch, time_horizon, latent_state_dim = Z_contexts.data.shape
    n_samples = batch * time_horizon

    if center_covariances:
        Z = Z_contexts.data - torch.mean(Z_contexts.data, dim=(0, 1))
        Z_prime = Z_prime_contexts.data - torch.mean(Z_prime_contexts.data, dim=(0, 1))
    else:
        Z, Z_prime = Z_contexts.data, Z_prime_contexts.data
    # Since we assume time homogenous dynamics, the CovX and CovY are "time-independent" so we compute them
    # using the `n=batch * time_horizon` samples, to get better empirical estimates.
    covZ = torch.einsum("btx,bty->xy", Z, Z) / n_samples
    covZp = torch.einsum("btx,bty->xy", Z_prime, Z_prime) / n_samples

    covZpZdt = vectorized_cov_cross_cov(
        X_contexts=Z[:, :-1, :],
        Y_contexts=Z_prime[:, 1:, :],
        run_checks=run_checks)

    if G_rep_Z is not None:  # Apply equivariant constraints to the empirical estimates by group averaging.
        assert G_rep_Z.ndim == 3, f"G_rep_Z: {G_rep_Z.shape}. Expected (|H|, |Z|, |Z|)"
        G_order = G_rep_Z.shape[0] + 1
        G_rep_Z_inv = torch.permute(G_rep_Z, dims=(0, 2, 1))
        # As identity e ∉ G.generators, we add the trivially transformed covariance matrix covX/covY/covYX to the sum.
        covZ_equiv = (covZ + torch.einsum('Gya,ab,Gbx->yx', G_rep_Z, covZ, G_rep_Z_inv)) / G_order
        covZp_equiv = (covZp + torch.einsum('Gya,ab,Gbx->yx', G_rep_Z, covZp, G_rep_Z_inv)) / G_order
        covZpZdt_equiv = (covZpZdt + torch.einsum('Gya,tab,Gbx->tyx', G_rep_Z, covZpZdt, G_rep_Z_inv)) / G_order
        # Update the covariance matrices with the equivariant constraints
        covZ, covZp, covZpZdt = covZ_equiv, covZp_equiv, covZpZdt_equiv

    # Compute both relaxed and un-relaxed scores, while computing gradients only for the selected score ============
    with torch.set_grad_enabled(grad_relaxed_score):
        # spectral_score[dt-1] := ||Cov(Z'_t+dt, Z_t)||^2_HS/(||Cov(Z')|| ||Cov(Z)|| | ∀ t, dt in [1, time_horizon]
        spectral_scores = vectorized_spectral_scores(covYXdt=covZpZdt,
                                                     covX=covZ,
                                                     covY=covZp,
                                                     run_checks=run_checks)
    with torch.set_grad_enabled(grad_correlation_score):
        # corr_score[dt-1] := ||Cov(Z')^1/2 Cov(Z'_t+dt, Z_t) Cov(Z)^1/2||^2_HS  | ∀ t, dt in [1, time_horizon]
        corr_scores = vectorized_correlation_scores(covYXdt=covZpZdt,
                                                    covX=covZ,  # illegal memory access if not clone
                                                    covY=covZp,  # dunno wtf is happening here.
                                                    run_checks=False)

    # Orthogonality regularization: || CovZZ - I || + || CovZ'Z' - I || ============================================
    I = torch.eye(latent_state_dim, device=covZ.device)
    orth_reg_Z = torch.linalg.norm(covZ - I, ord="fro")
    orth_reg_Zp = torch.linalg.norm(covZp - I, ord="fro")

    LatentSpaceMetrics = namedtuple(
        typename='LatentSpaceMetrics',
        field_names=['covZ', 'covZp', 'covZpZdt', 'spectral_scores', 'corr_scores', 'orth_reg_Z', 'orth_reg_Zp'])
    return LatentSpaceMetrics(covZ, covZp, covZpZdt, spectral_scores, corr_scores, orth_reg_Z, orth_reg_Zp)
