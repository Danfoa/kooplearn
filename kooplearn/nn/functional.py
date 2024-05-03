from escnn.group import Representation
from kooplearn._src.check_deps import check_torch_deps

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


def vectorized_cross_cov(X_contexts: torch.Tensor,
                         Y_contexts: Optional[torch.Tensor] = None,
                         max_dts: Optional[int] = None,
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
        max_dts: (int) Maximum number of distinct time differences dt in [1, min(min(time_horizon-1, max_dts)]
        considered to compute the cross-covariances. Default to `max_dts=time_horizon - 1`.
        run_checks: (bool) Run sanity checks to ensure the empirical estimates are correct. Default to False.
    Returns:
        CovYXdt: a tensor of shape (min(time_horizon-1, max_dts), |Y|, |X|) containing the Cross-Covariances
         per time difference dt. Such that CCov[dt-1] := Cov(y_i+dt, x_i) ∀ i dt in [0, time_horizon].
    """
    assert len(X_contexts.shape) == 3, f"state_traj: {X_contexts.shape}. Expected (batch, time_horizon, |X|)"
    assert max_dts is None or isinstance(max_dts, int), f"max_dts: {max_dts}. Expected None or int."

    num_samples, time_horizon, _ = X_contexts.shape
    dtype, device = X_contexts.dtype, X_contexts.device
    pred_horizon = time_horizon - 1
    max_dts = pred_horizon if max_dts is None else min(max_dts, pred_horizon)

    x_t = X_contexts[:, [0], :]  # (batch_size, 1, |X|)
    y_t_dt = Y_contexts[:, 1:max_dts + 1, :]  # (batch_size, max_dts, |Y|)

    # Compute cross-covariance in single batched/parallel operation
    CovYXdt = torch.einsum('bty,box->tyx', y_t_dt, x_t) / num_samples

    if run_checks:  # Sanity checks. These should be true by construction
        from kooplearn.nn.functional import covariance
        for dt in range(1, max_dts + 1):
            x_0 = X_contexts[:, 0, :]
            y_dt = Y_contexts[:, dt, :]
            CovYXdt_true = covariance(X=x_0, Y=y_dt, center=False)
            assert torch.allclose(CovYXdt[dt - 1], CovYXdt_true, rtol=1e-5, atol=1e-5), \
                f"Max error {torch.max(torch.abs(CovYXdt[dt - 1] - CovYXdt_true))}"

    return CovYXdt


def equivariant_vectorized_cross_cov(X_contexts: torch.Tensor,
                                     rep_X: Representation,
                                     Y_contexts: Optional[torch.Tensor] = None,
                                     rep_Y: Optional[Representation] = None,
                                     max_dts: Optional[int] = None,
                                     run_checks: bool = False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """ Compute empirical equivariant estimations of the cross-covariance operators.

    This function takes data from two random variables X and Y in the form of a trajectory of states in context window
    shape (n_samples, time_horizon, state_dim) there x_t, y_t are the states at time t in [0, time_horizon].

    It computes the equivariant cross covariance operators CovXYdt := Cov(x_i, y_i+dt) for all dt in [1, time_horizon],
    such that, `rep_Y(g) CovXYdt[k] = CovXYdt[k] rep_X(g)` for all g in G and k in [1, time_horizon]. This is achieved
    by first computing the empirical cross-covariance operators CovXYdt and then applying the group-average trick to
    improve the estimate by forcing it to be equivariant. For details see: "Group symmetry and covariance
    regularization" at https://people.lids.mit.edu/pari/group_symm.pdf Section 2.3. Basically, we improve the empirical
    estimate by applying:

    # CovYXdt = Cov(y_i+dt, x_i) in R^{|Y|x|X|}. That is CovYXdt: X -> Y
    CovYXdt := 1/|G| Σ_g ∈ G (ρ_Y(g) Cov(Y, X) ρ_X(g)^T

    Args:
        X_contexts: (batch, time_horizon, |X|) trajectory of states of random variable X
        Y_contexts: (batch, time_horizon, |Y|) trajectory of states of random variable Y.
        max_dts: (int) Maximum number of distinct time differences dt in [1, min(min(time_horizon-1, max_dts)]
        considered to compute the cross-covariances. Default to `max_dts=time_horizon - 1`.
        run_checks: (bool) Run sanity checks to ensure the empirical estimates are correct. Default to False.
    Returns:
        CCov: a tensor of shape (min(time_horizon-1, max_dts), state_dim, state_dim) containing the Cross-Covariances
         per time difference dt. Such that CCov[dt-1] := Cov(x_i, y_i+dt) ∀ i dt in [1, time_horizon-1].
    """
    CovYXdt = vectorized_cross_cov(X_contexts, Y_contexts, max_dts, run_checks)

    G = rep_X.group
    dtype, device = X_contexts.dtype, X_contexts.device
    rep_X_inv_block = torch.cat([torch.tensor(rep_X(~h)) for h in G.generators], dim=0)  # (|G|, |X|, |X|)
    rep_Y_block = torch.cat([torch.tensor(rep_Y(h)) for h in G.generators], dim=0)  # (|G|, |Y|, |Y|)

    # Compute group average:  1/|G| Σ_g ∈ G (ρ_Y(g) Cov(Y, X) ρ_X(g)^T) in single batched/parallel operation.
    # 'Gya,tab,Gbx->tyx' explains the operation ρ_Y(g) CovYXdt ρ_X(g)^T summed over g
    CovYXdt_equiv = torch.einsum('Gya,tab,Gbx->tyx', rep_Y_block, CovYXdt, rep_X_inv_block) / G.order

    if run_checks:  # Sanity checks. These should be true by construction
        for dt in range(1, max_dts + 1):
            CovYXdt = CovYXdt[dt - 1]
            G_CovYXdt = [CovYXdt]
            for h in G.generators:
                rep_X_g_inv = torch.tensor(rep_X(~h)).to(device=device)
                rep_X = torch.tensor(rep_Y(h)).to(device=device)
                CovYXdt_g = torch.einsum('ab,bc,cd->ad', rep_X, CovYXdt, rep_X_g_inv)
                G_CovYXdt.append(CovYXdt_g)
            CovYXdt_true = torch.sum(torch.tensor(G_CovYXdt), dim=0) / G.order

            assert torch.allclose(CovYXdt_equiv[dt - 1], CovYXdt_true, rtol=1e-5, atol=1e-5), \
                f"Max error {torch.max(torch.abs(CovYXdt_equiv[dt - 1] - CovYXdt_true))}"

    return CovYXdt_equiv

