from escnn.group import Representation
from kooplearn._src.check_deps import check_torch_deps

check_torch_deps()
from typing import Optional  # noqa: E402

import torch  # noqa: E402


def sqrtmh(A: torch.Tensor):
    # Credits to
    """Compute the square root of a Symmetric or Hermitian positive definite matrix or batch of matrices. Credits to  `https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228 <https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228>`_."""
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH


def covariance(X: torch.Tensor, Y: Optional[torch.Tensor] = None, center: bool = True):
    """Covariance matrix

    Args:
        X (torch.Tensor): Input covariates of shape ``(samples, features)``.
        Y (Optional[torch.Tensor], optional): Output covariates of shape ``(samples, features)`` Defaults to None.
        center (bool, optional): Whether to compute centered covariances. Defaults to True.

    Returns:
        torch.Tensor: Covariance matrix of shape ``(features, features)``. If ``Y is not None`` computes the cross-covariance between X and Y.
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
        return torch.mm(_X.T, _Y)


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
    cov_X, cov_Y, cov_XY = (
        covariance(X, center=center_covariances),
        covariance(Y, center=center_covariances),
        covariance(X, Y, center=center_covariances),
    )
    if schatten_norm == 2:
        # Using least squares in place of pinv for numerical stability
        M_X = torch.linalg.lstsq(cov_X, cov_XY).solution
        M_Y = torch.linalg.lstsq(cov_Y, cov_XY.T).solution
        return torch.trace(M_X @ M_Y)
    elif schatten_norm == 1:
        sqrt_cov_X = sqrtmh(cov_X)
        sqrt_cov_Y = sqrtmh(cov_Y)
        M = torch.linalg.multi_dot(
            [
                torch.linalg.pinv(sqrt_cov_X, hermitian=True),
                cov_XY,
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
        relaxed (bool, optional): Whether to use the relaxed (more numerically stable) or the full deep-projection loss. Defaults to True.
        metric_deformation (float, optional): Strength of the metric metric deformation loss: Defaults to 1.0.
        center_covariances (bool, optional): Use centered covariances to compute the VAMP score. Defaults to True.

    """
    cov_X, cov_Y, cov_XY = (
        covariance(X, center=center_covariances),
        covariance(Y, center=center_covariances),
        covariance(X, Y, center=center_covariances),
    )
    R_X = log_fro_metric_deformation_loss(cov_X)
    R_Y = log_fro_metric_deformation_loss(cov_Y)
    if relaxed:
        S = (torch.linalg.matrix_norm(cov_XY, ord="fro") ** 2 /
             (torch.linalg.matrix_norm(cov_X, ord=2) * torch.linalg.matrix_norm(cov_Y, ord=2)))
    else:
        M_X = torch.linalg.lstsq(cov_X, cov_XY).solution
        M_Y = torch.linalg.lstsq(cov_Y, cov_XY.T).solution
        S = torch.trace(M_X @ M_Y)
    return S - (0.5 * metric_deformation * (R_X + R_Y))


def log_fro_metric_deformation_loss(cov: torch.tensor):
    """Logarithmic + Frobenious metric deformation loss as used in :footcite:t:`Kostic2023DPNets`, defined as :math:`{{\\rm Tr}}(C^{2} - C -\ln(C))` .

    Args:
        cov (torch.tensor): A symmetric positive-definite matrix of shape (s, s) or a batch of matrices of shape (b, s, s).
    Returns:
        torch.tensor: The metric deformation loss of shape (1,) or (b,) depending on the input shape.
    """
    eps = torch.finfo(cov.dtype).eps * cov.shape[-1]
    vals_x = torch.linalg.eigvalsh(cov)
    vals_x = torch.where(vals_x > eps, vals_x, eps * torch.ones_like(vals_x))
    loss = torch.mean(-torch.log(vals_x) + vals_x * (vals_x - 1.0), dim=-1)
    return loss


def vectorized_cov_cross_cov(X_contexts: torch.Tensor,
                             Y_contexts: Optional[torch.Tensor] = None,
                             representation: Optional[Representation] = None,
                             cov_window_size: Optional[int] = None,
                             run_checks: bool = False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """ Compute empirical estimations of the covariance and cross-covariance operators from a state trajectory.

    TODO: Update documentation to explain purely in algebraic terms. Do not reference latent/feature spaces.
    This function computes the empirical approximation of the covariance and cross-covariance operators in
    batched/parallel matrix operations for efficiency.
    Args:
        X_contexts: (batch, time_horizon, state_dim) trajectory of states in main function space.
        Y_contexts: (batch, time_horizon, state_dim) trajectory of states in auxiliary function space.
        representation (optional Representation): Group representation on the state space. If provided, the empirical
         covariance and cross-covariance operators will be improved using the group average trick:
         Cov(0,i) = 1/|G| Σ_g ∈ G (ρ(g) Cov(x_0, x'_i) ρ(g)^-1), ensuring that the empirical operators are equivariant:
         Cov(0,i) ρ(g) = ρ(g) Cov(0,i) ∀ g ∈ G.
        run_checks: (bool) If True, check that the empirical operators are equivariant. Defaults to False.
    Returns:
        CCov: (time_horizon, time_horizon, state_dim, state_dim) Tensor containing all the Cross-Covariance
         empirical operators between the states in the main trajectory and states in the auxiliary trajectory.
         Each entry of the tensor is a (state_dim, state_dim) covariance operator estimate.
         Such that CCov(i,j) := Cov(x_i, x'_j) ∀ i, j in [0, time_horizon], j >= i.
        Cov: (time_horizon, state_dim, state_dim) Tensor containing all the Covariance empirical operators between time
         steps on the main state space. Cov(i) = Cov(x_i, x_i) ∀ i in [0, time_horizon]
        Cov_prime: (time_horizon, state_dim, state_dim) Tensor containing all the Covariance empirical operators between
         time steps on the auxiliary state space. Cov_prime(i) = Cov(x'_i, x'_i) ∀ i in [0, time_horizon]
    """
    assert len(X_contexts.shape) == 3, f"state_traj: {X_contexts.shape}. Expected (batch, time_horizon, state_dim)"
    assert Y_contexts is None or Y_contexts.shape == X_contexts.shape
    assert cov_window_size is None or isinstance(cov_window_size, int)
    num_samples, time_horizon, state_dim = X_contexts.shape
    dtype, device = X_contexts.dtype, X_contexts.device
    pred_horizon = time_horizon - 1
    cov_window_size = pred_horizon if cov_window_size is None else min(cov_window_size, pred_horizon)
    unique_function_space = Y_contexts is None

    if unique_function_space:  # If function space has a transfer-invariant density, no auxiliary function.
        Y_contexts = X_contexts

    # Expand state_traj to have an extra dimension for time_horizon
    state_traj_block = X_contexts.permute(1, 0, 2)  # (T, batch_size, state_dim)
    state_traj_block = state_traj_block.unsqueeze(1).expand(-1, time_horizon, -1, -1)  # (T, T', batch_size, state_dim)
    state_traj_block_prime = Y_contexts.permute(1, 0, 2)  # (T, batch_size, state_dim)
    state_traj_block_prime = state_traj_block_prime.unsqueeze(1).expand(-1, time_horizon, -1, -1)

    # Compute in a single tensor (parallel) operation all cross-covariance between time steps in original and auxiliary
    # state trajectories. CCov[i, j] := Cov(x_i, x'_j)
    CovXY = torch.einsum('...ob,...ba->...oa',  # -> (T, T', state_dim, state_dim)
                        state_traj_block.permute(0, 1, 3, 2),  # (T, T', state_dim, batch_size)
                        state_traj_block_prime.permute(1, 0, 2, 3)  # (T', T, batch_size, state_dim)
                        ) / num_samples

    if unique_function_space:  # If same function space Cov(t,t) is the diagonal of the cross-Cov matrix.
        CovX = CovXY[range(time_horizon), range(time_horizon)]
        CovY = CovX
    else:  # If diff function spaces, we need to compute the Covariance matrices for each state space.
        #      (T, state_dim, batch_size) @ (T, batch_size, state_dim) -> (T, state_dim, state_dim)
        CovX = X_contexts.permute(1, 2, 0) @ X_contexts.permute(1, 0, 2) / num_samples
        CovY = Y_contexts.permute(1, 2, 0) @ Y_contexts.permute(1, 0, 2) / num_samples

    if run_checks:  # Sanity checks. These should be true by construction
        for t in range(min(pred_horizon, cov_window_size)):
            from kooplearn.nn.functional import covariance
            CovXY_true = covariance(X=X_contexts[:, 0, :], Y=Y_contexts[:, t, :], center=False)
            assert torch.allclose(CovXY[0, t], CovXY_true, rtol=1e-5, atol=1e-5), \
                f"Max error {torch.max(torch.abs(CovXY[0, t] - CovXY_true))}"
            CovXY_true = covariance(X=X_contexts[:, t, :], Y=Y_contexts[:, t, :], center=False)
            assert torch.allclose(CovXY[t, t], CovXY_true, rtol=1e-5, atol=1e-5), \
                f"Max error {torch.max(torch.abs(CovXY[t, t] - CovXY_true))}"
            CovYY_true = covariance(X=Y_contexts[:, t, :], center=False)
            assert torch.allclose(CovY[t], CovYY_true, rtol=1e-5, atol=1e-5), \
                f"Max error {torch.max(torch.abs(CovY[t] - CovYY_true))}"

    if representation is not None:
        # # If state-space features a symmetry group we can improve the empirical estimates by understanding that the
        # theoretical operators are equivariant: ρ(g) Cov(X,Y) ρ(g)^T = Cov(ρ(g)X, ρ(g)Y) = Cov(X, Y) = CovXY
        # Thus we can apply the "group-average" trick to improve the estimate:
        # CovXY = 1/|G| Σ_g ∈ G (ρ(g) Cov(X,Y) ρ(g)^T) (see https://arxiv.org/abs/1111.7061)
        # This is a costly operation but is equivalent (without the memory overhead) to doing data augmentation of the
        # state space samples (with all group elements), and then computing the empirical covariance operators.
        # Furthermore, we can apply this operation in parallel to all Cov Ops for numerical efficiency in GPU.
        orbit_cross_Cov = [CovXY]
        orbit_Cov, orbit_Cov_prime = [CovX], [CovY]

        group = representation.group
        elements = group.generators if not group.continuous else group.grid(type='rand', N=10)
        for h in elements:  # Generators of the symmetry group. We only need these.
            # Compute each:      ρ(g) Cov(X,Y) ρ(g)^T   | ρ(g)^T = ρ(~g) = ρ(g^-1)
            rep_g = torch.tensor(representation(h), dtype=dtype, device=device)
            rep_g_inv = torch.tensor(representation(~h), dtype=dtype, device=device)
            #                                        t,l=time, n,m,a,o=state_dim
            orbit_cross_Cov.append(torch.einsum('na,ltao,om->ltnm', rep_g, CovXY, rep_g_inv))
            orbit_Cov.append(torch.einsum('na,tao,om->tnm', rep_g, CovX, rep_g_inv))
            orbit_Cov_prime.append(torch.einsum('na,tao,om->tnm', rep_g, CovY, rep_g_inv))

        # Compute group average:  1/|G| Σ_g ∈ G (ρ(g) Cov(X,Y) ρ(g)^T).
        CovXY = torch.mean(torch.stack(orbit_cross_Cov, dim=0), dim=0)
        CovX = torch.mean(torch.stack(orbit_Cov, dim=0), dim=0)
        CovY = torch.mean(torch.stack(orbit_Cov_prime, dim=0), dim=0)

        if run_checks:  # Check commutativity/equivariance of the empirical estimates of all Covariance operators
            for g in representation.group.elements:
                rep_h = torch.tensor(representation(g), dtype=dtype, device=device)
                cov_rep = torch.einsum('na,ltao->ltno', rep_h, CovXY)  # t,l=time, n,m,a,o=state_dim
                rep_cov = torch.einsum('ltao,om->ltam', CovXY, rep_h)
                window = min(pred_horizon, cov_window_size)
                assert torch.allclose(cov_rep[0, :window], rep_cov[0, :window], atol=1e-5), \
                    f"Max equivariance error {torch.max(torch.abs(cov_rep[0, :] - rep_cov[0, :]))}"
                # Check now commutativity of Cov and Cov_prime
                cov_rep = torch.einsum('na,tao->tno', rep_h, CovX)  # t=time, n,m,a,o=state_dim
                rep_cov = torch.einsum('tao,om->tam', CovX, rep_h)
                assert torch.allclose(cov_rep, rep_cov, atol=1e-5)
                cov_rep = torch.einsum('na,tao->tno', rep_h, CovY)  # t=time, n,m,a,o=state_dim
                rep_cov = torch.einsum('tao,om->tam', CovY, rep_h)
                assert torch.allclose(cov_rep, rep_cov, atol=1e-5)

    return CovXY, CovX, CovY
