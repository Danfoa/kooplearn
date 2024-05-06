import logging
import math
from functools import wraps
from typing import Optional, Union

import escnn.nn
import numpy as np
import torch
from escnn.nn import FieldType
from kooplearn._src.linalg import full_rank_lstsq, full_rank_equivariant_lstsq
from kooplearn._src.metrics import (chapman_kolmogorov_regularization, vectorized_spectral_scores,
                                    vectorized_correlation_scores)
from kooplearn.data import TensorContextDataset
from kooplearn.models import LatentBaseModel
from kooplearn.models.ae.utils import flatten_context_data, unflatten_context_data
from kooplearn.models.dpnet.dpnet import DPNet
from morpho_symm.utils.abstract_harmonics_analysis import isotypic_basis

from kooplearn.nn.functional import latent_space_metrics

logger = logging.getLogger(__name__)


class EquivDPNet(DPNet):

    def __init__(self,
                 encoder: type[escnn.nn.EquivariantModule],
                 encoder_kwargs: dict,
                 latent_dim: int,
                 loss_weights: Optional[dict] = None,
                 use_spectral_score: bool = True,
                 group_avg_trick: bool = False,
                 **kwargs,
                 ):
        assert 'out_type' not in encoder_kwargs.keys(), \
            f"Encoder `out_type` is automatically defined by {self.__class__.__name__}."
        # TODO: Dunno why not pass the instance of module instead of encoder(**encoder_kwargs).
        self.state_type: FieldType = encoder_kwargs.get('backbone').in_type
        # Define the group representation of the latent observable space, as multiple copies of the group representation
        # in the original state space. This latent group rep is defined in the `isotypic basis`.
        multiplicity = math.ceil(latent_dim / self.state_type.size)
        # Define the observation space representation in the isotypic basis. This function returns two `OrderedDict`
        # mapping iso-space ids (str) to `escnn.group.Representation` and dimensions (Slice) in the latent space.
        self.latent_iso_reps, self.latent_iso_dims = isotypic_basis(representation=self.state_type.representation,
                                                                    multiplicity=multiplicity,
                                                                    prefix='LatentState')
        # Thus, if you want the observables of the `isoX` latent subspace (isoX in self.latent_iso_reps.keys(), do:
        # z_isoX = z[..., self.latent_iso_dims[isoX]]  : where z is a vector of shape (..., latent_dim)
        # Similarly to apply the symmetry transformation to this vector-value observable field, do
        # g ▹ z_isoX = rep_IsoX(g) @ z_isoX     :    rep_IsoX = self.latent_iso_reps[isoX]
        # Define the latent group representation as a direct sum of the representations of each isotypic subspace.
        self.latent_state_type = FieldType(self.state_type.gspace,
                                           [rep_iso for rep_iso in self.latent_iso_reps.values()])

        encoder_kwargs['out_type'] = self.latent_state_type

        self.group_avg_trick = group_avg_trick

        super().__init__(encoder=encoder,
                         encoder_kwargs=encoder_kwargs,
                         latent_dim=latent_dim,
                         loss_weights=loss_weights,
                         use_spectral_score=use_spectral_score,
                         **kwargs)

    @wraps(LatentBaseModel.encode_contexts)
    def encode_contexts(
            self,
            state: TensorContextDataset,
            encoder: torch.nn.Module, **kwargs
            ) -> Union[dict, TensorContextDataset]:
        # Since encoder receives as input a escnn.nn.GeometricTensor, we need to mildly modify encode_contexts
        encoder_out = encoder(self.state_type(flatten_context_data(state)))
        # Handle case where the encoder outputs a tuple of the latent observables in distinct/lagged rep spaces H and H'
        distinct_spaces = isinstance(encoder_out, tuple)
        if distinct_spaces:  # z_t ∈ H and z'_t ∈ H'
            assert len(encoder_out) == 2, f"Expected encoder output (z_t, z'_t) {len(encoder_out)}"
            z_t, z_t_prime = encoder_out[0], encoder_out[1]
        else:  # z_t ∈ H and z'_t ∈ H
            z_t, z_t_prime = encoder_out, encoder_out

        latent_obs = unflatten_context_data(z_t.tensor,
                                            batch_size=len(state),
                                            features_shape=(self.latent_dim,))
        latent_obs_aux = latent_obs if not distinct_spaces else (
            unflatten_context_data(z_t_prime.tensor,
                                   batch_size=len(state),
                                   features_shape=(self.latent_dim,))
        )
        return dict(latent_obs=latent_obs, latent_obs_aux=latent_obs_aux)

    @wraps(DPNet.decode_contexts)  # Copies docstring from parent implementation
    def decode_contexts(
            self,
            latent_obs: TensorContextDataset,
            decoder: escnn.nn.EquivariantModule,
            **kwargs
            ) -> Union[dict, TensorContextDataset]:
        # Since an Equivariant decoder receives as input a escnn.nn.GeometricTensor, we need to:
        if isinstance(decoder, escnn.nn.EquivariantModule):
            # From (batch, context_length, latent_dim) to GeometricTensor(batch * context_length, latent_dim)
            flat_decoded_contexts = decoder(self.latent_state_type(flatten_context_data(latent_obs)))
            # From  GeometricTensor(batch * context_length, *features_shape) to (batch, context_length, *features_shape)
            decoded_contexts = unflatten_context_data(flat_decoded_contexts.tensor,
                                                      batch_size=len(latent_obs),
                                                      features_shape=self.state_features_shape)
        else:
            decoded_contexts = super(EquivDPNet, self).decode_contexts(latent_obs, decoder, **kwargs)

        return decoded_contexts

    def compute_loss_and_metrics(self,
                                 state: Optional[TensorContextDataset] = None,
                                 pred_state: Optional[TensorContextDataset] = None,
                                 latent_obs: Optional[TensorContextDataset] = None,
                                 latent_obs_aux: Optional[TensorContextDataset] = None,
                                 pred_latent_obs: Optional[TensorContextDataset] = None,
                                 ) -> dict[str, torch.Tensor]:
        run_checks = False  # TODO: remove form here
        device, dtype = latent_obs.data.device, latent_obs.data.dtype
        # Compute the scores and metrics for each isotypic subspace ====================================================
        iso_spaces_metrics = {irrep_id: None for irrep_id in self.latent_iso_reps.keys()}

        if not hasattr(self, '_G_rep_Z'):  # load only once to GPU for efficiency
            rep_Z = self.latent_state_type.representation
            H = self.state_type.fibergroup.generators  # Generators of the group without the identity element e
            self._G_rep_Z = torch.stack([torch.tensor(rep_Z(h)) for h in H], dim=0).to(device=device, dtype=dtype)

        for irrep_id, iso_rep in self.latent_iso_reps.items():
            iso_dims = list(self.latent_iso_dims[irrep_id])
            dim_s, dim_e = min(iso_dims), max(iso_dims) + 1
            G_rep_iso = self._G_rep_Z[..., dim_s:dim_e, dim_s:dim_e]

            lat_space_metrics = latent_space_metrics(Z_contexts=latent_obs[..., iso_dims],
                                                     Z_prime_contexts=latent_obs_aux[..., iso_dims],
                                                     G_rep_Z=G_rep_iso,
                                                     grad_relaxed_score=self.use_spectral_score,
                                                     grad_correlation_score=not self.use_spectral_score,
                                                     run_checks=run_checks)
            iso_spaces_metrics[irrep_id] = lat_space_metrics

        # Compute the total loss and metrics ===========================================================================
        metrics = self.iso2latent_space_metrics(iso_spaces_metrics)
        spectral_scores, corr_scores = metrics.pop('spectral_scores'), metrics.pop('corr_scores')
        orth_reg_loss = metrics.pop('orth_reg_loss')
        # Select the score measuring the invariance of the latent space to the Koopman operator.
        space_inv_score = spectral_scores if self.use_spectral_score else corr_scores
        score = space_inv_score.mean()  # TODO: Exponential weighting on time horizon.
        # Add the orthonormality regularization term to the loss
        alpha_orth = self.loss_weights.get('orthonormality', 1.0)
        orth_regularization = (alpha_orth * self.latent_dim) * orth_reg_loss.mean()
        score = score - orth_regularization
        # Change sign to minimize the loss and maximize the score. (Optimizer configured for gradient descent)
        loss = -score

        with torch.no_grad():
            metrics.update(spectral_score=spectral_scores.mean().item(),
                           corr_score=corr_scores.mean().item(),
                           score_gap=torch.abs(corr_scores - spectral_scores).mean().item(),
                           orth_reg=orth_reg_loss.mean().item())
            # If the model is fitted and state/latent_obs_state predictions are given, compute the prediction metrics.
            pred_metrics = LatentBaseModel.compute_loss_and_metrics(self,
                                                                    state=state,
                                                                    pred_state=pred_state,
                                                                    latent_obs=latent_obs,
                                                                    pred_latent_obs=pred_latent_obs)
            metrics.update(pred_metrics)

        return metrics | dict(loss=loss)

    def estimate_cov_cross_cov(self, latent_obs, latent_obs_aux, run_checks):
        """ Compute empirical equivariant estimations of the covariance and cross-covariance operators

        This function takes data from two random variables X and Y in the form of a trajectory of states in context
        window shape (n_samples, time_horizon, state_dim) there x_t, y_t are the states at time t in [0, time_horizon].

        The Variance operator VarX := Cov(X, X) ≈ 1/n Σ_i x_i x_i^T should be the same operator when estimated by the
        symmetry transformed states g·x_i = ρ_X(g) x_i ∀ g ∈ G. Therefore we have that:
        Cov(X, X) ≈ 1/n Σ_i (g·x_i) (g·x_i)^T = 1/n Σ_i ρ_X(g) x_i x_i^T ρ_X(g)^-1 = ρ_X(g) Cov(X, X) ρ_X(g)^-1.

        To improve the empirical estimation we compute the empirical estimation and then apply the group-average trick:
        VarX := 1/|G| Σ_g ∈ G (ρ_X(g) Cov(X, X) ρ_X(g)^-1 ,
        VarY := 1/|G| Σ_g ∈ G (ρ_Y(g) Cov(Y, Y) ρ_Y(g)^-1 , and
        CovXYdt := 1/|G| Σ_g ∈ G (ρ_Y(g) Cov(Y, X) ρ_X(g)^-1

        For details see: "Group symmetry and covariance regularization" at
        https://people.lids.mit.edu/pari/group_symm.pdf Section 2.3.

        Args:
            latent_obs: (batch, time_horizon, |X|) trajectory of states of random variable X
            latent_obs_aux: (batch, time_horizon, |Y|) trajectory of states of random variable Y.
            run_checks: (bool) Run sanity checks to ensure the empirical estimates are correct. Default to False.
        Returns:

        """
        covX, covY, covYXdt = super(EquivDPNet, self).estimate_cov_cross_cov(latent_obs, latent_obs_aux, run_checks)
        # Improve empirical covariance estimation by using the representations of the latent observable spaces.

        G = self.latent_state_type.fibergroup
        dtype, device = covYXdt.dtype, covYXdt.device

        if not hasattr(self, '_G_rep_Z'):  # load only once to GPU for efficiency
            rep_Z = self.latent_state_type.representation
            H = G.generators  # Generators of the group without the identity element e
            self._G_rep_Z = torch.stack([torch.tensor(rep_Z(h)) for h in H], dim=0).to(device=device, dtype=dtype)

        G_rep_Z = self._G_rep_Z.to(device=device, dtype=dtype)
        G_rep_Z_inv = torch.permute(G_rep_Z, dims=(0, 2, 1))

        # As identity e ∉ G.generators, we add the trivially transformed covariance matrix covX/covY/covYX to the sum.
        covX_equiv = (covX + torch.einsum('Gya,ab,Gbx->yx', G_rep_Z, covX, G_rep_Z_inv)) / G.order()
        covY_equiv = (covY + torch.einsum('Gya,ab,Gbx->yx', G_rep_Z, covY, G_rep_Z_inv)) / G.order()
        covYXdt_equiv = (covYXdt + torch.einsum('Gya,tab,Gbx->tyx', G_rep_Z, covYXdt, G_rep_Z_inv)) / G.order()

        if run_checks:  # Sanity checks. These should be true by construction
            rep_Z = self.latent_state_type.representation
            covX_e, covY_e = covX, covY
            G_covX, G_covY = [covX_e], [covY_e]
            for h in G.generators:
                rep_Z_g_inv = torch.tensor(rep_Z(~h)).to(device=device, dtype=dtype)
                rep_Z_g = torch.tensor(rep_Z(h)).to(device=device, dtype=dtype)
                covX_g = torch.einsum('ab,bc,cd->ad', rep_Z_g, covX_e, rep_Z_g_inv)
                G_covX.append(covX_g)
                covY_g = torch.einsum('ab,bc,cd->ad', rep_Z_g, covY_e, rep_Z_g_inv)
                G_covY.append(covY_g)
            covX_equiv_true = torch.sum(torch.stack(G_covX), dim=0) / G.order()
            assert torch.allclose(covX_equiv, covX_equiv_true, rtol=1e-5, atol=1e-5), \
                f"Max error {torch.max(torch.abs(covX_equiv - covX_equiv_true))}"
            covY_equiv_true = torch.sum(torch.stack(G_covY), dim=0) / G.order()
            assert torch.allclose(covY_equiv, covY_equiv_true, rtol=1e-5, atol=1e-5), \
                f"Max error {torch.max(torch.abs(covY_equiv - covY_equiv_true))}"

        return covX_equiv, covY_equiv, covYXdt_equiv

    def fit_evolution_operator(self, Z_t, Z_t_dt, bias=False):
        T, B = full_rank_equivariant_lstsq(X=Z_t.T, Y=Z_t_dt.T, bias=bias, group_average=False)
        linear_dynamics = torch.nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim, bias=bias)
        linear_dynamics.weight.data = T
        linear_dynamics.weight.requires_grad = False
        if bias:
            linear_dynamics.bias.data = torch.tensor(B, dtype=torch.float32)
            linear_dynamics.bias.requires_grad = False
        return linear_dynamics

    @torch.no_grad()
    def fit_linear_decoder(self, latent_states: torch.Tensor, states: torch.Tensor,
                           bias: bool = False) -> torch.nn.Module:
        """Fit a linear decoder mapping the latent state space Z to the state space X. use for mode decomp."""

        logger.info(f"Fitting linear decoder for {self.__class__.__name__} model")

        # Solve the least squares problem to find the linear decoder matrix and bias
        D, B = full_rank_equivariant_lstsq(X=latent_states,
                                           Y=states,
                                           rep_X=self.latent_state_type.representation,
                                           rep_Y=self.state_type.representation,
                                           group_average=False,
                                           bias=bias)

        # Check shape
        _expected_shape = (np.prod(self.state_features_shape), self.latent_dim)
        assert D.shape == _expected_shape, \
            f"Expected linear decoder shape {_expected_shape}, got {D.shape}"

        # Create a non-trainable linear layer to store the linear decoder matrix and bias term
        lin_decoder = torch.nn.Linear(in_features=self.latent_dim,
                                      out_features=np.prod(self.state_features_shape),
                                      bias=bias)
        lin_decoder.weight.data = D
        lin_decoder.weight.requires_grad = False

        if bias:
            lin_decoder.bias.data = torch.tensor(B, dtype=torch.float32)
            lin_decoder.bias.requires_grad = False

        return lin_decoder

    def iso2latent_space_metrics(self, iso_spaces_metrics: dict) -> dict:
        """ Compute the observable space metrics from the isotypic subspaces metrics.

        This function exploits the fact that the Hilbert-Schmidt (HS) norm of an operator (or the Frobenious norm
        of a matrix) that is block-diagonal is defined as the square root of the sum of the squared norms of the blocks:
        ||A||_HS = sqrt(||A_o||_HS^2 + ... + ||A_i||_HS^2)  | A := block_diag(A_o, ..., A_i).
        Thus, we have that the relaxed and unrelaxed DPNet scores can be decomposed into the score of the Iso Subspaces.
        corr_score = ||CovX^-1/2 CovXY CovY^-1/2||_HS = √(∑_iso(||Cov_iso(X)^-1/2 Cov_iso(XY) Cov_iso(Y)^-1/2||_HS))
        orth_reg_loss = ||CovX - I||_fro = √(∑_iso(||Cov_iso(X) - I||^2_fro))

        Args:
            iso_spaces_metrics:
        Returns:
            Dictionary containing:
        """
        iso_space_ids = list(self.latent_iso_reps.keys())
        # Compute the entire obs space Orthonormal regularization terms for all time horizon.
        covX_iso_orth_reg = torch.vstack([iso_spaces_metrics[irrep_id].orth_reg_Z for irrep_id in iso_space_ids])
        covX_iso_orth_reg = torch.sqrt(torch.sum(covX_iso_orth_reg ** 2, dim=0))
        covY_iso_orth_reg = torch.vstack([iso_spaces_metrics[irrep_id].orth_reg_Zp for irrep_id in iso_space_ids])
        covY_iso_orth_reg = torch.sqrt(torch.sum(covY_iso_orth_reg ** 2, dim=0))

        orth_reg_loss = (covX_iso_orth_reg + covY_iso_orth_reg) / 2.0

        # Compute the Correlation/Projection score
        corr_scores_iso = torch.stack([iso_spaces_metrics[irrep_id].corr_scores for irrep_id in iso_space_ids])
        corr_scores = torch.sum(corr_scores_iso, dim=0)

        # Compute the Spectral scores for the entire obs-space
        iso_S_score = torch.stack([iso_spaces_metrics[irrep_id].spectral_scores for irrep_id in iso_space_ids])
        spectral_scores = torch.sum(iso_S_score, dim=0)

        metrics = dict(spectral_scores=spectral_scores,
                       corr_scores=corr_scores,
                       orth_reg_loss=orth_reg_loss)

        with torch.no_grad():
            eigvals_X_iso = [torch.linalg.eigvalsh(iso_spaces_metrics[irrep_id].covZ) for irrep_id in iso_space_ids]
            cond_covX_iso = torch.tensor([torch.abs(eigvals_X.max() / eigvals_X.min()) for eigvals_X in eigvals_X_iso])
            # Get the vector of eigenvalues of the covariance matrices of the latent observable spaces
            eigvals_X = torch.concatenate(eigvals_X_iso)
            eps = 2 * torch.finfo(eigvals_X.dtype).eps
            sval_max_X, sval_min_X = torch.abs(eigvals_X.max()), torch.abs(eigvals_X.min())
            rank_Z = torch.sum(eigvals_X > torch.max(eigvals_X) * 5 * eps)
            metrics.update(rank_Z=rank_Z.item(),
                           cond_Z=(torch.abs(sval_max_X / sval_min_X)).item(),
                           cond_Z_iso_min=cond_covX_iso.min().item(),
                           covZ_norm=sval_max_X.item(),
                           covZ_sval_min=sval_min_X.item())
            metrics.update({f"condZ_iso/{i}": cond_covX_iso[i].item() for i in range(len(cond_covX_iso))})
        return metrics
