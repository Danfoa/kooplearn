import logging
import math
from functools import wraps
from typing import Optional, Union

import numpy as np
import scipy

from kooplearn._src.check_deps import check_torch_deps
from kooplearn._src.linalg import full_rank_lstsq
from kooplearn._src.metrics import (chapman_kolmogorov_regularization, vectorized_spectral_scores,
                                    vectorized_correlation_scores)
from kooplearn.data import TensorContextDataset  # noqa: E402
from kooplearn.models.ae.utils import flatten_context_data, unflatten_context_data
from kooplearn.models.base_model import LatentBaseModel, LightningLatentModel, _default_lighting_trainer
from kooplearn.nn.functional import log_fro_metric_deformation_loss, vectorized_cov_cross_cov, latent_space_metrics
from kooplearn.utils import check_if_resume_experiment

check_torch_deps()
import lightning  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class DPNet(LatentBaseModel):
    """ TODO """

    def __init__(self,
                 encoder: type[torch.nn.Module],
                 encoder_kwargs: dict,
                 latent_dim: int,
                 loss_weights: Optional[dict] = None,
                 use_spectral_score: bool = True,
                 center_covariance: bool = False,
                 ):
        super().__init__()  # Initialize torch.nn.Module

        self.encoder = encoder(**encoder_kwargs)
        self.latent_dim = latent_dim
        self.use_spectral_score = use_spectral_score
        self.center_covariances = center_covariance
        self.loss_weights = loss_weights if loss_weights is not None else dict(
            orthonormality=1.0,
            reconstruction=0.0)

        self.decoder: Union[torch.nn.Linear, None] = None

    def fit(
            self,
            train_dataloaders: Optional[Union[DataLoader, list[DataLoader]]] = None,
            val_dataloaders: Optional[Union[DataLoader, list[DataLoader]]] = None,
            datamodule: Optional[lightning.LightningDataModule] = None,
            trainer: Optional[lightning.Trainer] = None,
            optimizer_fn: torch.optim.Optimizer = torch.optim.Adam,
            optimizer_kwargs: Optional[dict] = None,
            ):
        self._is_fitted = False
        self.trainer = _default_lighting_trainer() if trainer is None else trainer
        assert isinstance(self.trainer, lightning.Trainer)

        self._check_dataloaders_and_shapes(datamodule, train_dataloaders, val_dataloaders)

        # TODO: if user wants custom LightningModule, we should allow them to pass it.
        #   Making this module a class attribute leads to the problem of recursive parameter search between LatentVar
        lightning_module = LightningLatentModel(
            latent_model=self,
            optimizer_fn=optimizer_fn,
            optimizer_kwargs=optimizer_kwargs,
            )

        # Fit the encoder, decoder and (optionally) the evolution operator =============================================
        ckpt_call = self.trainer.checkpoint_callback  # Get trainer checkpoint callback if any
        training_done, ckpt_path, best_path = check_if_resume_experiment(ckpt_call)

        if training_done:
            best_ckpt = torch.load(best_path)
            lightning_module.eval()
            self.decoder = torch.nn.Linear(in_features=self.latent_dim, out_features=np.prod(self.state_features_shape),
                                           bias=self.center_covariances)
            self.lin_decoder = self.decoder
            self.linear_dynamics = torch.nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim,
                                                   bias=False)
            lightning_module.load_state_dict(best_ckpt['state_dict'], strict=False)
            # from lightning.pytorch.trainer.states import TrainerStatus
            # self.trainer.state.status = TrainerStatus.FINISHED
            # self._is_fitted = True
            logger.info(f"Skipping training. Loaded best model from {best_path}")
        else:
            self.trainer.fit(
                model=lightning_module,
                train_dataloaders=train_dataloaders,
                val_dataloaders=val_dataloaders,
                datamodule=datamodule,
                ckpt_path=ckpt_path if ckpt_path.exists() else None,
                )

            with torch.no_grad():
                # Get latent observables of training dataset ===============================================================
                train_dataloader = train_dataloaders if train_dataloaders is not None else datamodule.train_dataloader()
                X, Z = self._predict_training_data(trainer, lightning_module, train_dataloader, ckpt_path=best_path)

                if self.lin_decoder is None:  # The model was trained without basis expansion loss
                    # Fit linear decoder to perform dynamic mode decomposition =============================================
                    Z_flat = flatten_context_data(Z)  # (n_samples * context_len, latent_dim)
                    X_flat = flatten_context_data(X).to(device=Z_flat.device)  # (n_samples * context_len, state_dim)
                    # Store the linear_decoder=`Ψ⁻¹_lin: Z -> X` as a non-trainable `torch.nn.Linear` Module.
                    self.decoder = self.fit_linear_decoder(
                        states=X_flat.T, latent_states=Z_flat.T, bias=self.center_covariances
                        )
                    self.lin_decoder = self.decoder

                # Fit the evolution operator ===============================================================================
                # TODO: Should use Linear class of Kooplearn and some default Tikonov regularization.
                Z_t = flatten_context_data(Z[:, :-1, :])
                Z_t_dt = flatten_context_data(Z[:, 1:, :])
                self.linear_dynamics = self.fit_evolution_operator(Z_t, Z_t_dt)

                # Update the checkpoint file with the fitted linear decoder and evolution operator =========================
                if best_path.exists():
                    # Update the checkpoint file with the fitted linear decoder and evolution operator
                    ckpt = torch.load(best_path)
                    ckpt['state_dict'].update(**lightning_module.state_dict())
                    torch.save(ckpt, best_path)

        self._is_fitted = True

    def forward(self, state_contexts: TensorContextDataset) -> any:
        r"""Forward pass of the DPNet model.

        If unfitted, this method will default only to return the output of `encode_contexts`, else it will return the
        output of `evolve_forward`.

        Args:
            state_contexts (TensorContextDataset): The state to be evolved forward. This should be a trajectory of
            states
            :math:`(x_t)_{t\\in\\mathbb{T}}` in the context window :math:`\\mathbb{T}`.
        Returns:
            Any: The output of the forward pass.
        """
        if self.is_fitted:
            return self.evolve_forward(state_contexts)
        else:
            encoder_out = self.encode_contexts(state=state_contexts, encoder=self.encoder)
            return encoder_out

    @wraps(LatentBaseModel.encode_contexts)
    def encode_contexts(
            self,
            state: TensorContextDataset,
            encoder: torch.nn.Module, **kwargs
            ) -> Union[dict, TensorContextDataset]:
        encoder_out = encoder(flatten_context_data(state))
        # Handle case where the encoder outputs a tuple of the latent observables in distinct/lagged rep spaces H and H'
        distinct_spaces = isinstance(encoder_out, tuple)
        if distinct_spaces:  # z_t ∈ H and z'_t ∈ H'
            assert len(encoder_out) == 2, f"Expected encoder output (z_t, z'_t) {len(encoder_out)}"
            z_t, z_t_prime = encoder_out[0], encoder_out[1]
        else:  # z_t ∈ H and z'_t ∈ H
            z_t, z_t_prime = encoder_out, encoder_out

        latent_obs = unflatten_context_data(z_t,
                                            batch_size=len(state),
                                            features_shape=(self.latent_dim,))
        latent_obs_aux = latent_obs if not distinct_spaces else (
            unflatten_context_data(z_t_prime,
                                   batch_size=len(state),
                                   features_shape=(self.latent_dim,))
        )
        return dict(latent_obs=latent_obs, latent_obs_aux=latent_obs_aux)

    def compute_loss_and_metrics(self,
                                 state: Optional[TensorContextDataset] = None,
                                 pred_state: Optional[TensorContextDataset] = None,
                                 latent_obs: Optional[TensorContextDataset] = None,
                                 latent_obs_aux: Optional[TensorContextDataset] = None,
                                 pred_latent_obs: Optional[TensorContextDataset] = None,
                                 ) -> dict[str, torch.Tensor]:
        """ Compute DPNet loss term and metrics of the current latent observable space.
        Args:
        Returns:
            loss: Scalar tensor containing the DPNet loss.
        """
        metrics = dict()

        run_checks = False  # TODO: remove from here, make a test of this.

        lat_space_metrics = latent_space_metrics(Z_contexts=latent_obs,
                                                 Z_prime_contexts=latent_obs_aux,
                                                 grad_relaxed_score=self.use_spectral_score,
                                                 grad_correlation_score=not self.use_spectral_score,
                                                 center_covariances=self.center_covariances,
                                                 run_checks=run_checks)

        # Compute the loss  ============================================================================================
        # Select the score measuring the invariance of the latent space to the Koopman operator.
        spectral_scores, corr_scores = lat_space_metrics.spectral_scores, lat_space_metrics.corr_scores
        space_inv_score = spectral_scores if self.use_spectral_score else corr_scores
        score = space_inv_score.mean()  # TODO: Exponential weighting on time horizon.

        # Add the orthonormality regularization term to the loss ___
        alpha_orth = self.loss_weights.get('orthonormality', 1.0)
        orth_reg_loss = (lat_space_metrics.orth_reg_Z + lat_space_metrics.orth_reg_Zp) / 2
        orth_regularization = (alpha_orth * self.latent_dim) * orth_reg_loss
        score = score - orth_regularization

        # Change sign to minimize the loss and maximize the score. (Optimizer configured for gradient descent)
        loss = -score

        # Additional metrics ===========================================================================================
        with torch.no_grad():
            covX, covY = lat_space_metrics.covZ, lat_space_metrics.covZp
            eigvals_X = torch.linalg.eigvalsh(covX)  # Uses inversion, which defeats the purpose of the relaxed score.
            eps = 2 * torch.finfo(eigvals_X.dtype).eps
            eig_max_X, eig_min_X = eigvals_X.max(), eigvals_X.min()
            # Compute the condition number of the covariance matrices
            metrics.update(spectral_score=spectral_scores.mean().item(),
                           corr_score=corr_scores.mean().item(),
                           score_gap=torch.abs(corr_scores - spectral_scores).mean().item(),
                           orth_reg=orth_reg_loss.mean().item(),
                           rank_Z=torch.sum(eigvals_X > torch.max(eigvals_X) * 5 * eps).item(),
                           cond_Z=torch.abs(eig_max_X) / torch.abs(eig_min_X),
                           covZ_norm=eig_max_X.item(),
                           covZ_sval_min=eig_min_X.item())
            # If the model is fitted and state/latent_obs_state predictions are given, compute the prediction metrics.
            pred_metrics = super().compute_loss_and_metrics(state=state,
                                                            pred_state=pred_state,
                                                            latent_obs=latent_obs,
                                                            pred_latent_obs=pred_latent_obs)
            metrics.update(pred_metrics)

        return metrics | dict(loss=loss)

    def fit_evolution_operator(self, Z_t, Z_t_dt):
        T, B = full_rank_lstsq(X=Z_t.T, Y=Z_t_dt.T, bias=self.center_covariances)
        linear_dynamics = torch.nn.Linear(
            in_features=self.latent_dim, out_features=self.latent_dim, bias=self.center_covariances)
        linear_dynamics.weight.data = T
        linear_dynamics.weight.requires_grad = False
        if self.center_covariances:
            linear_dynamics.bias.data = B
            linear_dynamics.bias.requires_grad = False
        return linear_dynamics

    @wraps(LatentBaseModel.modes)
    def modes(self,
              state: TensorContextDataset,
              predict_observables: bool = False,
              ):
        modes_info = super().modes(state, predict_observables=predict_observables)
        # Store the linear decoder for mode decomposition in original state space.
        modes_info.linear_decoder = self.decoder
        return modes_info

    @property
    def evolution_operator(self):
        if self.is_fitted:
            if self.linear_dynamics is None:
                raise RuntimeError(f"Evolution operator was not fitted during ")
            return self.linear_dynamics.weight
        else:
            raise RuntimeError(f"Evolution operator cannot be learned before the model is fitted.")

    @property
    def lookback_len(self) -> int:
        return 1

    @property
    def is_fitted(self):
        # if self.trainer is None:
        #     return False
        # else:
        #     return self.trainer.state.finished
        return self._is_fitted

    @wraps(LatentBaseModel._dry_run)
    def _dry_run(self, state: TensorContextDataset):
        # Before latent space is learned DPNet wont be able to do predictions, so we need to modify the dry_run method.
        # To check that the encoding is producing the expected outcomes
        class_name = self.__class__.__name__

        assert self.state_features_shape is not None, f"state_features_shape not identified for {class_name}"

        x_t = state
        encode_out = self.encode_contexts(state=x_t, encoder=self.encoder)

        try:
            z_t = encode_out if isinstance(encode_out, TensorContextDataset) else encode_out["latent_obs"]
        except KeyError as e:
            raise KeyError(f"Missing output of {class_name}.evolve_forward") from e

        # Check latent observable contexts shapes
        assert z_t.shape[-1] == self.latent_dim, \
            f"Expected latent observable state of dimension (...,{self.latent_dim}), but got {z_t.shape}"
