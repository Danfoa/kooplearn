import logging
import math
from functools import wraps
from typing import Optional, Union

import numpy as np
import scipy

from kooplearn._src.check_deps import check_torch_deps
from kooplearn._src.linalg import vectorized_cov_cross_cov, full_rank_lstsq
from kooplearn._src.metrics import (vectorized_spectral_scores,
                                    vectorized_correlation_scores)
from kooplearn.data import TensorContextDataset  # noqa: E402
from kooplearn.models.ae.utils import flatten_context_data
from kooplearn.models.base_model import LatentBaseModel, LightningLatentModel, _default_lighting_trainer
from kooplearn.nn.functional import log_fro_metric_deformation_loss
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
                 ):
        super().__init__()  # Initialize torch.nn.Module

        self.encoder = encoder(**encoder_kwargs)
        self.latent_dim = latent_dim
        self.use_spectral_score = use_spectral_score
        self.loss_weights = loss_weights if loss_weights is not None else dict(orthonormality=1.0)
        #
        self._is_fitted = False
        self.linear_dynamics: Union[torch.nn.Linear, None] = None
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

        # TODO: if user wants custom LightningModule, we should allow them to pass it.
        #   Making this module a class attribute leads to the problem of recursive parameter search between LatentVar
        #
        lightning_module = LightningLatentModel(
            latent_model=self,
            optimizer_fn=optimizer_fn,
            optimizer_kwargs=optimizer_kwargs,
            )

        self._check_dataloaders_and_shapes(datamodule, train_dataloaders, val_dataloaders)

        # Fit the encoder, decoder and (optionally) the evolution operator =============================================
        ckpt_call = self.trainer.checkpoint_callback  # Get trainer checkpoint callback if any
        training_done, ckpt_path, best_path = check_if_resume_experiment(ckpt_call)

        if training_done:
            best_ckpt = torch.load(best_path)
            lightning_module.eval()
            self.decoder = torch.nn.Linear(in_features=self.latent_dim, out_features=np.prod(self.state_features_shape))
            self.linear_dynamics = torch.nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim)
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

            # if self.decoder is None or self.evolution_operator is None:  # After latent space Z is learned,
            # we fit a linear decoder
            logger.info(f"Fitting linear decoder for mode decomposition")
            # TODO: remove from here, this seems to be experiment specific.
            if datamodule is not None and hasattr(datamodule, "augment"):
                datamodule.augment = False

            # Get latent observables of training dataset
            train_dataset = train_dataloaders.dataset if train_dataloaders is not None else datamodule.train_dataset
            train_dataloader = train_dataloaders if train_dataloaders is not None else datamodule.train_dataloader()
            predict_out = trainer.predict(model=lightning_module,
                                          dataloaders=train_dataloader,
                                          ckpt_path=best_path if best_path.exists() else None)
            Z = [out_batch['latent_obs'] for out_batch in predict_out]
            Z = TensorContextDataset(torch.cat([z.data for z in Z], dim=0))  # (n_samples, context_len, latent_dim)

            # TODO: Make this a separate method
            Z_0 = Z.data[:, 0, :]
            Z_1 = Z.data[:, 1, :]
            T, _ = full_rank_lstsq(X=Z_0.T, Y=Z_1.T, bias=False)

            linear_dynamics = torch.nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim, bias=False)
            linear_dynamics.weight.data = T
            linear_dynamics.weight.requires_grad = False
            self.linear_dynamics = linear_dynamics

            # (n_samples * context_len, latent_dim)
            Z_flat = flatten_context_data(Z)
            # (n_samples * context_len, state_dim)
            X_flat = flatten_context_data(train_dataset).to(device=Z_flat.device)

            # Store the linear_decoder=`Ψ⁻¹_lin: Z -> X` as a non-trainable `torch.nn.Linear` Module.
            self.decoder = self.fit_linear_decoder(states=X_flat.T, latent_states=Z_flat.T)

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
            latent_obs = self.encode_contexts(state=state_contexts, encoder=self.encoder)
            return dict(latent_obs=latent_obs)

    def compute_loss_and_metrics(self,
                                 state: Optional[TensorContextDataset] = None,
                                 pred_state: Optional[TensorContextDataset] = None,
                                 latent_obs: Optional[TensorContextDataset] = None,
                                 latent_obs_aux: Optional[TensorContextDataset] = None,
                                 pred_latent_obs: Optional[TensorContextDataset] = None,
                                 **kwargs
                                 ) -> dict[str, torch.Tensor]:
        """ Compute DPNet loss term and metrics of the current latent observable space.

        Args:
            spectral_score: (time_horizon - 1) Tensor containing the average spectral score between time steps separated
             apart by a shift of `dt` [steps/time]. That is:
             spectral_score[dt - 1] = avg(||Cov(z_i, z'_i+dt)||_HS^2/(||Cov(z_i, z_i)||_2*||Cov(z'_i+dt, z'_i+dt)||_2))
             | ∀ i in [0, time_horizon - dt], dt in [1, min(time_horizon - i, window_size)]
            corr_score: (time_horizon - 1) Tensor containing the correlation scores between time steps separated
             apart by a shift of `dt` [steps/time]. That is:
             corr_score[dt - 1] = avg(||Cov(z_i, z_i)^-1 Cov(z_i, z'_i+dt) Cov(z'_i+dt, z'_i+dt)^-1||_HS^2)
             | ∀ i in [0, time_horizon - dt], dt in [1, min(time_horizon - i, window_size)]
            orth_reg: (time_horizon) Tensor containing the orthonormality regularization term for each time step.
             That is orth_reg[t] = || Cov(t,t) - I ||_2
            ck_reg: (time_horizon - 1,) Average CK error per `dt` time steps. That is:
             ck_error[dt - 2] = avg(|| Cov(t, t+dt) - Cov(t, t+1) Cov(t+1, t+2) ... Cov(t+dt-1, t+dt) ||)
             | ∀ t in [0, time_horizon - 2], dt in [2, min(time_horizon - 2, ck_window_length)]
        Returns:
            loss: Scalar tensor containing the DPNet loss.
        """
        metrics = dict(loss=float('nan'))

        if not self.is_fitted:
            run_checks = False  # TODO: remove from here, make a test of this.
            # Compute the empirical covariance and cross-covariance operators, ensuring that operators are equivariant.
            # CCov[i, j]   := Cov(z_i, z'_j)     | i,j in [time_horizon], j > i  --> i.e.,  Upper triangular tensor
            # Cov[t]       := Cov(z_t, z_t)      | t in [time_horizon]
            # Cov_prime[t] := Cov(z'_t,z'_t)     | t in [time_horizon]
            dt = 1
            CCov, Cov, Cov_prime = vectorized_cov_cross_cov(
                X_contexts=latent_obs.data[:, :-dt, :],
                Y_contexts=latent_obs_aux.data if latent_obs_aux is not None else latent_obs.data[:, dt:, :],
                cov_window_size=None,
                representation=None,
                run_checks=run_checks)

            # Logarithmic + Frobenious metric deformation loss
            metric_def_loss_Z = log_fro_metric_deformation_loss(Cov)
            if latent_obs_aux is not None:
                metric_def_loss_Z_prime = log_fro_metric_deformation_loss(Cov_prime)
                metric_def_loss = (metric_def_loss_Z + metric_def_loss_Z_prime) / 2.0
            else:
                metric_def_loss = metric_def_loss_Z

            metrics.update(orth_reg=metric_def_loss.mean().item())

            if self.use_spectral_score:
                # spectral_score := mean(||Cov(t, t+dt)||^2_HS/(||Cov(t)|| ||Cov(t+dt)||)) | t, dt in [1, time_horizon)
                spectral_score = vectorized_spectral_scores(CCov=CCov,
                                                            Cov=Cov,
                                                            Cov_prime=Cov_prime,
                                                            run_checks=run_checks)
                metrics.update(spectral_score=spectral_score.mean().item())
                latent_space_invariance_score = spectral_score
                #
                # # corr_scores[dt - 1] := ||Cov(t)^-1 Cov(t, t+dt) Cov(t+d)^-1||^2_HS        | dt in [1, time_horizon)
                # corr_scores = vectorized_correlation_scores(CCov=CCov,
                #                                             Cov=Cov,
                #                                             Cov_prime=Cov_prime,
                #                                             run_checks=run_checks)
                # metrics.update(corr_score=corr_scores.mean().item())
            else:
                # corr_scores[dt - 1] := ||Cov(t)^-1 Cov(t, t+dt) Cov(t+d)^-1||^2_HS        | dt in [1, time_horizon)
                corr_scores = vectorized_correlation_scores(CCov=CCov,
                                                            Cov=Cov,
                                                            Cov_prime=Cov_prime,
                                                            run_checks=run_checks)
                metrics.update(corr_score=corr_scores.mean().item())
                latent_space_invariance_score = corr_scores

            # Compute the Chapman-Kolmogorov regularization scores for all possible step transitions. In return, we get:
            # ck_regularization[i,j] = || Cov(i, j) - ( Cov(i, i+1), ... Cov(j-1, j) ) ||_2  | j >= i + 2
            # TODO: CK computation is not yet vectorized, this is a bottleneck in performance.
            # ck_regularization = chapman_kolmogorov_regularization(CCov=CCov,  # Cov=Cov, Cov_prime=Cov_prime,
            #                                                       ck_window_length=max_ck_window_length,
            #                                                       debug=debug)

            # Compute the loss
            # =============================================================================================
            alpha_orth = self.loss_weights.get('orthonormality', 1.0)

            # Max value of score is self.latent_dim
            latent_space_invariance_score = torch.mean(latent_space_invariance_score)

            # max value of orthogonal regularization is 1.0
            orth_regularization = alpha_orth * metric_def_loss.mean()  # / self.obs_state_dim

            # Add max orth_reg to make it positive.
            score = latent_space_invariance_score - orth_regularization # - (alpha_orth * self.latent_dim)
            # score = score - ck_regularization

            # Change sign to minimize the loss and maximize the score.
            loss = -score
            assert not torch.isnan(loss), f"Loss is NaN."

            metrics.update(loss=loss)

        # If model is fitted and state/latent_obs_state predictions are given, compute the reconstruction, prediction
        # and linear dynamics loss.
        pred_metrics = super().compute_loss_and_metrics(state=state,
                                                        pred_state=pred_state,
                                                        latent_obs=latent_obs,
                                                        pred_latent_obs=pred_latent_obs)

        return metrics | pred_metrics

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
