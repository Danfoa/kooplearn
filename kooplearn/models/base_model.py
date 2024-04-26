import logging
import os
import pathlib
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import kooplearn
import lightning
import numpy as np
import scipy
import torch
import torch.optim
from kooplearn._src.serialization import pickle_load, pickle_save
from kooplearn._src.utils import check_is_fitted, flatten_dict
from kooplearn.data import TensorContextDataset
from kooplearn.models.ae.utils import flatten_context_data, multi_matrix_power, unflatten_context_data
from kooplearn.utils import ModesInfo
from torch.utils.data import DataLoader

logger = logging.getLogger("kooplearn")


class LatentBaseModel(kooplearn.abc.BaseModel, torch.nn.Module, ABC):
    r"""Abstract base class for latent models of Markov processes.

    This class defines the interface for discrete-time latent models of Markov processes :math:`(\mathbf{x}_t)_{
    t\\in\\mathbb{T}}`, where :math:`\mathbf{x}_t` is the system state vector-valued observables at time :math:`t`
    and :math:`\\mathbb{T}` is the time index set. These models [...]

    [Suggestion: Since in practice/code we need to work with final-dimensional vector spaces, we will try to
    always highlight the relationship between infinite-dimensional objects (function space, operator) and its
    finite-dimensional representation/approximation (\\mathbb{R}^l`, matrix).] This will make the code more readable
    and easily modifiable.]

    The class is free to enable the practitioner to define at wish the encoding-decoding process definition, but
    assumes the the evolution of the latent state :math:`\mathbf{z} \\in \\mathcal{Z} \approx \\mathbb{R}^l` is
    modeled by a linear evolution operator :math:`T: \\mathcal{Z} \to \\mathcal{Z}` (i.e., approximated by a matrix
    of shape
    :math:`(l, l)`). Such that :math:`\mathbf{z}_{t+1} = T \\, \mathbf{z}_t`. The spectral decomposition of the
    evolution operator
    :math:`T = V \\Lambda V^*` is assumed to approximate the spectral decomposition of the process's true evolution
    operator. Therefore, the eigenvectors :math:`V` and eigenvalues :math:`\\Lambda` are the approximations of the
    eigenfunctions and eigenvalues of the true evolution operator.
    [TODO:
    Define the functional-analytical spectral decomposition notation and symbols used in the class.
        ... define code/notation conventions for the names and symbols/variable-names used in the class ()
        We will denote the:
        - name: latent state / latent observables  - symbol :math:`\mathbf{z}`    - var_name: z
        - name: encoder / observable_function     - symbol :math:`\\phi` - var_name: encoder
        - name: evolution operator                - symbol :math:`T`    - var_name: evolution_operator
        - name: decoder / observable_function     - symbol :math:`\\psi^-1` - var_name: decoder

    Define the abstract functions input types when useful/needed.
    ]
    """

    def __init__(self):
        super().__init__()  # Initialize torch.nn.Module

        # Parameters populated during fit _____________________________________________________________________________
        # Linear decoder for mode decomposition. Fitted after learning the latent representation space
        self.linear_dynamics: Union[torch.nn.Linear, None] = None
        self.lin_decoder: Union[torch.nn.Linear, None] = None
        self.state_features_shape: Union[tuple, None] = None  # Shape of the state/input features
        # Lightning parameters
        self.trainer: Union[lightning.Trainer, None] = None
        # TODO: circular reference triggers infinite search of model parameters. need to handle
        # self.lightning_model: Union[LightningLatentModel, None] = None
        # TODO: Automatically determined by looking at self.trainer.status. Handle loading from checkpoint
        self._is_fitted = False
        # _____________________________________________________________________________________________________________

    def forward(self, state_contexts: TensorContextDataset) -> any:
        r"""Forward pass of the torch.nn.Module. By default this method calls the `evolve_forward` method.

        Args:
            state_contexts (TensorContextDataset): The state to be evolved forward. This should be a trajectory of
            states
            :math:`(x_t)_{t\\in\\mathbb{T}}` in the context window :math:`\\mathbb{T}`.
        Returns:
            Any: The output of the forward pass.
        """
        return self.evolve_forward(state_contexts)

    def evolve_forward(self, state: TensorContextDataset) -> dict:
        r"""Evolves the given state forward in time using the model's encoding, evolution, and decoding processes.

        This method first encodes the given state into the latent space using the model's encoder.
        It then evolves the encoded state forward in time using the model's evolution operator.
        If a decoder is defined, it decodes the evolved latent state back into the original state space.
        The method also performs a reconstruction of the original state from the encoded latent state.

        Args:
            state (TensorContextDataset): The state to be evolved forward. This should be a trajectory of states
            :math:`(x_t)_{t\\in\\mathbb{T}}` in the context window :math:`\\mathbb{T}`.

        Returns:
            dict: A dictionary containing the following keys:  # TODO: Change to NamedTuple
                - `latent_obs`: The latent observables, :math:`z_t`. They represent the encoded state in the latent
                space.
                - `pred_latent_obs`: The predicted latent observables, :math:`\\hat{z}_t`. They represent the model's
                prediction of the latent observables.
                - `pred_state`: (Optional) The predicted state, :math:`\\hat{x}_t`. This can be None if the decoder
                is not defined.
                - `rec_state`: (Optional) The reconstructed state, :math:`\\tilde{x}_t`. It represents the state that
                the model reconstructs from the latent observables. This can be None if the decoder is not defined.
                The dictionary may also contain additional outputs from the `encode_contexts`, `decode_contexts`,
                and `evolve_contexts`.
        """
        assert hasattr(self, "encoder"), f"Model {self.__class__.__name__} does not have an `encoder` torch Module."
        # encoding/observation-function-evaluation =====================================================================
        encoder_out = self.encode_contexts(state, encoder=self.encoder)
        z_t = encoder_out if isinstance(encoder_out, TensorContextDataset) else encoder_out.pop("latent_obs")
        encoder_out = {} if isinstance(encoder_out, TensorContextDataset) else encoder_out

        # Evolution of latent observables ==============================================================================
        # Compute the approximate evolution of the latent state z̄_t for t in look-forward/prediction-horizon
        evolved_out = self.evolve_contexts(latent_obs=z_t, **encoder_out)
        pred_z_t = evolved_out if isinstance(evolved_out, TensorContextDataset) else evolved_out.pop("pred_latent_obs")
        evolved_out = {} if isinstance(evolved_out, TensorContextDataset) else evolved_out

        # (Optional) decoder/observation-function-inversion ============================================================
        # Compute the approximate evolution of the state x̄_t for t in look-forward/prediction-horizon
        pred_x_t, decoder_out = None, {}
        if hasattr(self, "decoder") and isinstance(self.decoder, torch.nn.Module):
            decoder_out = self.decode_contexts(latent_obs=pred_z_t, decoder=self.decoder, **evolved_out)
            pred_x_t = decoder_out.pop("decoded_contexts") if isinstance(decoder_out, dict) else decoder_out
            decoder_out = {} if isinstance(decoder_out, TensorContextDataset) else decoder_out

        return dict(latent_obs=z_t,
                    pred_latent_obs=pred_z_t,
                    pred_state=pred_x_t,
                    # Attached any additional outputs from encoder/decoder/evolution
                    **encoder_out,
                    **evolved_out,
                    **decoder_out)

    def encode_contexts(
            self, state: TensorContextDataset, encoder: torch.nn.Module, **kwargs
            ) -> Union[dict, TensorContextDataset]:
        r"""Encodes the given state into the latent space using the model's encoder.

        Args:
            state (TensorContextDataset): The state to be encoded. This should be a trajectory of states
             :math:`(x_t)_{t\\in\\mathbb{T}} : x_t \\in \\mathcal{X}` in the context window :math:`\\mathbb{T}`.
            encoder (torch.nn.Module): A torch module parameterizing the map between the state space
             :math:`\\mathcal{X}` and the latent space :math:`\\mathcal{Z}`.
        Returns:
            Either of the following:
            - TensorContextDataset: trajectory of encoded latent observables :math:`z_t`
            - dict: A dictionary containing the key "latent_obs" mapping to a TensorContextDataset
        """
        # From (batch, context_length, *features_shape) to (batch * context_length, *features_shape)
        flat_encoded_contexts = encoder(flatten_context_data(state))
        # From (batch * context_length, latent_dim) to (batch, context_length, latent_dim)
        latent_obs_contexts = unflatten_context_data(flat_encoded_contexts,
                                                     batch_size=len(state),
                                                     features_shape=(self.latent_dim,))
        return latent_obs_contexts

    def decode_contexts(
            self, latent_obs: TensorContextDataset, decoder: torch.nn.Module, **kwargs
            ) -> Union[dict, TensorContextDataset]:
        r"""Decodes the given latent observables back into the original state space using the model's decoder.

        Args:
            latent_obs (TensorContextDataset): The latent observables to be decoded. This should be a trajectory of
                latent observables :math:`(z_t)_{t\\in\\mathbb{T}}` in the context window :math:`\\mathbb{T}`.
            decoder (torch.nn.Module): A torch module parameterizing the map between the latent space
                :math:`\\mathcal{Z}` and the state space :math:`\\mathcal{X}`.
        Returns:
            Either of the following:
            - TensorContextDataset: trajectory of decoded states :math:`x_t`
            - dict: A dictionary containing the key "decoded_contexts" mapping to a TensorContextDataset
        """
        # From (batch, context_length, latent_dim) to (batch * context_length, latent_dim)
        flat_decoded_contexts = decoder(flatten_context_data(latent_obs))
        # From (batch * context_length, *features_shape) to (batch, context_length, *features_shape)
        decoded_contexts = unflatten_context_data(flat_decoded_contexts,
                                                  batch_size=len(latent_obs),
                                                  features_shape=self.state_features_shape)
        return decoded_contexts

    def evolve_contexts(self, latent_obs: TensorContextDataset, **kwargs) -> Union[dict, TensorContextDataset]:
        r"""Evolves the given latent observables forward in time using the model's evolution operator.

        If the model has been fitted to forecast we avoid powering the evolution operator and instead use the
        eigendecomposition of the operator to compute the evolution of the latent observables.

        Args:
            latent_obs (TensorContextDataset): The latent observables to be evolved forward. This should be a
            trajectory of latent observables
            :math:`(z_t)_{t\\in\\mathbb{T}}` in the context window :math:`\\mathbb{T}`.

        Returns:
            Either of the following:
            - TensorContextDataset: trajectory of predicted latent observables :math:`\\hat{z}_t`
            - dict: A dictionary containing the key "pred_latent_obs" mapping to a TensorContextDataset
        """
        assert latent_obs.data.ndim == 3, \
            f"Expected tensor of shape (batch, context_len, latent_dim), got {latent_obs.data.shape}"
        assert latent_obs.data.shape[2] == self.latent_dim, \
            f"Expected latent dimension {self.latent_dim}, got {latent_obs.data.shape[2]}"

        context_length = latent_obs.context_length

        # Initial condition to evolve in time.
        z_0 = latent_obs.lookback(self.lookback_len).squeeze()

        if self.is_fitted:
            if not hasattr(self, "_eigvals"): self.eig()  # Ensure eigendecomposition is in cache
            # T = V Λ V^-1 : V = eigvecs_r, Λ = eigvals, V^-1 = eigvecs_r_inv
            eigvals, eigvecs_r, eigvecs_r_inv = self._eigvals, self._eigvecs_r, self._eigvecs_r_inv
            z_0_eigbasis = torch.einsum("oi,...i->...o", eigvecs_r_inv.data, z_0.to(dtype=eigvecs_r.dtype))
            # Compute the powers of the eigenvalues used to evolve the latent state z_t | t in [0, context_length]
            # powered_eigvals: (context_length, latent_dim) -> [1, λ, λ^2, ..., λ^context_length]
            exponents = torch.arange(context_length, device=z_0.device).unsqueeze(1)
            powered_eigvals = eigvals.pow(exponents)
            # Compute z_t_eigbasis[batch, t] = Λ^t V^-1 z_0 | t in [0, context_length]
            z_t_eigbasis = torch.einsum("to,...o->...to", powered_eigvals, z_0_eigbasis)
            # Convert back to the original basis z_t[batch,t] = V Λ^t V^-1 z_0 | t in [0, context_length]
            z_t = torch.einsum("oi,...ti->...to", eigvecs_r.data, z_t_eigbasis).real.to(dtype=z_0.dtype)
        else:
            # T : (latent_dim, latent_dim)
            evolution_operator = self.evolution_operator
            # z_0: (..., latent_dim)
            # Compute the powers of the evolution operator used T_t such that z_t = T_t @ z_0 | t in [0, context_length]
            # powered_evolution_ops: (context_length, latent_dim, latent_dim) -> [I, T, T^2, ..., T^context_length]
            powered_evolution_ops = multi_matrix_power(evolution_operator, context_length)
            # Compute evolved latent observable states z_t | t in [0, context_length] (single parallel operation)
            z_t = torch.einsum("toi,...i->...to", powered_evolution_ops, z_0)

        z_pred_t = TensorContextDataset(z_t)
        return z_pred_t

    # @torch.no_grad
    def eig(self,
            eval_left_on: Optional[TensorContextDataset] = None,
            eval_right_on: Optional[TensorContextDataset] = None,
            ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Returns the eigenvalues of the Koopman/Transfer operator and optionally evaluates left and right
        eigenfunctions.

        TODO: Should make everything default to backend torch. As the entire model is on torch, independent on device
        TODO: Should improve the documentation of the method.

        Args:
            eval_left_on (TensorContextDataset or None): Dataset of context windows on which the left eigenfunctions
            are evaluated.
            eval_right_on (TensorContextDataset or None): Dataset of context windows on which the right
            eigenfunctions are evaluated.

        Returns:
            Eigenvalues of the Koopman/Transfer operator, shape ``(rank,)``. If ``eval_left_on`` or ``eval_right_on``
             are not ``None``, returns the left/right eigenfunctions evaluated at ``eval_left_on``/``eval_right_on``:
             shape ``(n_samples, rank)``.
        """
        if not hasattr(self, "_eigvals"):
            T = self.evolution_operator
            T_np = T.detach().cpu().numpy()
            # T is a square real-valued matrix.
            eigvals, eigvecs_l, eigvecs_r = scipy.linalg.eig(T_np, left=True, right=True)
            eigvecs_r_inv = np.linalg.inv(eigvecs_r)
            eigvecs_l_inv = np.linalg.inv(eigvecs_l)

            # Left and right eigenvectors are stored in columns: eigvecs_l/r[:, i] is the i-th left/right eigenvector
            # T @ eigvecs_r[:, i] = eigvals[i] @ eigvecs_r[:, i] <==>  T = eigvecs_r @ eigvals[i] @ eigvecs_r^-1
            # assert np.allclose(eigvecs_r @ np.diag(eigvals) @ eigvecs_r.conj().T, T_np, rtol=1e-5, atol=1e-5)

            # Store as torch parameters for lighting to manage device automatically. And forecast avoiding matrix power
            self._eigvals = torch.nn.Parameter(torch.tensor(eigvals, device=T.device), requires_grad=False)
            self._eigvecs_l = torch.nn.Parameter(torch.tensor(eigvecs_l, device=T.device), requires_grad=False)
            self._eigvecs_r = torch.nn.Parameter(torch.tensor(eigvecs_r, device=T.device), requires_grad=False)
            self._eigvecs_r_inv = torch.nn.Parameter(torch.tensor(eigvecs_r_inv, device=T.device), requires_grad=False)
            self._eigvecs_l_inv = torch.nn.Parameter(torch.tensor(eigvecs_l_inv, device=T.device), requires_grad=False)

        eigvals, eigvecs_l, eigvecs_r = self._eigvals, self._eigvecs_l, self._eigvecs_r
        eigvecs_r_inv, eigvecs_l_inv = self._eigvecs_r_inv, self._eigvecs_l_inv

        left_eigfn, right_eigfn = None, None
        if eval_right_on is not None:
            # Ensure data is on the same device as the model
            eval_right_on.to(device=self.evolution_operator.device, dtype=self.evolution_operator.dtype)
            # Compute the latent observables for the data (batch, context_len, latent_dim)
            z_t = self.encode_contexts(state=eval_right_on, encoder=self.encoder)
            # Evaluation of eigenfunctions in (batch/n_samples, context_len, latent_dim)
            # TODO: This is the batch form equivalent of primal.evaluate_eigenfunction with the additional feature of
            #  performing the operation on the selected device (cpu or gpu)
            right_eigfn = torch.einsum(
                "...il,...l->...i", eigvecs_l_inv.data, z_t.data.to(dtype=eigvecs_l_inv.dtype)
                )

        if eval_left_on is not None:
            # Ensure data is on the same device as the model
            eval_left_on.to(device=self.evolution_operator.device, dtype=self.evolution_operator.dtype)
            # Compute the latent observables for the data (batch, context_len, latent_dim)
            z_t = self.encode_contexts(state=eval_left_on, encoder=self.encoder)
            # Evaluation of eigenfunctions in parallel batch form (..., context_len, latent_dim)
            # left_eigfn[...,t, i] = <v_i, z_t>_C : i=1,...,l
            # TODO: This is the batch form equivalent of primal.evaluate_eigenfunction with the additional feature of
            #  performing the operation on the selected device (cpu or gpu)
            left_eigfn = torch.einsum(
                "...il,...l->...i", eigvecs_r_inv.data, z_t.data.to(dtype=eigvecs_r_inv.dtype)
                )

        # TODO: Does it make sense for DeepLearning based models to return as numpy? ...
        eigvals = eigvals.detach().cpu().numpy()
        left_eigfn = left_eigfn.detach().cpu().numpy() if left_eigfn is not None else None
        right_eigfn = right_eigfn.detach().cpu().numpy() if right_eigfn is not None else None

        if eval_left_on is None and eval_right_on is None:
            return eigvals
        elif eval_left_on is None and eval_right_on is not None:
            return eigvals, right_eigfn
        elif eval_left_on is not None and eval_right_on is None:
            return eigvals, left_eigfn
        else:
            return eigvals, left_eigfn, right_eigfn

    def modes(self,
              state: TensorContextDataset,
              predict_observables: bool = False,
              ):
        r"""Compute the mode decomposition of the state and/or observables

        Informally, if :math:`(\\lambda_i, \\xi_i, \\psi_i)_{i = 1}^{r}` are eigentriplets of the Koopman/Transfer
        operator, for any observable :math:`f` the i-th mode of :math:`f` at :math:`x` is defined as:
        :math:`\\lambda_i \\langle \\xi_i, f \\rangle \\psi_i(x)`. See :footcite:t:`Kostic2022` for more details.

        When applying mode decomposition to the latent observable states :math:`\mathbf{z}_t \in \mathbb{R}^l`,
        the modes are obtained using the eigenvalues and eigenvectors of the solution operator :math:`T`, as there is no
        need to approximate the inner product :math:`\\langle \\xi_i, f \\rangle` using a kernel/Data matrix.
        Consider that the evolution of the latent observable is given by:

        .. math::
        \mathbf{z}_{t+k} = T^k \mathbf{z}_t = (V \Lambda^k V^H) \mathbf{z}_t = \sum_{i=1}^{l} \lambda_i^k \langle
        v_i, \mathbf{z}_t \rangle v_i


        dd
        Note: when we compute the modes of the latent observable state, the mode decomposition reduces to a traditional
        mode decomposition using the eigenvectors and eigenvalues of the solution operator :math:`T`

        """

        assert self.is_fitted, \
            f"Instance of {self.__class__.__name__} is not fitted. Please call the `fit` method before calling `modes`"
        if predict_observables:
            raise NotImplementedError("Need to implement the approximation of the outer product / inner products "
                                      "between each dimension/function of the latent space and observables provided")

        # Left eigenfunctions are the inner products between the latent observables z_t and the right eigenvectors
        # `v_i` of the evolution operator T @ v_i = λ_i v_i. wWhere <v_i, z_t> is the value of eigenfunction i.
        # That is: z_t_eig := V^-1 z_t = [<v_i, z_t>, ..., ]_{i=1,...,l}
        eigvals, z_t_eig = self.eig(eval_left_on=state)

        return ModesInfo(dt=1.0,
                         eigvals=eigvals,
                         eigvecs_r=self._eigvecs_r.detach().cpu().numpy(),
                         state_eigenbasis=z_t_eig)

    def predict(
            self,
            data: TensorContextDataset,
            t: int = 1,
            predict_observables: bool = True,
            reencode_every: int = 0,
            ):
        r"""Predicts the state or, if the system is stochastic, its expected value :math:`\mathbb{E}[X_t | X_0 = X]`
        after ``t`` instants given the initial conditions ``data.lookback(self.lookback_len)`` being the lookback
        slice of ``data``.
        If ``data.observables`` is not ``None``, returns the analogue quantity for the observable instead.

        Args:
            data (TensorContextDataset): Dataset of context windows. The lookback window of ``data`` will be used as
            the initial condition, see the note above.
            t (int): Number of steps in the future to predict (returns the last one).
            predict_observables (bool): Return the prediction for the observables in ``data.observables``,
            if present. Defaults to ``True``.
            reencode_every (int): When ``t > 1``, periodically reencode the predictions as described in
            :footcite:t:`Fathi2023`. Only available when ``predict_observables = False``.

        Returns:
           The predicted (expected) state/observable at time :math:`t`. The result is composed of arrays with shape
           matching ``data.lookforward(self.lookback_len)`` or the contents of ``data.observables``. If
           ``predict_observables = True`` and ``data.observables != None``, the returned ``dict``will contain the
           special key ``__state__`` containing the prediction for the state as well.
        """
        # TODO: Requires update
        raise NotImplementedError("This method is not updated yet.")
        check_is_fitted(self, ["_state_trail_dims"])
        assert tuple(data.shape[2:]) == self._state_trail_dims

        data = self._to_torch(data)
        if predict_observables and hasattr(data, "observables"):
            observables = data.observables
            observables["__state__"] = None
        else:
            observables = {"__state__": None}

        results = {}
        for obs_name, obs in observables.items():
            if (reencode_every > 0) and (t > reencode_every):
                if (predict_observables is True) and (observables is not None):
                    raise ValueError(
                        "rencode_every only works when forecasting states, not observables. Consider setting "
                        "predict_observables to False."
                        )
                else:
                    num_reencodings = floor(t / reencode_every)
                    for k in range(num_reencodings):
                        raise NotImplementedError
            else:
                with torch.no_grad():
                    evolved_data = evolve_forward(
                        data,
                        self.lookback_len,
                        t,
                        self.lightning_module.encoder,
                        self.lightning_module.decoder,
                        self.lightning_module.evolution_operator,
                        )
                    evolved_data = evolved_data.data.detach().cpu().numpy()
                    if obs is None:
                        results[obs_name] = evolved_data
                    elif callable(obs):
                        results[obs_name] = obs(evolved_data)
                    else:
                        raise ValueError(
                            "Observables must be either None, or callable."
                            )

        if len(results) == 1:
            return results["__state__"]
        else:
            return results

    @abstractmethod
    def compute_loss_and_metrics(self,
                                 state: Optional[TensorContextDataset] = None,
                                 pred_state: Optional[TensorContextDataset] = None,
                                 latent_obs: Optional[TensorContextDataset] = None,
                                 pred_latent_obs: Optional[TensorContextDataset] = None,
                                 **kwargs
                                 ) -> dict[str, torch.Tensor]:
        r"""Compute the loss and metrics of the model.

        The implementation of this method for the abstract LatentModel class computes only the reconstruction and
        prediction loss TODO: Add more details

        Args:
            state: trajectory of states :math:`(x_t)_{t\\in\\mathbb{T}}` in the context window
            :math:`\\mathbb{T}`.
            pred_state: predicted trajectory of states :math:`(\\hat{x}_t)_{t\\in\\mathbb{T}}`
            latent_obs: trajectory of latent observables :math:`(z_t)_{t\\in\\mathbb{T}}` in the context window
            :math:`\\mathbb{T}`.
            pred_latent_obs: predicted trajectory of latent observables :math:`(\\hat{z}_t)_{t\\in\\mathbb{T}}`
            **kwargs:

        Returns:
            Dictionary containing the key "loss" and other metrics to log.
        """
        metrics = {}
        lookback_len = self.lookback_len

        MSE = torch.nn.MSELoss()

        if state is not None and pred_state is not None:
            # Reconstruction + prediction loss
            rec_loss = MSE(state.lookback(lookback_len), pred_state.lookback(lookback_len))
            pred_loss = MSE(state.lookforward(lookback_len), pred_state.lookforward(lookback_len))
            metrics.update(reconstruction_loss=rec_loss.item(), prediction_loss=pred_loss.item())

        if latent_obs is not None and pred_latent_obs is not None:
            # Latent space prediction loss
            latent_obs_pred_loss = MSE(latent_obs.lookforward(lookback_len), pred_latent_obs.lookforward(lookback_len))
            metrics.update(linear_dynamics_loss=latent_obs_pred_loss.item())

        return metrics

    @property
    def evolution_operator(self) -> torch.Tensor:
        raise NotImplementedError()

    def save(self, filename: os.PathLike):
        """Serialize the model to a file.

        Args:
            filename (path-like or file-like): Save the model to file.
        """
        # self.lightning_module._kooplearn_model_weakref = None  ... Why not simply use self reference?
        pickle_save(self, filename)

    @classmethod
    def load(cls, path: os.PathLike) -> 'LatentBaseModel':
        """Load a serialized model from a file.

        Args:
            filename (path-like or file-like): Load the model from file.

        Returns:
            Saved instance of `LatentBaseModel`.
        """
        restored_obj = pickle_load(cls, path)
        # Restore the weakref # TODO Why?
        # restored_obj.lightning_module._kooplearn_model_weakref = weakref.ref(
        #     restored_obj
        #     )
        return restored_obj

    @torch.no_grad()
    def fit_linear_decoder(self, latent_states: torch.Tensor, states: torch.Tensor) -> torch.nn.Module:
        """Fit a linear decoder mapping the latent state space Z to the state space X. use for mode decomp."""

        logger.info(f"Fitting linear decoder for {self.__class__.__name__} model")
        use_bias = False  # TODO: Unsure if to enable. This can be another hyperparameter, or set to true by default.

        # Solve the least squares problem to find the linear decoder matrix and bias
        from kooplearn._src.linalg import full_rank_lstsq
        D, bias = full_rank_lstsq(X=latent_states, Y=states, bias=use_bias)

        # Check shape
        _expected_shape = (np.prod(self.state_features_shape), self.latent_dim)
        assert D.shape == _expected_shape, \
            f"Expected linear decoder shape {_expected_shape}, got {D.shape}"

        # Create a non-trainable linear layer to store the linear decoder matrix and bias term
        lin_decoder = torch.nn.Linear(in_features=self.latent_dim,
                                      out_features=np.prod(self.state_features_shape),
                                      bias=use_bias)
        lin_decoder.weight.data = D
        lin_decoder.weight.requires_grad = False

        if use_bias:
            lin_decoder.bias.data = torch.tensor(bias, dtype=torch.float32)
            lin_decoder.bias.requires_grad = False

        return lin_decoder

    @torch.no_grad()
    def _predict_training_data(self,
                               trainer: lightning.Trainer,
                               lightning_module: 'LightningLatentModel',
                               train_dataloader: DataLoader,
                               ckpt_path: Optional[pathlib.Path] = None) -> tuple[TensorContextDataset]:
        """ Obtain the samples of states X and latent observables Z for all training dataset.

        Note that the `LightningLatentModel.predict_step` returns the latent observables and the input states. Both are
        needed as the train dataloader could be configured to perform shuffling/augmentation or other transformations
        to the input data before feeding it to the encoder.

        As Lightning automatically sends outputs to CPU, so we don't need to care about GPU memory.

        Args:
            trainer:
            lightning_module:
            train_dataloader:
            ckpt_path:
        Returns:
            X: TensorContextDataset of states X of shape (n_samples, context_len, **state_feature_shape)
            Z: TensorContextDataset of latent observables Z of shape (n_samples, context_len, latent_dim)
        """

        predict_out = trainer.predict(model=lightning_module,
                                      dataloaders=train_dataloader,
                                      ckpt_path=ckpt_path if (ckpt_path and ckpt_path.exists()) else None)

        # This structure of the output is enforced by the predict_step method of the LightningLatentModel
        Z = [out_batch['latent_obs'] for out_batch in predict_out]
        Z = TensorContextDataset(torch.cat([z.data for z in Z], dim=0))  # (n_samples, context_len, latent_dim)
        X = [out_batch['state'] for out_batch in predict_out]
        X = TensorContextDataset(torch.cat([x.data for x in X], dim=0))  # (n_samples, context_len, state_dim)

        # Check the shapes
        assert Z.shape[-1] == self.latent_dim, f"Expected latent_dim {self.latent_dim}, got {Z.shape[-1]}"
        assert X.shape[2:] == self.state_features_shape, \
            f"Expected state_features_shape {self.state_features_shape}, got {X.shape[2:]}"

        return X, Z

    def _dry_run(self, state: TensorContextDataset):
        class_name = self.__class__.__name__

        assert self.state_features_shape is not None, f"state_features_shape not identified for {class_name}"
        x_t = state

        model_out = self.evolve_forward(x_t)
        try:
            z_t = model_out.pop("latent_obs")
            pred_z_t = model_out.pop("pred_latent_obs")
            pred_x_t = model_out.pop("pred_state")
        except KeyError as e:
            raise KeyError(f"Missing output of {class_name}.evolve_forward") from e

        # Check latent observable contexts shapes
        assert z_t.shape == pred_z_t.shape, \
            f"Encoded latent context shape {z_t.shape} different from input shape {pred_z_t.shape}"
        if pred_x_t is not None:
            assert pred_x_t.shape == x_t.shape, \
                f"Evolved latent context shape {pred_x_t.shape} different from input shape {x_t.shape}"

    def _check_dataloaders_and_shapes(self, datamodule, train_dataloaders, val_dataloaders):
        """Check dataloaders and the shape of the first batch to determine state features shape."""
        # TODO: Should we add a features_shape attribute to the Context class, and avoid all this?

        # If you supply a datamodule you can't supply train_dataloader or val_dataloaders
        # TODO: Lightning has checks this before each train starts. Is this necessary?
        # if (train_dataloaders is not None or val_dataloaders is not None) and datamodule is not None:
        #     raise ValueError(
        #         f"You cannot pass `train_dataloader` or `val_dataloaders` to `{self.__class__.__name__}.fit(
        #         datamodule=...)`")

        # Get the shape of the first batch to determine the lookback_len
        # TODO: lookback_len is model-fixed and all this process is unnecessary if we know the `state_obs_shape`
        if train_dataloaders is None:
            assert isinstance(datamodule, lightning.LightningDataModule)
            for batch in datamodule.train_dataloader():
                assert isinstance(batch, TensorContextDataset)
                with torch.no_grad():
                    self.state_features_shape = tuple(batch.shape[2:])
                    self._dry_run(batch)
                break
        else:
            assert isinstance(train_dataloaders, torch.utils.data.DataLoader)
            for batch in train_dataloaders:
                assert isinstance(batch, TensorContextDataset)
                with torch.no_grad():
                    self.state_features_shape = tuple(batch.shape[2:])
                    self._dry_run(batch)
                break
        # Get the shape of the first batch to determine the lookback_len and state features shape
        if train_dataloaders is None:
            assert isinstance(datamodule, lightning.LightningDataModule)
            for batch in datamodule.train_dataloader():
                assert isinstance(batch, TensorContextDataset)
                with torch.no_grad():
                    self._state_observables_shape = tuple(batch.shape[2:])
                    self._dry_run(batch)
                break
        else:
            assert isinstance(train_dataloaders, torch.utils.data.DataLoader)
            for batch in train_dataloaders:
                assert isinstance(batch, TensorContextDataset)
                with torch.no_grad():
                    self._state_observables_shape = tuple(batch.shape[2:])
                    self._dry_run(batch)
                break


class LightningLatentModel(lightning.LightningModule):
    """Base `LightningModule` class to define the common codes for training instances of `LatentBaseModels`.

    For most Latent Models, this class should suffice to train the both AE based and DPNet's Latent Models.
    User should inherit this class in case he/she wants to modify some of the lighting hooks/callbacks, logging or the
    basic  generic pipeline defined in this class.

    DAE, and DPNets models should be trained by this same class instance. So the class should cover the common pipeline
    between Autoencoder based models and representation-learning-then-operator-regression based models.
    """

    def __init__(self,
                 latent_model: LatentBaseModel,
                 optimizer_fn: type[torch.optim.Optimizer] = torch.optim.Adam,
                 optimizer_kwargs: Optional[dict] = None,
                 ):
        super(LightningLatentModel, self).__init__()
        self.latent_model: LatentBaseModel = latent_model
        self._optimizer_fn = optimizer_fn
        self._optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        # TODO: Deal with with latent_model hparams if needed.

    def forward(self, state_contexts: TensorContextDataset) -> Any:
        out = self.latent_model.forward(state_contexts)
        return out

    def training_step(self, train_contexts: TensorContextDataset, batch_idx):
        model_out = self(train_contexts)
        out = self.latent_model.compute_loss_and_metrics(state=train_contexts, **model_out)
        assert "loss" in out, f"Loss not found within {out.keys()}"
        loss = out["loss"]
        self.log_metrics(out, suffix="train", batch_size=len(train_contexts))
        return loss

    def validation_step(self, val_contexts: TensorContextDataset, batch_idx):
        model_out = self(val_contexts)
        out = self.latent_model.compute_loss_and_metrics(state=val_contexts, **model_out)
        assert "loss" in out, f"Loss not found within {out.keys()}"
        loss = out["loss"]
        self.log_metrics(out, suffix="val", batch_size=len(val_contexts))
        return loss

    def test_step(self, test_contexts: TensorContextDataset, batch_idx):
        model_out = self(test_contexts)
        out = self.latent_model.compute_loss_and_metrics(state=test_contexts, **model_out)
        assert "loss" in out, f"Loss not found within {out.keys()}"
        loss = out["loss"]
        self.log_metrics(out, suffix="test", batch_size=len(test_contexts))
        return loss

    def predict_step(self, batch, batch_idx, **kwargs):
        with torch.no_grad():
            return self(batch) | dict(state=batch)

    def log_metrics(self, metrics: dict, suffix='', batch_size=None):
        flat_metrics = flatten_dict(metrics)
        for k, v in flat_metrics.items():
            name = f"{k}/{suffix}"
            self.log(name, v, prog_bar=(k == "loss"), batch_size=batch_size)

    def on_train_epoch_start(self) -> None:
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self) -> None:
        self.log('time_per_epoch', time.time() - self._epoch_start_time, prog_bar=False, on_epoch=True)

    def configure_optimizers(self) -> Any:
        if "lr" in self._optimizer_kwargs:
            self.lr = self._optimizer_kwargs["lr"]
        else:
            self.lr = 1e-3
            self._optimizer_kwargs["lr"] = self.lr
            _class_name = self.__class__.__name__
            logger.warning(
                f"Using default learning rate value lr=1e-3 for {self.__class__.__name__}. "
                f"You can specify the learning rate by passing it to the optimizer_kwargs initialization argument.")
        return self._optimizer_fn(self.parameters(), **self._optimizer_kwargs)

    # @deprecated("ContextWindow Tensors should implement the .to method, making this unnecessary.")
    # def transfer_batch_to_device(self, batch, device, dataloader_idx):
    #     batch.data = batch.data.to(device)
    #     return batch


def _default_lighting_trainer() -> lightning.Trainer:
    return lightning.Trainer(accelerator='cuda' if torch.cuda.is_available() else 'cpu',
                             devices='auto',
                             logger=None,
                             log_every_n_steps=1,
                             max_epochs=100,
                             enable_progress_bar=True)
