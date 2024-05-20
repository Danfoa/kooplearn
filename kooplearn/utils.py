import pathlib
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
from lightning.pytorch.callbacks import ModelCheckpoint
from plotly.graph_objs import Figure
import logging

logger = logging.getLogger(__name__)

def check_if_resume_experiment(ckpt_call: Optional[ModelCheckpoint] = None):
    """ Checks if an experiment should be resumed based on the existence of checkpoint files.

    This function checks if the last checkpoint file and the best checkpoint file exist.
    If the best checkpoint file exists and the last checkpoint file does not, the experiment was terminated.
    If both files exist, the experiment was not terminated.

    The names of the best and last checkpoint files are automatically queried from the ModelCheckpoint instance. Thus,
    if user defines specific names for the checkpoint files, the function will still work as expected.

    Args:
        ckpt_call (Optional[ModelCheckpoint]): The ModelCheckpoint callback instance used in the experiment.
            If None, the function will return False, None, None.

    Returns:
        tuple: A tuple containing three elements:
            - terminated (bool): True if the experiment was terminated, False otherwise.
            - ckpt_path (pathlib.Path): The path to the last checkpoint file.
            - best_path (pathlib.Path): The path to the best checkpoint file.
    """
    if ckpt_call is None:
        return False, None, None

    ckpt_path = pathlib.Path(ckpt_call.dirpath).joinpath(ckpt_call.CHECKPOINT_NAME_LAST + ckpt_call.FILE_EXTENSION)
    best_path = pathlib.Path(ckpt_call.dirpath).joinpath(ckpt_call.filename + ckpt_call.FILE_EXTENSION)

    if best_path.exists():
        return True, ckpt_path, best_path
    elif ckpt_path.exists():
        return False, ckpt_path, best_path
    else:
        return False, ckpt_path, best_path


@dataclass
class ModesInfo:
    """
    Data structure to store the data and utility functions used for dynamic mode decomposition.

    By default this data structure assumes that the state variables `z_k ∈ R^l` are real-valued. Thus the returned modes
    will not contain both complex-conjugate mode instances, but rather only their resultant real parts.

    This class takes an eigenvalue λ_i and uses its polar representation to compute relevant information of the
    eigenspace dynamics. That is for λ_i:= r_i * exp(j*θ_i), the r_i is the modulus and θ_i is the angle. We can then
    compute the frequency of oscillation and the decay rate of the mode dynamics z_k+1^i = r_i * exp(j*θ_i) * z_k^i.

    - Frequency: f_i = θ_i / (2π * dt)
    - Decay Rate: δ_i = ln(r_i) / dt


    Attributes:
        dt (float): Time step between discrete frames z_k and z_k+1, such that in continuous time z_k := z_(k*dt)
        eigvals (np.ndarray): Vector of N eigenvalues, λ_i ∈ C : i ∈ [1,...,N], of the evolution operator T. Shape: (N,)
        eigvecs_r (np.ndarray): Right eigenvectors stacked in column form in a matrix of shape (l, N), where each
        eigenvector, v_i : v_i ∈ C^l, i ∈ [1,...,N], of the evolution operator: T v_i = λ_i v_i.
        state_eigenbasis (np.ndarray): Vector of inner products <v_i, z_k> ∈ C between the state/latent state and the
        mode eigenvectors. Shape: (..,context_window, N)
          These values represent the scale and angle of state in each of the N eigenspaces of T.
        linear_decoder (Optional[np.ndarray]): Linear decoder matrix to project the modes from the latent space to
        the observation space. Shape: (l, o)
    """
    dt: float
    eigvals: np.ndarray
    eigvecs_r: np.ndarray
    state_eigenbasis: np.ndarray
    modes_group: Optional[list[str]] = None
    linear_decoder: Optional[Union[np.ndarray, any]] = None
    sort_metric: str = "modulus"

    plotly_template: str = "plotly_dark"

    def __post_init__(self):
        """Identifies real and complex conjugate pairs of eigenvectors, along with their associated dimensions

        Assuming `z_k ∈ R^l` is real-valued, we will not obtain l modes, considering that for any eigenvector v_i
        associated with a complex eigenvalue λ_i ∈ C will have corresponding conjugate eigenpair (v_i^*, λ_i^*).
        Sort and cluster the eigenvalues by the chosen sorting metric.
        """
        # Sort and cluster the eigenvalues by magnitude and field (real, complex) ======================================
        from kooplearn._src.utils import parse_cplx_eig
        # Keep memory of the original number of modes/eigenvalues
        self._n_original_modes = len(self.eigvals)

        # Snap eigvals close to the init circle to be in the unit circle
        self.is_mode_marginally_stable = np.abs(np.abs(self.eigvals) - 1) < 1e-3
        self.eigvals[self.is_mode_marginally_stable] = (self.eigvals[self.is_mode_marginally_stable] /
                                                        np.abs(self.eigvals[self.is_mode_marginally_stable]))
        # Snap unstable eigvals to the unit circle
        is_mode_unstable = np.abs(self.eigvals) > 1.0
        self.eigvals[is_mode_unstable] /= np.abs(self.eigvals[is_mode_unstable])
        if np.any(is_mode_unstable):
            logger.warning(f"Unstable eigenvalues were snapped to the unit circle: {np.argwhere(is_mode_unstable)}")

        real_eigs, cplx_eigs, real_eigs_indices, cplx_eigs_indices = parse_cplx_eig(self.eigvals)

        if self.sort_metric == "modulus":
            real_eigs_modulus = np.abs(real_eigs)
            cplx_eigs_modulus = np.abs(cplx_eigs)
            eigs_sort_metric = np.concatenate((real_eigs_modulus, cplx_eigs_modulus))
            eigs_indices = np.concatenate((real_eigs_indices, cplx_eigs_indices))
            # Sort the eigenvalues by modulus |λ_i| in descending order.
            sorted_indices = np.flip(np.argsort(eigs_sort_metric))
        elif self.sort_metric == "modulus-amplitude":
            real_eigs_modulus = np.abs(real_eigs)
            cplx_eigs_modulus = np.abs(cplx_eigs)
            real_eigs_amplitude = np.abs(self.state_eigenbasis[0, 0, real_eigs_indices])
            cplx_eigs_amplitude = np.abs(self.state_eigenbasis[0, 0, cplx_eigs_indices])
            eigs_sort_metric = np.concatenate((real_eigs_modulus * real_eigs_amplitude,
                                               2 * cplx_eigs_modulus * cplx_eigs_amplitude))
            eigs_indices = np.concatenate((real_eigs_indices, cplx_eigs_indices))
            # Sort the eigenvalues by the product of modulus |λ_i| and amplitude |<v_i, z_k>| in descending order.
            sorted_indices = np.flip(np.argsort(eigs_sort_metric))
        elif self.sort_metric == "freq":
            # Sort real-eigvals by modulus (decreasing), and complex eigvals by frequency (increasing)
            freqs_cplx = np.angle(cplx_eigs) / (2 * np.pi * self.dt)
            sorted_cplx_indices = cplx_eigs_indices[np.argsort(freqs_cplx)]
            sorted_real_indices = np.flip(real_eigs_indices[np.argsort(np.abs(real_eigs))])
            eigs_indices = np.arange(len(self.eigvals))
            sorted_indices = np.concatenate((sorted_real_indices, sorted_cplx_indices))
        else:
            raise NotImplementedError(f"Mode sorting by {self.sort_metric} is not implemented yet.")

        # Store the sorted indices of eigenspaces by modulus, and the indices of real and complex eigenspaces.
        # Such that self.eigvals[self._sorted_eigs_indices] returns the sorted eigenvalues.
        self._sorted_eigs_indices = eigs_indices[sorted_indices]
        # Check resultant number of modes is equivalent to n_real_eigvals + 1/2 n_complex_eigvals
        assert len(self._sorted_eigs_indices) == len(cplx_eigs) + len(real_eigs)

        # Modify the eigenvalues, eigenvectors, and state_eigenbasis to be sorted, and ignore complex conjugates. =====
        self.eigvals = self.eigvals[self._sorted_eigs_indices]
        self.eigvecs_r = self.eigvecs_r[:, self._sorted_eigs_indices]
        self.state_eigenbasis = self.state_eigenbasis[..., self._sorted_eigs_indices]
        self.is_mode_marginally_stable = self.is_mode_marginally_stable[self._sorted_eigs_indices]
        # Utility array to identify if in the new order the modes/eigvals are to be treated as complex or real
        self.is_complex_mode = [idx in cplx_eigs_indices for idx in self._sorted_eigs_indices]

        # If the input state_eigenbasis is a trajectory of states in time of shape (..., context_window, l),
        # we compute the predicted state_eigenbasis by applying the linear dynamics of each eigenspace to the
        # initial state_eigenbasis a.k.a the eigenfunctions evaluated at time 0.
        context_window = self.state_eigenbasis.shape[-2]
        eigfn_0 = self.state_eigenbasis[..., 0, :]
        eigval_t = np.asarray([self.eigvals ** t for t in range(context_window)])  # λ_i^t for t in [0,time_horizon)
        eigfn_pred = np.einsum("...l,tl->...tl", eigfn_0, eigval_t)  # (...,context_window, l)
        assert self.state_eigenbasis.shape == eigfn_pred.shape
        self.pred_state_eigenbasis = eigfn_pred

        # Compute the real-valued modes associated with the values in `state_eigenbasis` ===============================
        # Change from the spectral/eigen-basis of the evolution operator to its original basis obtaining a tensor of
        # This process will generate N_u complex-valued mode vectors z_k^(i) ∈ C^l, where
        # N_u = n_real_eigvals + 1/2 n_complex_eigvals. The shape cplx_modes: (..., N_u, l)
        self.cplx_modes = np.einsum("...le,...e->...el", self.eigvecs_r, self.state_eigenbasis)
        self.cplx_modes_pred = np.einsum("...le,...e->...el", self.eigvecs_r, self.pred_state_eigenbasis)
        if len(real_eigs) > 0:  # Check real modes have zero imaginary part
            _real_eigval_modes = self.cplx_modes[..., np.logical_not(self.is_complex_mode), :]
            assert np.allclose(_real_eigval_modes.imag, 0, rtol=1e-5, atol=1e-5), \
                f"Real modes have non-zero imaginary part: {np.max(_real_eigval_modes.imag)}"

        # Store all values in a dataframe useful for plotting and visualization =======================================
        self._data_df = pd.DataFrame({
            "modes_sorted_idx":     range(self.n_modes),
            "modes_original_idx":   self._sorted_eigs_indices,
            "modes_frequencies":    self.modes_frequency,
            "modes_modulus":        self.modes_modulus,
            "modes_amplitude":      self.modes_amplitude,
            "modulus-amplitude":    self.modes_modulus * self.modes_amplitude,
            "modes_decay_rate":     self.modes_decay_rate,
            "modes_transient_time": self.modes_transient_time,
            "eigvals_re":           self.eigvals.real,
            "eigvals_im":           self.eigvals.imag,
            "is_complex_mode":      self.is_complex_mode,
            "is_marginally_stable": ["stable" if s else "transient" for s in self.is_mode_marginally_stable],
            })

        self._modes_vs_time_fig = None
        self._eigval_metrics_fig = None

    def sort_by(self, sort_metric: str):
        self.sort_metric = sort_metric
        self.__post_init__()

    @property
    def n_modes(self):
        return len(self.eigvals)

    @property
    def n_dc_modes(self):
        return np.sum(np.logical_not(self.is_complex_mode))

    @property
    def modes(self):
        """ Compute the real-valued modes of the system.

        Each mode associated with a complex eigenvalue will be scaled to twice its real part, considering the conjugate
        pair v_i^* λ_i * <u_i,z_t> + v_i^* λ_i^* * <u_i^*,z_t>. This process will generate N_u complex-valued mode
        vectors z_k^(i) ∈ C^l, where N_u = n_real_eigvals + 1/2 n_complex_eigvals <= l.

        Returns:
            modes (np.ndarray): Array of shape (..., N_u, s) of real modes of the system, computed from the input
            `self.state_eigenbasis` of shape (..., l). Where s=l if `linear_decoder` is None, otherwise s=o.
            The modes are sorted by the selected metric in `sort_by`.
        """
        real_modes = self.cplx_modes_pred.real
        real_modes[..., self.is_complex_mode, :] *= 2
        if self.linear_decoder is not None:
            import escnn
            import torch
            if isinstance(self.linear_decoder, torch.nn.Linear):
                device, dtype = self.linear_decoder.weight.device, self.linear_decoder.weight.dtype
                real_modes = self.linear_decoder(
                    torch.tensor(real_modes, device=device, dtype=dtype)
                    ).detach().cpu().numpy()
            elif isinstance(self.linear_decoder, escnn.nn.Linear):  # in case of equivariant linear decoder
                matrix, _ = self.linear_decoder.expand_parameters()
                device, dtype, in_type = matrix.device, matrix.dtype, self.linear_decoder.in_type
                initial_shape = real_modes.shape  # (n_trajs, context_window, N_u, l)
                output_shape = initial_shape[:-1] + (self.linear_decoder.out_type.size,)
                # Use contiguous to avoid shuffling dims on reshape
                modes = torch.tensor(real_modes, device=device, dtype=dtype).contiguous()
                # Equiv linear layer is expecting a GeometricTensor instance of shape (n_samples, l)
                modes_flat = torch.reshape(modes, (-1, initial_shape[-1]))
                modes_flat_typed = in_type(modes_flat)
                real_modes_typed = self.linear_decoder(modes_flat_typed)
                # Reshape to (n_trajs, context_window, N_u, o)
                real_modes_orig_shape = torch.reshape(real_modes_typed.tensor, output_shape)
                real_modes = real_modes_orig_shape.detach().cpu().numpy()
            elif isinstance(self.linear_decoder, np.ndarray):
                real_modes = np.einsum("ol,...l->...o", self.linear_decoder, real_modes)
            else:
                raise ValueError("linear_decoder must be `None`, `torch.nn.Linear` or `np.ndarray`.")

        return real_modes

    @property
    def modes_modulus(self):
        return np.abs(self.eigvals) ** (1 / self.dt)

    @property
    def modes_frequency(self):
        """Compute the frequency of oscilation of each mode's eigenspace dynamics

        Returns:
            np.ndarray: Array of frequencies in Hz of each mode's eigenspace dynamics. Shape: (N,)
        """
        angles = np.angle(self.eigvals)
        freqs = angles / (2 * np.pi * self.dt)
        return freqs

    @property
    def modes_amplitude(self):
        # Compute <u_i, z_k>.
        mode_amplitude = np.abs(self.state_eigenbasis)[0, 0, ...]  # Return projections of the initial state
        mode_amplitude[..., self.is_complex_mode] *= 2
        return mode_amplitude

    @property
    def modes_phase(self):
        mode_phase = np.angle(self.state_eigenbasis)
        # set phase of real modes to zero
        mode_phase[..., np.logical_not(self.is_complex_mode)] = 0
        return mode_phase

    @property
    def modes_decay_rate(self):
        """Compute the decay rate of each mode's eigenspace dynamics

        If the decay rate is positive, the mode initial condition will converge to 50% of its initial value at
        approximately 7 * decay_rate seconds.

        """
        decay_rates = np.log(np.abs(self.eigvals)) / self.dt
        return decay_rates

    @property
    def modes_transient_time(self):
        """

        Returns:
            Time in seconds until which the initial condition of the mode decays to 10% of its initial value
            (if decay rate is positive).
        """
        decay_rates = self.modes_decay_rate
        transient_time = 1 / decay_rates * np.log(90. / 100.)
        transient_time[np.abs(decay_rates - 0.0) < 0.001] = (1 / 0.001) * np.log(90. / 100.)
        return transient_time

    def plot_eigfn_dynamics(
            self,
            eigfn: Optional[np.ndarray] = None,
            selected_mode_idx: Optional[list[int]] = None,
            replot: bool = False,
            ):
        """ Create plotly (n_modes) x 2 subplots to show the eigenfunctions dynamics.

        The first column will plot the eigenfunctions in the complex plane, while the second column will plot the
        eigenfunctions real part vs time. In the second plot we will show a vertical line for the mode's transient time,
        if the decay rate is positive.

        Args:
            eigfn (np.ndarray): Array of shape (context_window, N) or (N,) containing a trajectory of evaluated
                eigenfunctions at each time step or a single initial time-frame of eigenfunctions, respectively.
            selected_mode_idx: Modes to enable in the visualization. If None, all modes will be enabled.

        Returns:
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        true_eigfn_color = "red"

        if eigfn is not None:
            assert eigfn.shape[-1] == self._n_original_modes, \
                f"Expected eigenfunctions of shape (..., {self._n_original_modes}), got {eigfn.shape}."
            assert eigfn.ndim <= 2, \
                f"Expected eigenfunctions of shape (context_window_length, n_modes) or (n_modes,), got {eigfn.shape}."
            if eigfn.ndim == 1:  # Expand axis if only the eigenfunctions at time 0 are given. (n_modes,) -> (1,
                # n_modes)
                eigfn = eigfn[np.newaxis, :]
                gt_sorted_eigfn = None
            else:
                gt_sorted_eigfn = eigfn[..., self._sorted_eigs_indices]  # Sort the eigenfunctions
        else:
            eigfn = self.state_eigenbasis[0]
            gt_sorted_eigfn = eigfn  # Sort the eigenfunctions

        context_window_length, n_modes = eigfn.shape
        selected_mode_idx = np.ones_like(self.eigvals, dtype=bool) if selected_mode_idx is None else selected_mode_idx

        if self._modes_vs_time_fig is None or replot:
            # Fix plot visual parameters
            # ====================================================================================
            width = 150
            fig_width = 3 * width  # Assuming the second column and margins approximately take up another 400 pixels.
            fig_height = width * self.n_modes
            vertical_spacing = width * 0.25 / fig_height

            time = np.linspace(0, context_window_length * self.dt, context_window_length)

            # λ_i^t for t in [0, context_window_length)
            eigval_t = np.asarray([self.eigvals ** t for t in range(context_window_length)])
            eigfn_0 = eigfn[0, :]
            # pred_sorted_eigfn.shape = (context_window, self.n_modes)
            pred_sorted_eigfn = np.einsum("...l,tl->...tl", eigfn_0, eigval_t)

            fig = make_subplots(rows=self.n_modes, cols=2, column_widths=[0.33, 0.66],
                                subplot_titles=[f"Mode {i // 2}" for i in range(2 * self.n_modes)],
                                vertical_spacing=vertical_spacing,
                                shared_xaxes=True,
                                # shared_yaxes='rows'
                                )

            time_horizon = context_window_length * self.dt
            time_normalized = time / (time_horizon * self.dt)

            COLOR_SCALE = "Blugrn"
            for mode_idx in range(self.n_modes):
                show_legend = mode_idx == 0
                is_cmplx_mode = self.is_complex_mode[mode_idx]
                is_enabled_mode = True  # TODO: Make user selection dependent

                pred_eigfn_re = pred_sorted_eigfn[:, mode_idx].real  # Re(λ_i^t * <u_i,z_t>)
                pred_eigfn_im = pred_sorted_eigfn[:, mode_idx].imag  # Im(λ_i^t * <u_i,z_t>)

                if gt_sorted_eigfn is not None:
                    eigfn_re = gt_sorted_eigfn[:, mode_idx].real
                    eigfn_im = gt_sorted_eigfn[:, mode_idx].imag

                # Plotly usefull parameters
                legendgroup_pred = f"Pred"
                legendgroup_true = f"True"
                # First Column: Complex plane mode evolution -----------------------------------------------------------
                meta = dict(xaxis='mode im part', yaxis='mode re part')
                fig.add_trace(go.Scatter(x=pred_eigfn_im,  # PREDICTED eigenfunction
                                         y=pred_eigfn_re,
                                         mode='markers',
                                         marker=dict(color=time_normalized, colorscale=COLOR_SCALE, size=4),
                                         name=f"{legendgroup_pred} {mode_idx}",
                                         showlegend=show_legend,
                                         legendgroup=legendgroup_pred,
                                         meta=meta),
                              row=mode_idx + 1, col=1)

                if gt_sorted_eigfn is not None:
                    fig.add_trace(go.Scatter(x=eigfn_im,  # TRUE eigenfunction
                                             y=eigfn_re,
                                             mode='lines',
                                             line=dict(color=true_eigfn_color, width=1),
                                             name=f"{legendgroup_true} {mode_idx}",
                                             showlegend=show_legend,
                                             legendgroup=legendgroup_true,
                                             meta=meta),
                                  row=mode_idx + 1, col=1)

                # Second Column: Real part of the mode evolution vs. time ----------------------------------------------
                meta = dict(xaxis='time', yaxis='mode re part')
                fig.add_trace(go.Scatter(x=time,
                                         y=pred_eigfn_re * (2 if is_cmplx_mode else 1),
                                         mode='markers',
                                         marker=dict(color=time_normalized, colorscale=COLOR_SCALE, size=4),
                                         name=f"{legendgroup_pred} {mode_idx}",
                                         showlegend=False,
                                         legendgroup=legendgroup_pred,
                                         meta=meta),
                              row=mode_idx + 1, col=2)

                if gt_sorted_eigfn is not None:
                    # Plot the true real part of the eigenfunction dynamics vs time
                    fig.add_trace(go.Scatter(x=time,
                                             y=eigfn_re * (2 if is_cmplx_mode else 1),
                                             mode='lines',
                                             line=dict(color=true_eigfn_color, width=1),
                                             name=f"{legendgroup_pred} {mode_idx}",
                                             showlegend=False,
                                             legendgroup=legendgroup_true,
                                             meta=meta),
                                  row=mode_idx + 1, col=2)
                    # Plot the area between the predicted and true real part of the eigenfunction dynamics
                    fig.add_trace(go.Scatter(x=np.concatenate((time, time[::-1])),
                                             y=np.concatenate((pred_eigfn_re * (2 if is_cmplx_mode else 1),
                                                               eigfn_re[::-1] * (2 if is_cmplx_mode else 1))),
                                             fill='toself',
                                             fillcolor='rgba(1,1,1,0.1)',
                                             line=dict(width=0),
                                             showlegend=False,
                                             name=f"{legendgroup_pred} {mode_idx}",
                                             legendgroup=legendgroup_pred,
                                             meta=meta),
                                  row=mode_idx + 1, col=2)

            fig.update_layout(
                autosize=True,
                # width=fig_width,
                height=fig_height,
                showlegend=True,
                template=self.plotly_template,
                )

            # Set some prefixed range modes to improve visualization.
            for mode_idx in range(self.n_modes):
                fig.update_xaxes(rangemode='tozero', row=mode_idx + 1, col=1)
                fig.update_yaxes(rangemode='tozero', row=mode_idx + 1, col=1)
                fig.update_yaxes(rangemode='tozero', row=mode_idx + 1, col=2)
                fig.update_yaxes(scaleanchor="x", scaleratio=1, row=mode_idx + 1, col=1, )
            self._modes_vs_time_fig = fig
        else:
            raise NotImplementedError("Updating an already created figure is not yet implemented.")
            # fig = self._modes_vs_time_fig

        return fig

    def plot_modes_visualization(self, df, fig: Optional[Figure] = None, mode_group=None):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if df is None:  # Construct dataframe containing the data for easy plotting and selections across subplots.
            df = pd.DataFrame({
                "modes_idx":            range(self.n_modes),
                "modes_frequencies":    self.modes_frequency,
                "modes_modulus":        self.modes_modulus,
                "modes_decay_rate":     self.modes_decay_rate,
                "modes_transient_time": self.modes_transient_time})
            df["group"] = mode_group
            if mode_group is not None:
                assert len(
                    mode_group) == self._n_original_modes, "Number of colors must match the number of input eigvals"
                mode_group = mode_group[self._sorted_eigs_indices]  # Get the ordered and relevant mode colors.
                df["group"] = mode_group

        fig = make_subplots(rows=3,
                            cols=1,
                            subplot_titles=['Eigvalue', 'Modulus', 'Frequency'],
                            shared_xaxes=False,
                            shared_yaxes=False)

        # Plot the Real (y axis) and Imaginary (x axis) parts of the eigenvalues in the complex plane
        fig.add_trace(go.Scatter(x=self.eigvals.real,
                                 y=self.eigvals.imag,
                                 mode='markers',
                                 marker=dict(color='blue', size=8),
                                 name="Eigenvalues"),
                      row=1, col=1)

    def plot_eigvals_metrics(self, replot: bool = False,
                             mode_group: Optional[list[str]] = None,
                             fig: Optional[Figure] = None,
                             subplot_coords=((1, 1), (2, 1), (3, 1), (4, 1)),
                             ):
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        DISCRETE_COLOR_SCALE, CONTINUOUS_COLOR_SCALE = px.colors.qualitative.Prism, px.colors.sequential.Cividis_r

        if self._eigval_metrics_fig is None or replot:

            def get_px_figure(dataframe, x_col, y_col, marginal_y=None, marginal_x=None):
                discrete_color_map = None
                if 'group' in dataframe.columns:
                    coloring = 'group'
                else:
                    coloring = "modes_modulus"
                # coloscale = DISCRETE_COLOR_SCALE if 'group' in dataframe.columns else CONTINUOUS_COLOR_SCALE

                return px.scatter(dataframe,
                                  x=x_col,
                                  y=y_col,
                                  # symbol="is_complex_mode",
                                  # symbol_map={True: "circle", False: "square"},
                                  hover_data=["modes_transient_time", "modes_decay_rate"],
                                  hover_name="modes_sorted_idx",
                                  custom_data=["modes_sorted_idx", "modes_original_idx"],
                                  size="modes_modulus",
                                  size_max=10,
                                  # text=dataframe.modes_sorted_idx,  # Display index as text
                                  color=coloring,
                                  color_discrete_sequence=DISCRETE_COLOR_SCALE,
                                  color_continuous_scale=CONTINUOUS_COLOR_SCALE,
                                  template=self.plotly_template,
                                  marginal_y=marginal_y,
                                  marginal_x=marginal_x,
                                  )

            df = self._data_df
            df['modulus-amplitude'] = df['modes_modulus'] * df['modes_amplitude']
            if mode_group is not None:
                df["group"] = mode_group[self._sorted_eigs_indices]
            else:
                df["group"] = df["is_marginally_stable"]

            # Plot the Im vs. Real part of the eigenvalues in the complex plane
            fig_eig = get_px_figure(df, x_col="eigvals_re", y_col="eigvals_im")
            fig_modulus = get_px_figure(df, x_col="modes_sorted_idx", y_col="modes_modulus")
            fig_mod_amplitude = get_px_figure(df, x_col="modes_sorted_idx", y_col='modulus-amplitude')
            fig_freq = get_px_figure(df, x_col="modes_sorted_idx", y_col="modes_frequencies")

            # Create 3 subplots for the eigenvalues, modulus, and frequency. And transfer all traces to the subplots.
            if fig is None:
                fig = make_subplots(rows=4, cols=1,
                                    subplot_titles=['Eigenvalues', 'Modulus', 'Amplitude x Modulus', 'Frequency'])

            shared_style = dict(unselected={"marker": {"opacity": 0.3}, "textfont": {"color": "rgba(0, 0, 0, 0)"}})
            marker_style = {"line": {"width": 2, "color": "#BFC2C2"}}
            for trace in fig_eig.data:
                row, col = subplot_coords[0]
                # Plot the unit circle
                fig.add_trace(go.Scatter(x=np.cos(np.linspace(0, 2 * np.pi, 100)),
                                         y=np.sin(np.linspace(0, 2 * np.pi, 100)),
                                         mode='lines',
                                         line=dict(color='rgba(204, 0, 102,100)', width=4),
                                         name="Unit Circle",
                                         showlegend=False),
                              row=row, col=col)
                meta = dict(xaxis='eigvals_re', yaxis='eigvals_im')
                trace.update(meta=meta, **shared_style)
                trace.marker.update(**marker_style)
                fig.add_trace(trace, row=1, col=1)
                # Set the aspect ratio to be 1:1
                x_min, x_max = df["eigvals_re"].min(), df["eigvals_re"].max()
                y_min, y_max = df["eigvals_im"].min(), df["eigvals_im"].max()
                x_range, y_range = x_max - x_min, y_max - y_min
                # Agugment the limits by 10% of the range
                x_min, x_max = x_min - 0.1 * x_range, x_max + 0.1 * x_range
                y_min, y_max = y_min - 0.1 * y_range, y_max + 0.1 * y_range
                fig.update_yaxes(title_text="Imaginary",
                                 autorange=False,
                                 range=[y_min, y_max],
                                 row=row, col=col)
                fig.update_xaxes(title_text="Real",
                                 scaleanchor="y",
                                 autorange=False,
                                 range=[x_min, x_max],
                                 scaleratio=1, row=row, col=col)
            for trace in fig_modulus.data:
                row, col = subplot_coords[1]
                meta = dict(xaxis='modes_sorted_idx', yaxis='modes_modulus')
                trace.update(meta=meta, **shared_style, showlegend=False)
                trace.on_selection(self.selection_callback_handler)
                trace.marker.update(**marker_style)
                fig.add_trace(trace, row=row, col=col)
                fig.update_yaxes(title_text="Modulus", row=row, col=col)
                fig.update_xaxes(title_text="Modes", row=row, col=col)
            for trace in fig_mod_amplitude.data:
                row, col = subplot_coords[2]
                meta = dict(xaxis='modes_sorted_idx', yaxis='modulus-amplitude')
                trace.update(meta=meta, **shared_style, showlegend=False)
                trace.on_selection(self.selection_callback_handler)
                trace.marker.update(**marker_style)
                fig.add_trace(trace, row=row, col=col)
                fig.update_yaxes(title_text="Amplitude x Modulus", row=row, col=col)
                fig.update_xaxes(title_text="Modes", row=row, col=col)
            for trace in fig_freq.data:
                row, col = subplot_coords[3]
                meta = dict(xaxis='modes_sorted_idx', yaxis='modes_frequencies')
                trace.update(meta=meta, **shared_style, showlegend=False)
                trace.on_selection(self.selection_callback_handler)
                trace.marker.update(**marker_style)
                fig.add_trace(trace, row=row, col=col)
                fig.update_yaxes(
                    title_text=f"Frequency [{'Hz' if self.dt != 1.0 else '1/steps'}]", row=row, col=col)
                fig.update_xaxes(title_text="Modes", row=row, col=col)

            fig.update_layout(dragmode="select",  # instead of zooming in allow user to select ranges.
                              # hovermode=False,
                              # newselection_mode="lasso",
                              template=self.plotly_template,
                              )

            self._eigval_metrics_fig = fig
        else:
            raise NotImplementedError("Updating an already created figure is not yet implemented.")

        return fig

    def selection_callback_handler(self, trace, points, selector) -> None:
        print(f"Callback points: {points}")

    def visual_mode_selection(self, mode_colors: Optional[list[str]] = None):

        from dash import Dash, dcc, html, Input, Output, callback
        import pandas as pd
        import plotly.express as px

        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

        app = Dash(__name__, external_stylesheets=external_stylesheets)

        modes_info = self

        if mode_colors is None:
            mode_colors = np.arange(modes_info.n_modes).tolist()
        else:
            assert len(mode_colors) == self._n_original_modes, "Number of colors must match the number of input eigvals"
            mode_colors = mode_colors[self._sorted_eigs_indices]  # Get the ordered and relevant mode colors.

        # Assuming `modes_info` is an instance of `ModesInfo` class
        df = pd.DataFrame({
            "modes_idx":            range(modes_info.n_modes),
            "modes_frequencies":    modes_info.modes_frequency,
            "modes_modulus":        modes_info.modes_modulus,
            "modes_decay_rate":     modes_info.modes_decay_rate,
            "modes_transient_time": modes_info.modes_transient_time,
            "color":                mode_colors,
            })

        app.layout = html.Div([
            dcc.Graph(id="g1", config={"displayModeBar": False}),
            dcc.Graph(id="g2", config={"displayModeBar": False}),
            dcc.Graph(id="g3", config={"displayModeBar": False}),
            ])

        def get_figure(df, y_col, selectedpoints, selectedpoints_local):
            x_col = "modes_idx"

            if selectedpoints_local and "range" in selectedpoints_local and selectedpoints_local["range"]:
                ranges = selectedpoints_local["range"]
                selection_bounds = {
                    "x0": ranges["x"][0],
                    "x1": ranges["x"][1],
                    "y0": ranges["y"][0],
                    "y1": ranges["y"][1],
                    }
                custom_selection = True
            else:
                selection_bounds = {
                    "x0": np.min(df[x_col]),
                    "x1": np.max(df[x_col]),
                    "y0": np.min(df[y_col]),
                    "y1": np.max(df[y_col]),
                    }
                custom_selection = False

            fig = px.scatter(df,
                             x=df[x_col],
                             y=df[y_col],
                             text=df.index,  # Display index as text
                             color=df['color'],  # Use the 'color' column for coloring
                             color_discrete_sequence=px.colors.qualitative.Plotly
                             )

            fig.update_traces(
                selectedpoints=selectedpoints,
                customdata=df.modes_idx,
                mode="markers+text",
                marker={"size": 20},
                unselected={"marker": {"opacity": 0.3}, "textfont": {"color": "rgba(0, 0, 0, 0)"}},
                )

            fig.update_layout(
                margin={"l": 20, "r": 0, "b": 15, "t": 5},
                dragmode="select",  # instead of zooming in allow user to select ranges.
                hovermode=False,
                newselection_mode="gradual",
                )

            if custom_selection:
                fig.add_shape(dict(
                    {"type": "rect", "line": {"width": 2, "dash": "dot", "color": "darkgrey"}},
                    **selection_bounds
                    ))

            return fig

        @callback(
            Output("g1", "figure"),
            Output("g2", "figure"),
            Output("g3", "figure"),
            Input("g1", "selectedData"),
            Input("g2", "selectedData"),
            Input("g3", "selectedData"),
            )
        def callback(selection1, selection2, selection3):
            selectedpoints = np.asarray(df.index)

            for selected_data in [selection1, selection2, selection3]:
                if selected_data and selected_data["points"]:
                    selected_idx = [int(p["text"]) for p in selected_data["points"]]
                    selectedpoints = np.intersect1d(selectedpoints, selected_idx)
            print(f"Selected points: {selectedpoints}")

            return [
                get_figure(df, "modes_modulus", selectedpoints, selection1),
                get_figure(df, "modes_frequencies", selectedpoints, selection2),
                get_figure(df, "modes_decay_rate", selectedpoints, selection3),
                ]

        app.run_server(debug=False)
