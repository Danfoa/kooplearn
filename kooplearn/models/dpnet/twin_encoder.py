import copy
from typing import Optional, Union

import escnn.nn
import torch.nn
from escnn.nn import EquivariantModule

class TwinEncoder(torch.nn.Module):
    """
    A module composed of a single backbone encoder and two head layers which map to two different latent spaces H and
    H' of the same dimensionality.
    """
    def __init__(self,
                 backbone: Union[torch.nn.Module, EquivariantModule],
                 hidden_dim: int,
                 latent_dim: int,
                 latent_state_type: Optional[escnn.nn.FieldType] = None):
        """ TODO
        Args:
            backbone:
            latent_dim:
            latent_state_type:
        """
        super().__init__()
        self.backbone = backbone
        self.equivariant = isinstance(backbone, EquivariantModule)

        # Setting the bias of the linear layer to true is equivalent to setting the constant function in the basis
        # of the space of functions. Then the bias of each dimension is the coefficient of the constant function.
        if self.equivariant:
            # Bias term (a.k.a the constant function) is present only on the trivial isotypic subspace
            assert latent_state_type is not None, \
                f"Latent state type must be provided when using equivariant encoder"
            self.obs_H = escnn.nn.Linear(
                in_type=self.backbone.out_type,
                out_type=latent_state_type,
                bias=True)
            self.obs_H_prime = escnn.nn.Linear(
                in_type=self.backbone.out_type,
                out_type=latent_state_type,
                bias=True)
        else:
            assert latent_dim is not None and hidden_dim is not None, \
                f"Both latent and hidden dimensions must be provided when using non-equivariant encoder"
            self.obs_H = torch.nn.Linear(
                in_features=hidden_dim,
                out_features=latent_dim,
                bias=True)
            self.obs_H_prime = torch.nn.Linear(
                in_features=hidden_dim,
                out_features=latent_dim,
                bias=True)

    def forward(self, input):
        features = self.backbone(input)

        z = self.obs_H(features)
        z_prime = self.obs_H_prime(features)

        return z, z_prime


