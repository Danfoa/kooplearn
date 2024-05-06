import copy
from typing import Optional, Tuple, Union

import escnn.nn
import torch.nn
from escnn.nn import EquivariantModule


class TwinEncoder(torch.nn.Module):
    """
    A module composed of a single backbone encoder and two head layers which map to two different latent spaces H and
    H' of the same dimensionality.
    """

    def __init__(self,
                 backbone: Union[torch.nn.Module],
                 hidden_dim: int,
                 latent_dim: int,
                 activation: type[torch.nn.Module] = torch.nn.Identity):
        super().__init__()
        self.backbone = backbone
        self.activation = activation()

        activation_name = self.activation.__class__.__name__.lower()
        activation_name = "linear" if activation_name == "selu" or activation_name == 'identity' else activation_name

        self.obs_H = torch.nn.Linear(
            in_features=hidden_dim,
            out_features=latent_dim,
            bias=False)  # Avoid biasing the learned functions
        self.obs_H_prime = torch.nn.Linear(
            in_features=hidden_dim,
            out_features=latent_dim,
            bias=False)  # Avoid biasing the learned functions
        torch.nn.init.kaiming_normal_(self.obs_H.weight, nonlinearity=activation_name)
        torch.nn.init.kaiming_normal_(self.obs_H_prime.weight, nonlinearity=activation_name)

    def forward(self, input):
        features = self.backbone(input)

        z = self.obs_H(features)
        z_prime = self.obs_H_prime(features)

        return self.activation(z), self.activation(z_prime)


class EquivTwinEncoder(escnn.nn.EquivariantModule):
    """
    A module composed of a single backbone encoder and two head layers which map to two different latent spaces H and
    H' of the same dimensionality.
    """

    def __init__(self,
                 backbone: escnn.nn.EquivariantModule,
                 out_type: escnn.nn.FieldType,
                 activation: type[escnn.nn.EquivariantModule] = escnn.nn.IdentityModule):
        super().__init__()
        self.in_type = backbone.in_type
        self.out_type = out_type

        self.backbone = backbone
        self.activation = activation(in_type=out_type)

        self.obs_H = escnn.nn.Linear(
            in_type=self.backbone.out_type,
            out_type=out_type,
            bias=False)
        self.obs_H_prime = escnn.nn.Linear(
            in_type=self.backbone.out_type,
            out_type=out_type,
            bias=False)

        # with torch.no_grad():
        #     W1, _ = self.obs_H.expand_parameters()
        #     W2, _ = self.obs_H_prime.expand_parameters()
        #     # Perform SVD decomposition
        #     U1, S1, V1 = torch.svd(W1)
        #     U2, S2, V2 = torch.svd(W2)
        #     self.obs_H.weights.data = self.obs_H.weights.data / S1.max()
        #     self.obs_H_prime.weights.data = self.obs_H_prime.weights.data / S2.max()

    def forward(self, input: escnn.nn.GeometricTensor):
        assert input.type == self.backbone.in_type, \
            f"Input type {input.type} does not match the expected type {self.backbone.in_type}"

        features = self.backbone(input)
        z = self.obs_H(features)
        z_prime = self.obs_H_prime(features)

        return self.activation(z), self.activation(z_prime)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return self.out_type.size, self.out_type.size
