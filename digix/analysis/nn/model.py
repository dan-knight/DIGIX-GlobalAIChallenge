from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_d: int,
        output_d: int,
        hidden_d: int,
        num_layers: int,
        activation: "nn.Module"
    ):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        
        self.input_projection = nn.Linear(input_d, hidden_d)
        self.hidden_layers: list[nn.Module] = [
            *[nn.Linear(hidden_d, hidden_d) for _ in range(num_layers)]
        ]
        self.activation = activation
        self.output_projection = nn.Linear(hidden_d, output_d)

    def forward(self, inputs: "Tensor"):
        x = torch.flatten(inputs, 1)
        x = self.activation(self.input_projection(x))
        for layer in self.hidden_layers:
            x = self.activation(x + layer(x))
        output = self.output_projection(x)
        return output
    