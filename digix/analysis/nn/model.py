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
    ):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_d, hidden_d), nn.ReLU()])
        self.layers = nn.Sequential(
            nn.Linear(input_d, hidden_d),
            nn.ReLU(),
            *layers,
            nn.Linear(hidden_d, output_d)
        )

    def forward(self, inputs: "Tensor"):
        x = torch.flatten(inputs, 1)
        return self.layers(x)
    