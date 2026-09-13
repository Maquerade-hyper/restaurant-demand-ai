from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.
    """

    def __init__(
        self,
        d_model: int,
        max_length: int = 512,
    ) -> None:

        super().__init__()

        position = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            *
            (
                -math.log(10000.0)
                /
                d_model
            )
        )

        encoding = torch.zeros(
            max_length,
            d_model,
        )

        encoding[:, 0::2] = torch.sin(
            position * div_term
        )

        encoding[:, 1::2] = torch.cos(
            position * div_term
        )

        encoding = encoding.unsqueeze(0)

        self.register_buffer(
            "encoding",
            encoding,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        sequence_length = x.size(1)

        return (
            x
            +
            self.encoding[
                :,
                :sequence_length,
                :,
            ]
        )


class TemporalTransformer(nn.Module):
    """
    Part 25B - Lightweight Temporal Transformer.

    Input:
        [batch, sequence, features]

    Output:
        [batch]

    Designed deliberately small for CPU benchmarking.
    """

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        feedforward_dim: int = 64,
        dropout: float = 0.10,
        max_sequence_length: int = 128,
    ) -> None:

        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads"
            )

        self.input_projection = nn.Linear(
            input_size,
            d_model,
        )

        self.position = PositionalEncoding(
            d_model=d_model,
            max_length=max_sequence_length,
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.head = nn.Sequential(
            nn.Linear(
                d_model,
                d_model,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                d_model,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.input_projection(
            x
        )

        x = self.position(
            x
        )

        x = self.encoder(
            x
        )

        # Last temporal representation is used
        # for next-step forecasting.
        x = x[:, -1, :]

        x = self.norm(
            x
        )

        output = self.head(
            x
        )

        return output.squeeze(
            -1
        )