from __future__ import annotations

import numpy as np
import torch

from torch.utils.data import Dataset


class TemporalSequenceDataset(Dataset):
    """
    Part 25A - Temporal Transformer Dataset.

    Converts a single chronological demand series into:

        [t-28 ... t-1] -> demand(t)

    The target at t is never included in the input sequence.
    """

    def __init__(
        self,
        values,
        sequence_length: int = 28,
        start_index: int | None = None,
        end_index: int | None = None,
    ) -> None:

        values = np.asarray(
            values,
            dtype=np.float32,
        ).reshape(-1)

        if len(values) <= sequence_length:
            raise ValueError(
                "Not enough observations for sequence length"
            )

        if sequence_length < 2:
            raise ValueError(
                "sequence_length must be >= 2"
            )

        self.values = values
        self.sequence_length = sequence_length

        first_target = sequence_length

        if start_index is None:
            start_index = first_target

        if end_index is None:
            end_index = len(values)

        self.indices = np.arange(
            max(
                first_target,
                start_index,
            ),
            min(
                len(values),
                end_index,
            ),
            dtype=np.int64,
        )

        if len(self.indices) == 0:
            raise ValueError(
                "No valid temporal sequences"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self,
        index: int,
    ):

        target_index = int(
            self.indices[index]
        )

        start = (
            target_index
            -
            self.sequence_length
        )

        sequence = self.values[
            start:target_index
        ]

        target = self.values[
            target_index
        ]

        return (
            torch.tensor(
                sequence,
                dtype=torch.float32,
            ).unsqueeze(-1),
            torch.tensor(
                target,
                dtype=torch.float32,
            ),
        )