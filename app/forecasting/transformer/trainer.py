from __future__ import annotations

import time

import numpy as np
import torch

from torch.utils.data import DataLoader

from app.forecasting.transformer.dataset import (
    TemporalSequenceDataset,
)

from app.forecasting.transformer.model import (
    TemporalTransformer,
)


class TransformerTrainer:
    """
    Part 25C - Temporal Transformer Training & Inference.

    Important temporal contract:

    Validation sequences use the historical training context,
    but validation targets themselves remain unseen during
    training.

    This prevents the validation window from being incorrectly
    treated as an independent time series.
    """

    def __init__(
        self,
        sequence_length: int = 28,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        feedforward_dim: int = 64,
        dropout: float = 0.10,
        epochs: int = 12,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 4,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:

        self.sequence_length = sequence_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.seed = seed
        self.device = torch.device(device)

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model: TemporalTransformer | None = None

        self.mean = 0.0
        self.std = 1.0

        self.training_seconds = 0.0

    # ============================================================
    # FIT
    # ============================================================

    def fit(
        self,
        train_values,
        validation_values=None,
    ) -> dict:

        train_values = np.asarray(
            train_values,
            dtype=np.float32,
        ).reshape(-1)

        if len(train_values) <= self.sequence_length:
            raise ValueError(
                "Training series is too short"
            )

        self.mean = float(
            np.mean(train_values)
        )

        self.std = float(
            np.std(train_values)
        )

        if self.std < 1e-6:
            self.std = 1.0

        scaled_train = (
            train_values - self.mean
        ) / self.std

        train_dataset = TemporalSequenceDataset(
            scaled_train,
            sequence_length=self.sequence_length,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.model = TemporalTransformer(
            input_size=1,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout,
            max_sequence_length=self.sequence_length,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        loss_function = torch.nn.HuberLoss()

        start_time = time.perf_counter()

        best_loss = float("inf")
        best_state = None
        stale_epochs = 0

        history = []

        for epoch in range(self.epochs):

            self.model.train()

            total_loss = 0.0
            count = 0

            for x, y in train_loader:

                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()

                prediction = self.model(x)

                loss = loss_function(
                    prediction,
                    y,
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

                batch_count = len(y)

                total_loss += (
                    float(loss.item())
                    * batch_count
                )

                count += batch_count

            train_loss = (
                total_loss
                /
                max(count, 1)
            )

            validation_loss = None

            if validation_values is not None:

                validation_loss = self.validation_loss(
                    history_values=train_values,
                    validation_values=validation_values,
                )

            monitored_loss = (
                validation_loss
                if validation_loss is not None
                else train_loss
            )

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            )

            if monitored_loss < best_loss:

                best_loss = monitored_loss

                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value
                    in self.model.state_dict().items()
                }

                stale_epochs = 0

            else:

                stale_epochs += 1

            if stale_epochs >= self.patience:
                break

        self.training_seconds = (
            time.perf_counter()
            - start_time
        )

        if best_state is not None:

            self.model.load_state_dict(
                best_state
            )

        return {
            "epochs_completed": len(history),
            "best_loss": best_loss,
            "history": history,
            "training_seconds": self.training_seconds,
        }

    # ============================================================
    # VALIDATION LOSS
    # ============================================================

    def validation_loss(
        self,
        history_values,
        validation_values,
    ) -> float:
        """
        Evaluate validation targets using historical context.

        Example:

            history = 1 ... 150
            validation = 151 ... 164
            sequence_length = 14

        The first validation prediction receives:

            observations 137 ... 150

        as its input context.

        We do NOT require validation itself to contain 14
        observations before creating its first target.
        """

        if self.model is None:
            raise RuntimeError(
                "Model has not been fitted"
            )

        history_values = np.asarray(
            history_values,
            dtype=np.float32,
        ).reshape(-1)

        validation_values = np.asarray(
            validation_values,
            dtype=np.float32,
        ).reshape(-1)

        if len(history_values) < self.sequence_length:
            raise ValueError(
                "Historical context is too short"
            )

        if len(validation_values) == 0:
            raise ValueError(
                "Validation series is empty"
            )

        combined = np.concatenate(
            [
                history_values,
                validation_values,
            ]
        )

        scaled = (
            combined - self.mean
        ) / self.std

        history_length = len(
            history_values
        )

        validation_start = history_length

        validation_end = len(
            combined
        )

        dataset = TemporalSequenceDataset(
            scaled,
            sequence_length=self.sequence_length,
            start_index=validation_start,
            end_index=validation_end,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.model.eval()

        loss_function = torch.nn.HuberLoss()

        total = 0.0
        count = 0

        with torch.no_grad():

            for x, y in loader:

                x = x.to(self.device)
                y = y.to(self.device)

                prediction = self.model(x)

                loss = loss_function(
                    prediction,
                    y,
                )

                batch_count = len(y)

                total += (
                    float(loss.item())
                    * batch_count
                )

                count += batch_count

        return (
            total
            /
            max(count, 1)
        )

    # ============================================================
    # PREDICT
    # ============================================================

    def predict(
        self,
        history,
        horizon: int = 1,
    ) -> np.ndarray:

        if self.model is None:
            raise RuntimeError(
                "Model has not been fitted"
            )

        if horizon < 1:
            raise ValueError(
                "horizon must be >= 1"
            )

        history = np.asarray(
            history,
            dtype=np.float32,
        ).reshape(-1)

        if len(history) < self.sequence_length:
            raise ValueError(
                "Not enough history for prediction"
            )

        working = list(history)
        predictions = []

        self.model.eval()

        with torch.no_grad():

            for _ in range(horizon):

                window = np.asarray(
                    working[
                        -self.sequence_length:
                    ],
                    dtype=np.float32,
                )

                scaled = (
                    window - self.mean
                ) / self.std

                x = torch.tensor(
                    scaled,
                    dtype=torch.float32,
                    device=self.device,
                ).reshape(
                    1,
                    self.sequence_length,
                    1,
                )

                scaled_prediction = (
                    self.model(x).item()
                )

                prediction = (
                    scaled_prediction
                    * self.std
                    + self.mean
                )

                prediction = max(
                    0.0,
                    float(prediction),
                )

                predictions.append(
                    prediction
                )

                working.append(
                    prediction
                )

        return np.asarray(
            predictions,
            dtype=np.float64,
        )