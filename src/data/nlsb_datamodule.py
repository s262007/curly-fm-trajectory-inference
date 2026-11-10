"""DataModule for the NLSB scRNA-seq dataset.

Source: https://github.com/take-koshizuka/NLSB/tree/main/data

The dataset contains 5 developmental snapshots (ts = 0,1,2,3,4) of a 100-dimensional
scRNA-seq embedding together with RNA-velocity vectors. Following the project email
instructions, we treat snapshots ``0, 2, 4`` as training timepoints and snapshots
``1, 3`` as validation (held-out) timepoints.

The class produces:

* ``self.positions``: list[Tensor] of length ``n_timepoints``, one ``(N_t, D)`` tensor per snapshot
* ``self.velocities``: list[Tensor], same structure as positions
* ``self.train_ts`` / ``self.val_ts``: snapshot indices used for train / val
* per-snapshot ``train_dataloaders`` / ``val_dataloaders`` (LV-style API).

A ``StandardScaler``-like normalisation is applied across all cells (consistent with the
NLSB reference implementation): positions are centred / scaled by feature-wise mean/std
computed on the union of all snapshots and velocities are divided by the same scale.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
from pytorch_lightning.utilities.combined_loader import CombinedLoader


class NLSBSnapshotDataset(Dataset):
    """Returns (x, v) pairs for a single snapshot."""

    def __init__(self, x: torch.Tensor, v: torch.Tensor):
        assert x.shape == v.shape, f"x {x.shape} vs v {v.shape}"
        self.x = x
        self.v = v

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.v[idx]


class NLSBDataModule(LightningDataModule):
    """LightningDataModule for the NLSB EB scRNA-seq dataset.

    Parameters
    ----------
    data_dir:
        Directory containing ``train_rna.npz``, ``val_rna.npz``, ``test_rna.npz``.
    files:
        Which raw files to combine into the cell pool. By default we use all
        three so that every snapshot contains as many cells as possible — the
        train/val split happens *across snapshot indices*, not across the
        original NLSB file boundaries.
    dim:
        Number of feature dimensions to keep (defaults to the full 100).
    train_ts / val_ts:
        Snapshot indices to use as training / validation marginals. Defaults
        follow the project email: train on 0,2,4 — validate on 1,3.
    normalise:
        If True, fit a feature-wise standardiser on the pooled cells of the
        training snapshots and apply it to both X and V (V is divided by the
        scale only, matching the NLSB ``v /= scaler.scale_`` convention).
    """

    def __init__(
        self,
        data_dir: str = "data/nlsb",
        files: Sequence[str] = ("train_rna.npz", "val_rna.npz", "test_rna.npz"),
        dim: int = 100,
        train_ts: Sequence[int] = (0, 2, 4),
        val_ts: Sequence[int] = (1, 3),
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
        normalise: bool = True,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.files = list(files)
        self.dim = int(dim)
        self.train_ts = list(train_ts)
        self.val_ts = list(val_ts)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.normalise = bool(normalise)
        self.seed = int(seed)

        # ---- 1. load & pool raw arrays ---------------------------------
        X_list, V_list, T_list = [], [], []
        for fname in self.files:
            path = os.path.join(self.data_dir, fname)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Missing NLSB file: {path}. Download from "
                    "https://github.com/take-koshizuka/NLSB/tree/main/data"
                )
            npz = np.load(path)
            X_list.append(npz["X"])
            V_list.append(npz["v"])
            T_list.append(npz["ts"])

        X = np.concatenate(X_list, axis=0).astype(np.float64)
        V = np.concatenate(V_list, axis=0).astype(np.float64)
        ts = np.concatenate(T_list, axis=0).astype(np.int64)

        # keep only the requested feature dimensions
        X = X[:, : self.dim]
        V = V[:, : self.dim]

        # ---- 2. standardise on training snapshots ----------------------
        if self.normalise:
            train_mask = np.isin(ts, self.train_ts)
            mean = X[train_mask].mean(axis=0, keepdims=True)
            scale = X[train_mask].std(axis=0, keepdims=True)
            scale[scale == 0] = 1.0
            X = (X - mean) / scale
            V = V / scale  # NLSB convention: divide velocity by feature scale
            self.scaler_mean_ = torch.tensor(mean.squeeze(0), dtype=torch.float32)
            self.scaler_scale_ = torch.tensor(scale.squeeze(0), dtype=torch.float32)
        else:
            self.scaler_mean_ = torch.zeros(self.dim, dtype=torch.float32)
            self.scaler_scale_ = torch.ones(self.dim, dtype=torch.float32)

        # ---- 3. group by snapshot --------------------------------------
        self.timepoints: List[int] = sorted(int(t) for t in np.unique(ts))
        self.num_timesteps = len(self.timepoints)

        self.positions: List[torch.Tensor] = []
        self.velocities: List[torch.Tensor] = []
        self.cell_counts: List[int] = []
        for t in self.timepoints:
            mask = ts == t
            self.positions.append(torch.tensor(X[mask], dtype=torch.float32))
            self.velocities.append(torch.tensor(V[mask], dtype=torch.float32))
            self.cell_counts.append(int(mask.sum()))

        # convenience views: only train / only val
        self.x_train = [self.positions[t] for t in self.train_ts]
        self.v_train = [self.velocities[t] for t in self.train_ts]
        self.x_val = [self.positions[t] for t in self.val_ts]
        self.v_val = [self.velocities[t] for t in self.val_ts]

        # ---- 4. per-snapshot dataloaders -------------------------------
        gen = torch.Generator().manual_seed(self.seed)
        self.train_dataloaders: List[DataLoader] = []
        for x_t, v_t in zip(self.x_train, self.v_train):
            ds = NLSBSnapshotDataset(x_t, v_t)
            self.train_dataloaders.append(
                DataLoader(
                    ds,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    shuffle=True,
                    drop_last=False,
                    generator=gen,
                )
            )

        self.val_dataloaders: List[DataLoader] = []
        for x_t, v_t in zip(self.x_val, self.v_val):
            ds = NLSBSnapshotDataset(x_t, v_t)
            self.val_dataloaders.append(
                DataLoader(
                    ds,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    shuffle=False,
                    drop_last=False,
                )
            )

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def train_dataloader(self):
        return CombinedLoader(self.train_dataloaders, mode="min_size")

    def val_dataloader(self):
        if not self.val_dataloaders:
            return None
        return CombinedLoader(self.val_dataloaders, mode="min_size")

    def test_dataloader(self):
        # same as validation here (held-out snapshots 1, 3)
        return self.val_dataloader()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = [
            f"NLSB EB dataset — dim={self.dim}",
            f"  snapshots: {self.timepoints}",
            f"  cells per snapshot: {dict(zip(self.timepoints, self.cell_counts))}",
            f"  train snapshots: {self.train_ts}",
            f"  val   snapshots: {self.val_ts}",
            f"  normalised: {self.normalise}",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    dm = NLSBDataModule()
    print(dm.summary())
