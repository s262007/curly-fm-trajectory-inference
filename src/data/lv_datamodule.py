import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from pytorch_lightning.utilities.combined_loader import CombinedLoader

class LVDataset(Dataset):
    def __init__(self, x, v):
        self.x = x
        self.v = v
        
    def __getitem__(self, idx):
        return self.x[idx], self.v[idx]
        
    def __len__(self):
        return len(self.x)

class LVDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "Answer Luca",
        split_ratios: tuple[float, float, float] = (0.8, 0.0, 0.2),
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.data_dir = data_dir

        self.train_dataloaders = []
        self.val_dataloaders = []
        self.test_dataloaders = []

        # Load Lotka-Volterra data and computed velocities
        # Data shape is originally (N, T, D) i.e. (100, 5, 2) for lv_train
        pos_path = os.path.join(self.data_dir, "lv_train")
        vel_path = os.path.join(self.data_dir, "lv_train_velocities.npy")
        
        positions = np.load(pos_path)
        velocities = np.load(vel_path)
        
        # Convert to torch tensor
        data_tensor = torch.tensor(positions, dtype=torch.float32)
        vel_tensor = torch.tensor(velocities, dtype=torch.float32)

        # Transpose from (N, T, D) to (T, N, D) to match codebase structure
        data_tensor = data_tensor.transpose(0, 1)
        vel_tensor = vel_tensor.transpose(0, 1)

        self.num_timesteps = data_tensor.shape[0]
        num_particles = data_tensor.shape[1]

        generator = torch.Generator().manual_seed(self.seed)

        self.xs = []
        self.vs = []

        # Shuffle particles
        shuffled_indices = torch.randperm(num_particles, generator=generator)
        data_tensor = data_tensor[:, shuffled_indices, :]
        vel_tensor = vel_tensor[:, shuffled_indices, :]

        split_index_1 = int(num_particles * split_ratios[0])  
        split_index_2 = int(num_particles * (split_ratios[0] + split_ratios[1]))
 
        for idx in range(self.num_timesteps): 
            frame_data_X = data_tensor[idx]
            frame_data_V = vel_tensor[idx]   

            # Split
            train_data_X = frame_data_X[:split_index_1]
            val_data_X = frame_data_X[split_index_1:split_index_2]
            test_data_X = frame_data_X[split_index_2:]

            train_data_V = frame_data_V[:split_index_1]
            val_data_V = frame_data_V[split_index_1:split_index_2]
            test_data_V = frame_data_V[split_index_2:]

            train_dataset = LVDataset(train_data_X, train_data_V)
            val_dataset = LVDataset(val_data_X, val_data_V)
            test_dataset = LVDataset(test_data_X, test_data_V)

            self.xs.append(train_data_X)
            self.vs.append(train_data_V)

            self.train_dataloaders.append(
                DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    shuffle=True,
                    drop_last=False, # Dropping last could be an issue with small N
                )
            )
            if len(val_dataset) > 0:
                self.val_dataloaders.append(
                    DataLoader(
                        val_dataset,
                        batch_size=self.batch_size,
                        num_workers=self.hparams.num_workers,
                        pin_memory=self.hparams.pin_memory,
                        shuffle=False,
                        drop_last=False,
                    )
                )

            if len(test_dataset) > 0:
                self.test_dataloaders.append(
                    DataLoader(
                        test_dataset,
                        batch_size=test_data_X.shape[0] if test_data_X.shape[0] > 0 else 1,
                        num_workers=self.hparams.num_workers,
                        pin_memory=self.hparams.pin_memory,
                        shuffle=False,
                        drop_last=False,
                    )
                )

    def train_dataloader(self):
        return CombinedLoader(self.train_dataloaders, mode="min_size")

    def val_dataloader(self):
        if len(self.val_dataloaders) > 0:
            return CombinedLoader(self.val_dataloaders, mode="min_size")
        return None

    def test_dataloader(self):
        if len(self.test_dataloaders) > 0:
            return CombinedLoader(self.test_dataloaders, "max_size")
        return None
