from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.single_marginal_utils import *
import torchcfm
from torchcfm.utils import plot_trajectories
import ot as pot
import torchdyn
from torchdyn.core import NeuralODE
from torchcfm.utils import *
from src.models.components.mmd_metrics import *
import scipy

class CFMModule(LightningModule):
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        net: torch.nn.Module,
        geodesic_net: torch.nn.Module,
        datamodule: LightningModule,
        compile: bool,
    ) -> None:
    
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.geodesic_net = geodesic_net

        self.datamodule = datamodule

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.x0 = list()
        self.x1 = list()
        self.v0 = list()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.geodesic_net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def get_xt_xt_dot(self, t, x0, x1, sigma=0.0):
        with torch.enable_grad():
            t = t[..., None]
            t.requires_grad_(True)
            geo_input = torch.cat([x0, x1, t], dim=-1)
            mu_t = (1 - t) * x0 + t * x1 + t * (1 - t) * self.geodesic_net(geo_input)
            
            mu_t_dot_list = []
            for i in range(mu_t.shape[-1]):
                mu_t_dot_list.append(
                    torch.autograd.grad(torch.sum(mu_t[..., i]), t, create_graph=True)[0]
                )
            mu_t_dot = torch.cat(mu_t_dot_list, -1)
            
            epsilon = torch.randn_like(x0)
            xt = mu_t + torch.sqrt(t*(1-t)) * sigma * epsilon
            
        return xt, mu_t_dot, epsilon

    def get_ut_knn_gaussian(self, xt, x0, x1, v0, v1, k=10, eps=1e-12):
        x = torch.cat([x0, x1], dim=0)
        v = torch.cat([v0, v1], dim=0)
        dists = torch.cdist(xt, x)
        
        real_k = min(k, x.shape[0])
        knn_dists, knn_idx = torch.topk(dists, k=real_k, dim=1, largest=False)
        
        h = knn_dists[:, -1:].clamp_min(eps)
        w = torch.exp(-(knn_dists**2) / (2 * h**2))
        w = w / (w.sum(dim=1, keepdim=True) + eps)
        
        v_knn = v[knn_idx]
        v_xt = (w.unsqueeze(-1) * v_knn).sum(dim=1)
        return v_xt

    def parse_batch(self, batch):
        """Parse the batch from CombinedLoader.
        
        CombinedLoader with mode='min_size' returns:
          (list_of_dataloader_outputs, batch_idx, dataloader_idx)
        where list_of_dataloader_outputs is a list of T tuples, each (x, v).
        
        We pair consecutive timesteps: (x_t, v_t) -> (x_{t+1}, v_{t+1})
        """
        # Unwrap the CombinedLoader tuple: (data_list, batch_idx, dataloader_idx)
        if isinstance(batch, (list, tuple)):
            # Check if it looks like (data_list, int, int) from CombinedLoader
            if len(batch) == 3 and isinstance(batch[1], int):
                batch = batch[0]
            # Also handle (data_list, int) format
            elif len(batch) == 2 and isinstance(batch[1], int):
                batch = batch[0]
        
        # Now batch should be a list of T tuples, each (x_i, v_i)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            first_elem = batch[0]
            # Check it's actually a list of (x, v) tuples/lists
            if isinstance(first_elem, (list, tuple)) and len(first_elem) == 2:
                x0s, x1s, v0s, v1s = [], [], [], []
                for i in range(len(batch) - 1):
                    x0s.append(batch[i][0])
                    v0s.append(batch[i][1])
                    x1s.append(batch[i+1][0])
                    v1s.append(batch[i+1][1])
                x0 = torch.cat(x0s, dim=0)
                x1 = torch.cat(x1s, dim=0)
                v0 = torch.cat(v0s, dim=0)
                v1 = torch.cat(v1s, dim=0)
                return x0, x1, v0, v1
        
        # Fallback: assume it's already (x0, x1, v0, v1)
        return batch

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x0, x1, v0, v1 = self.parse_batch(batch)
        batch_size = x0.shape[0]
        
        # 1. Provide coupling
        t = torch.rand(1).type_as(x0) * torch.ones((batch_size, batch_size), device=x0.device)
        x0_r = x0.repeat(batch_size, 1, 1)
        x1_r = x1.repeat(batch_size, 1, 1).transpose(0, 1)
        
        xt, mu_t_dot, _ = self.get_xt_xt_dot(t, x0_r, x1_r, sigma=0.05)
        
        xt_flat = xt.reshape(-1, xt.shape[-1])
        ut_flat = self.get_ut_knn_gaussian(xt_flat, x0, x1, v0, v1, k=20)
        ut = ut_flat.reshape(batch_size, batch_size, -1)
        
        L2_cost = 0.5 * ((mu_t_dot.detach() - ut) ** 2).sum(-1)
        
        _, j = scipy.optimize.linear_sum_assignment(L2_cost.detach().cpu().numpy())
        
        pi_x0 = x0[j]
        pi_v0 = v0[j]
        
        # 2. Re-compute with coupled pairs
        t = torch.rand(batch_size).type_as(x0)
        xt, mu_t_dot, eps = self.get_xt_xt_dot(t, pi_x0, x1, sigma=0.05)
        
        ut = self.get_ut_knn_gaussian(xt, pi_x0, x1, pi_v0, v1, k=20)
        
        # Stage 1: Geo loss
        cosine_loss = 1 - torch.nn.functional.cosine_similarity(ut, mu_t_dot).mean()
        loss_geo = torch.mean((ut - mu_t_dot) ** 2) + cosine_loss
        
        # Stage 2: Vel loss
        vt = self.net(torch.cat([xt.detach(), t[:, None]], dim=-1))
        loss_vel = torch.mean((vt - mu_t_dot.detach()) ** 2)
        
        loss = loss_geo + loss_vel

        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        x0, x1, v0, v1 = self.parse_batch(batch)

        if (self.current_epoch+1)%100==0:
            node = NeuralODE(
            torch_wrapper(self.net), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
            with torch.no_grad():
                traj = node.trajectory(
                    x0,
                    t_span=torch.linspace(0, 1, 100),
                )
                # torch.save(traj, os.path.join('traj_cfm', f"traj_{self.current_epoch}.pt"))
                # plot_trajectories_f(traj, x1,'traj', self.current_epoch)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)
        x0, x1, v0, v1 = self.parse_batch(batch)
        self.x0.append(x0)
        self.x1.append(x1)
        self.v0.append(v0)

        self.val_loss(loss)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        if len(self.x0) == 0:
            return
            
        x0 = torch.cat(self.x0, dim=0)
        x1 = torch.cat(self.x1, dim=0)
        v0 = torch.cat(self.v0, dim=0)
        
        self.x0 = list()
        self.x1 = list()    
        self.v0 = list()

        vt_0 = self.net(torch.cat([x0, torch.zeros_like(x0[:, :1])], dim=-1))

        train_x = torch.cat(self.datamodule.xs, dim=0).to(self.device)
        train_vel = torch.cat(self.datamodule.vs, dim=0).to(self.device)
        wrapped_model = TorchWrapperWithMetrics(self.net, train_x, train_vel)
        node = NeuralODE(
            wrapped_model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        with torch.no_grad():
            z0 = torch.cat([x0, torch.zeros_like(x0[:, :2])], dim=1)
            traj_augmented = node.trajectory(
                z0,
                t_span=torch.linspace(0, 1, 2),
            )

        traj = traj_augmented[..., :-2]
        cossin_traj = traj_augmented[..., -2]
        L2_traj = traj_augmented[..., -1]
        self.val_cos_integral = cossin_traj[-1].mean()       
        self.val_L2_integral = L2_traj[-1].mean()
        
        self.val_w2_x = torchcfm.optimal_transport.wasserstein(traj[-1], x1)
        self.val_cos_v0 = 1 - torch.nn.functional.cosine_similarity(vt_0, v0).mean()
        metrics_dict = compute_distribution_distances_with_prefix(traj[-1], x1, "val")
        self.val_mmd = metrics_dict['val/RBF_MMD'] 

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/w2_x", self.val_w2_x, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/cos_v0", self.val_cos_v0, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mmd", self.val_mmd, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/cosdist_integral", self.val_cos_integral, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/L2_integral", self.val_L2_integral, on_step=False, on_epoch=True, prog_bar=True)

        # plot_velocity_field(self.net, self.datamodule.train_x, self.device, self.current_epoch)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)
        x0, x1, v0, v1 = self.parse_batch(batch)

        self.x0.append(x0)
        self.x1.append(x1)
        self.v0.append(v0)

        self.test_loss(loss)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        if len(self.x0) == 0:
            return
            
        x0 = torch.cat(self.x0, dim=0)
        x1 = torch.cat(self.x1, dim=0)
        v0 = torch.cat(self.v0, dim=0)
        
        self.x0 = list()
        self.x1 = list()    
        self.v0 = list()
        vt_0 = self.net(torch.cat([x0, torch.zeros_like(x0[:, :1])], dim=-1))

        train_x = torch.cat(self.datamodule.xs, dim=0).to(self.device)
        train_vel = torch.cat(self.datamodule.vs, dim=0).to(self.device)
        wrapped_model = TorchWrapperWithMetrics(self.net, train_x, train_vel)
        node = NeuralODE(
            wrapped_model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        with torch.no_grad():
            z0 = torch.cat([x0, torch.zeros_like(x0[:, :2])], dim=1)
            traj_augmented = node.trajectory(
                z0,
                t_span=torch.linspace(0, 1, 2),
            )

        traj = traj_augmented[..., :-2]
        cossin_traj = traj_augmented[..., -2]
        L2_traj = traj_augmented[..., -1]
        self.test_cos_integral = cossin_traj[-1].mean()       
        self.test_L2_integral = L2_traj[-1].mean()
        
        self.test_w2_x = torchcfm.optimal_transport.wasserstein(traj[-1], x1)
        self.test_cos_v0 = 1 - torch.nn.functional.cosine_similarity(vt_0, v0).mean()
        metrics_dict = compute_distribution_distances_with_prefix(traj[-1], x1,"test")
        self.test_mmd = metrics_dict['test/RBF_MMD']  

        # update and log metrics
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/w2_x", self.test_w2_x, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/cos_v0", self.test_cos_v0, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mmd", self.test_mmd, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/cos_dist_integral", self.test_cos_integral, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/L2_integral", self.test_L2_integral, on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        params = list(self.net.parameters()) + list(self.geodesic_net.parameters())
        optimizer = self.hparams.optimizer(params=params)
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = CFMModule(None, None, None, None)
