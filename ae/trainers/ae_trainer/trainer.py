# imports SummaryWriter from pytorch
from torch.utils.tensorboard import SummaryWriter
from flax.training import train_state, checkpoints
from flax import linen as nn
from tqdm import tqdm
import jax
import numpy as np
import os
from typing import Callable
import optax
from torch.utils.data import DataLoader


class TrainerModule:
    def __init__(
        self,
        checkpoint_path,
        model,
        c_hid,
        loss_fn: Callable,
        latent_dim,
        train_loader: DataLoader,
        val_loader: DataLoader,
        callback=None,
        lr=1e-3,
        seed=42,
    ):
        super().__init__()
        self.c_hid = c_hid
        self.latent_dim = latent_dim
        self.lr = lr
        self.seed = seed
        # Create empty model. Note: no parameters yet
        self.model = (
            model  # = Autoencoder(c_hid=self.c_hid, latent_dim=self.latent_dim)
        )
        self.checkpoint_path = checkpoint_path
        # Prepare logging
        self.log_dir = os.path.join(checkpoint_path, f"cifar10_{self.latent_dim}")
        self.generate_callback = (
            callback  # GenerateCallback(self.exmp_imgs, every_n_epochs=50)
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_loader = val_loader
        self.logger = SummaryWriter(log_dir=self.log_dir)
        self.loss_fn = loss_fn
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model()

    @staticmethod
    def train(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        latent_dim: int,
        c_hid: int,
        loss_fn: Callable,
        lr: float,
        seed: int,
        num_epochs: int,
        checkpoint_path: str,
    ):
        # Create a trainer module with specified hyperparameters
        trainer = TrainerModule(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_path=checkpoint_path,
            c_hid=c_hid,
            latent_dim=latent_dim,
            loss_fn=loss_fn,
            lr=lr,
            seed=seed,
        )
        if not trainer.checkpoint_exists():  # Skip training if pretrained model exists
            trainer.train_model(num_epochs=num_epochs)
            trainer.load_model()
        else:
            trainer.load_model(pretrained=True)
        test_loss = trainer.eval_model(test_loader)
        # Bind parameters to model for easier inference
        trainer.model_bd = trainer.model.bind({"params": trainer.state.params})
        return trainer, test_loss

    def create_functions(self):
        # Training function
        def train_step(state, batch):
            loss_fn = lambda params: self.loss_fn(
                params, batch, self.model
            )  # mse_recon_loss(self.model, params, batch)
            loss, grads = jax.value_and_grad(loss_fn)(
                state.params
            )  # Get loss and gradients for loss
            state = state.apply_gradients(grads=grads)  # Optimizer update step
            return state, loss

        self.train_step = jax.jit(train_step)
        # Eval function
        def eval_step(state, batch):
            return self.loss_fn  # mse_recon_loss(self.model, state.params, batch)

        self.eval_step = jax.jit(eval_step)

    def init_model(self):
        # Initialize model
        rng = jax.random.PRNGKey(self.seed)
        rng, init_rng = jax.random.split(rng)
        params = self.model.init(init_rng, self.exmp_imgs)["params"]
        # Initialize learning rate schedule and optimizer
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=1e-3,
            warmup_steps=100,
            decay_steps=500 * len(self.train_loader),
            end_value=1e-5,
        )
        optimizer = optax.chain(
            optax.clip(1.0), optax.adam(lr_schedule)  # Clip gradients at 1
        )
        # Initialize training state
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=optimizer
        )

    def train_model(self, num_epochs=500):
        # Train model for defined number of epochs
        best_eval = 1e6
        for epoch_idx in tqdm(range(1, num_epochs + 1)):
            self.train_epoch(epoch=epoch_idx)
            if epoch_idx % 10 == 0:
                eval_loss = self.eval_model(self.val_loader)
                self.logger.add_scalar("val/loss", eval_loss, global_step=epoch_idx)
                if eval_loss < best_eval:
                    best_eval = eval_loss
                    self.save_model(step=epoch_idx)
                self.generate_callback.log_generations(
                    self.model, self.state, logger=self.logger, epoch=epoch_idx
                )
                self.logger.flush()

    def train_epoch(self, epoch):
        # Train model for one epoch, and log avg loss
        losses = []
        for batch in self.train_loader:
            self.state, loss = self.train_step(self.state, batch)
            losses.append(loss)
        losses_np = np.stack(jax.device_get(losses))
        avg_loss = losses_np.mean()
        self.logger.add_scalar("train/loss", avg_loss, global_step=epoch)

    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg loss
        losses = []
        batch_sizes = []
        for batch in data_loader:
            loss = self.eval_step(self.state, batch)
            losses.append(loss)
            batch_sizes.append(batch[0].shape[0])
        losses_np = np.stack(jax.device_get(losses))
        batch_sizes_np = np.stack(batch_sizes)
        avg_loss = (losses_np * batch_sizes_np).sum() / batch_sizes_np.sum()
        return avg_loss

    def save_model(self, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(
            ckpt_dir=self.log_dir,
            target=self.state.params,
            prefix=f"cifar10_{self.latent_dim}_",
            step=step,
        )

    def load_model(self, pretrained=False):
        # Load model. We use different checkpoint for pretrained models
        if not pretrained:
            params = checkpoints.restore_checkpoint(
                ckpt_dir=self.log_dir,
                target=self.state.params,
                prefix=f"cifar10_{self.latent_dim}_",
            )
        else:
            params = checkpoints.restore_checkpoint(
                ckpt_dir=os.path.join(
                    self.checkpoint_path, f"cifar10_{self.latent_dim}.ckpt"
                ),
                target=self.state.params,
            )
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=self.state.tx
        )

    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this autoencoder
        return os.path.isfile(
            os.path.join(self.checkpoint_path, f"cifar10_{self.latent_dim}.ckpt")
        )
