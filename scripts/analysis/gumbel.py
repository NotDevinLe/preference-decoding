import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.distributions import RelaxedBernoulli
class SparsePCALightning(pl.LightningModule):
    def __init__(self, input_dim, n_components, sparsity_weight=0.1,
                 temperature=1.0, hard_gumbel=True):
        super().__init__()
        self.save_hyperparameters()
        # Main networks
        self.encoder = torch.nn.Linear(input_dim, n_components, bias=False)
        self.decoder = torch.nn.Linear(n_components, input_dim, bias=False)
        # Learnable mask parameters - logits for Gumbel-Softmax
        # Shape: [input_dim] for shared feature masks across all components
        self.mask_logits = torch.nn.Parameter(
            torch.zeros(input_dim)
        )
        self.sparsity_weight = sparsity_weight
        self.temperature = temperature
        self.hard_gumbel = hard_gumbel
    def gumbel_softmax_sample(self, logits, temperature, hard=True):
        """
        Sample from Gumbel-Softmax distribution using torch.distributions.RelaxedBernoulli
        Args:
            logits: [n_components, input_dim] - unnormalized log probabilities
            temperature: float - temperature parameter
            hard: bool - if True, use straight-through estimator
        Returns:
            Binary mask with gradients
        """
        # Create RelaxedBernoulli distribution
        # logits are already in the right shape for Bernoulli
        dist = RelaxedBernoulli(temperature=temperature, logits=logits)
        if hard:
            # Sample and apply straight-through estimator
            soft_sample = dist.rsample()  # Reparameterized sample
            hard_sample = (soft_sample > 0.5).float()
            # Straight-through: hard forward, soft backward
            masks = (hard_sample - soft_sample).detach() + soft_sample
        else:
            # Just use the soft sample
            masks = dist.rsample()
        return masks
    def get_masks(self, training=True):
        """
        Get binary mask for input dimensions
        Returns:
            masks: [input_dim] - binary mask shared across all components
        """
        if training:
            # During training, sample from Gumbel-Softmax
            masks = self.gumbel_softmax_sample(
                self.mask_logits,
                self.temperature,
                hard=self.hard_gumbel
            )
        else:
            # During inference, use hard thresholding
            masks = (torch.sigmoid(self.mask_logits) > 0.5).float()
        return masks
    def forward(self, x):
        """Forward pass with masked input features"""
        # Get masks
        masks = self.get_masks(training=self.training)  # [input_dim]
        # Apply mask to input features
        x_masked = x * masks  # Broadcasting: [batch_size, input_dim] * [input_dim]
        # Encode masked input
        z = self.encoder(x_masked)
        # Decode back to full input space
        x_hat = self.decoder(z)
        return z, x_hat, masks
    def training_step(self, batch, batch_idx):
        # Unpack batch (TensorDataset returns a tuple)
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        # Normalize data across dimensions (features)
        x = F.normalize(x, p=2, dim=1)  # L2 normalize each sample
        z, x_hat, masks = self.forward(x)
        # Reconstruction loss
        recon_loss = F.mse_loss(x_hat, x)
        # Sparsity loss - encourage masks to be sparse
        # Use the probability of being "on" (sigmoid of logits)
        mask_probs = torch.sigmoid(self.mask_logits)
        sparsity_loss = mask_probs.mean()  # Penalize high probabilities
        # Total loss
        loss = recon_loss + self.sparsity_weight * sparsity_loss
        # Logging
        self.log('train_loss', loss, prog_bar=True)
        self.log('recon_loss', recon_loss, prog_bar=True)
        self.log('sparsity_loss', sparsity_loss, prog_bar=True)
        self.log('avg_mask_prob', mask_probs.mean(), prog_bar=True)
        self.log('active_features', masks.sum().float(), prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        # Unpack batch (TensorDataset returns a tuple)
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = F.normalize(x, p=2, dim=1)
        z, x_hat, masks = self.forward(x)
        val_loss = F.mse_loss(x_hat, x)
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
    def get_sparse_components(self):
        """
        Get the learned sparse components for analysis
        Returns:
            components: [n_components, input_dim] - component loadings
            masks: [input_dim] - binary mask showing which features are active
            mask_probs: [input_dim] - probability of each feature being active
        """
        with torch.no_grad():
            masks = self.get_masks(training=False)
            # Get decoder weights as components (transposed to [n_components, input_dim])
            components = self.decoder.weight.t()  # [n_components, input_dim]
            mask_probs = torch.sigmoid(self.mask_logits)
        return components.detach().cpu().numpy(), masks.detach().cpu().numpy(), mask_probs.detach().cpu().numpy()
# Example usage and training setup
def create_dataloader(X, batch_size=32, shuffle=True):
    """Create a simple DataLoader for training"""
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
# Temperature scheduling callback
class TemperatureScheduler(pl.Callback):
    def __init__(self, initial_temp=1.0, final_temp=0.1, anneal_rate=0.99):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.anneal_rate = anneal_rate
    def on_train_epoch_start(self, trainer, pl_module):
        # Anneal temperature over time
        current_temp = max(
            self.final_temp,
            self.initial_temp * (self.anneal_rate ** trainer.current_epoch)
        )
        pl_module.temperature = current_temp


if __name__ == '__main__':
    # Load reward matrix
    import numpy as np
    reward_data = np.load('data/reward-matrix.npz')

    A, P, Q = reward_data['Y'].shape
    
    # Reshape to (P*Q, A) - each row is one preference pair, columns are actions
    X = reward_data['Y'].transpose(1, 2, 0).reshape(P * Q, A)

    # Create model
    model = SparsePCALightning(
        input_dim=A,  # Number of actions/features
        n_components=10,
        sparsity_weight=0.003,
        temperature=1.0,
        hard_gumbel=True
    )
    # Create data loaders (80/20 train/val split)
    n_samples = X.shape[0]
    n_train = int(0.8 * n_samples)
    train_loader = create_dataloader(X[:n_train], batch_size=32)
    val_loader = create_dataloader(X[n_train:], batch_size=32, shuffle=False)
    # Create trainer with temperature scheduling
    temp_scheduler = TemperatureScheduler(initial_temp=1.0, final_temp=0.1)
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[temp_scheduler],
        accelerator='auto'
    )
    # Train
    trainer.fit(model, train_loader, val_loader)
    # Get learned sparse components
    components, masks, mask_probs = model.get_sparse_components()
    print(f"Selected features: {masks.sum()}/{len(masks)}")
    print(f"Average mask probability: {mask_probs.mean():.3f}")
    print(f"Active feature indices: {np.where(masks)[0]}")