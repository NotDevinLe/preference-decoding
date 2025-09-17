import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.distributions import RelaxedBernoulli
import wandb
import matplotlib.pyplot as plt
import numpy as np

class SparsePCALightning(pl.LightningModule):
    def __init__(self, input_dim, n_components, lr = 1e-3, sparsity_weight=0.1,
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
        self.lr = lr
        self.sparsity_weight = sparsity_weight
        self.temperature = temperature
        self.hard_gumbel = hard_gumbel
        
        # Initialize wandb logging
        self.log_components_plot = True
        self.log_mask_plot = True
        
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
        
    def create_component_heatmap(self, components):
        """Create a heatmap of the learned components"""
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(components, cmap='RdBu_r', aspect='auto')
        ax.set_xlabel('Features/Dimensions')
        ax.set_ylabel('Components')
        ax.set_title('Learned Sparse PCA Components')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        return fig
        
    def create_mask_plot(self, masks, mask_probs):
        """Create a plot showing the learned feature masks"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Binary masks
        ax1.bar(range(len(masks)), masks, alpha=0.7, color='blue')
        ax1.set_xlabel('Feature Index')
        ax1.set_ylabel('Binary Mask (0/1)')
        ax1.set_title('Learned Binary Feature Masks')
        ax1.set_ylim(-0.1, 1.1)
        
        # Mask probabilities
        ax2.bar(range(len(mask_probs)), mask_probs, alpha=0.7, color='red')
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Mask Probability')
        ax2.set_title('Feature Mask Probabilities')
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        return fig
        
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
        self.log('temperature', self.temperature, prog_bar=True)
        
        # Log component and mask plots every 10 epochs
        if batch_idx == 0 and self.current_epoch % 10 == 0:
            with torch.no_grad():
                components, masks, mask_probs = self.get_sparse_components()
                
                if self.log_components_plot:
                    comp_fig = self.create_component_heatmap(components)
                    wandb.log({"components_heatmap": wandb.Image(comp_fig)})
                    plt.close(comp_fig)
                
                if self.log_mask_plot:
                    mask_fig = self.create_mask_plot(masks, mask_probs)
                    wandb.log({"feature_masks": wandb.Image(mask_fig)})
                    plt.close(mask_fig)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        # Unpack batch (TensorDataset returns a tuple)
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = F.normalize(x, p=2, dim=1)
        z, x_hat, masks = self.forward(x)
        val_loss = F.mse_loss(x_hat, x)
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss
        
    def on_train_end(self):
        """Log final results and create summary plots"""
        with torch.no_grad():
            components, masks, mask_probs = self.get_sparse_components()
            
            # Log final component heatmap
            comp_fig = self.create_component_heatmap(components)
            wandb.log({"final_components": wandb.Image(comp_fig)})
            plt.close(comp_fig)
            
            # Log final mask plot
            mask_fig = self.create_mask_plot(masks, mask_probs)
            wandb.log({"final_masks": wandb.Image(mask_fig)})
            plt.close(mask_fig)
            
            # Log final statistics
            wandb.log({
                "final_active_features": int(masks.sum()),
                "final_total_features": len(masks),
                "final_sparsity_ratio": float(masks.sum() / len(masks)),
                "final_avg_mask_prob": float(mask_probs.mean())
            })
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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
    # Initialize wandb


    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sparsity_weight", type=float, default=0.0)
    args = parser.parse_args()
    sparsity_weight = args.sparsity_weight
    wandb.init(
        project="sparse-pca-analysis",
        name="gumbel-sparse-pca",
        config={
            "model": "SparsePCALightning",
            "n_components": 10,
            "lr": 1e-3,
            "sparsity_weight": sparsity_weight,
            "initial_temperature": 1.0,
            "final_temperature": 0.1,
            "anneal_rate": 0.99,
            "batch_size": 32,
            "max_epochs": 100
        }
    )
    
    # Load reward matrix
    reward_data = np.load('reward_matrix_flexible.npz')

    # Reshape to (P*Q, A) - each row is one preference pair, columns are actions
    X = reward_data['Y_chosen']
    
    # Log data statistics
    wandb.log({
        "data_shape": X.shape,
        "data_mean": float(X.mean()),
        "data_std": float(X.std()),
        "data_min": float(X.min()),
        "data_max": float(X.max())
    })

    # Create model
    model = SparsePCALightning(
        input_dim=X.shape[1],  # Number of actions/features
        n_components=10,
        lr=1e-3,
        sparsity_weight=sparsity_weight,
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
        accelerator='auto',
        logger=pl.loggers.WandbLogger(project="sparse-pca-analysis")
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Get learned sparse components
    components, masks, mask_probs = model.get_sparse_components()
    
    # Log final results
    print(f"Selected features: {masks.sum()}/{len(masks)}")
    print(f"Average mask probability: {mask_probs.mean():.3f}")
    print(f"Active feature indices: {np.where(masks)[0]}")
    
    # Create and log final summary plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Component heatmap
    im1 = ax1.imshow(components, cmap='RdBu_r', aspect='auto')
    ax1.set_xlabel('Features/Dimensions')
    ax1.set_ylabel('Components')
    ax1.set_title('Final Learned Components')
    plt.colorbar(im1, ax=ax1)
    
    # Feature masks
    ax2.bar(range(len(masks)), masks, alpha=0.7, color='green')
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Binary Mask (0/1)')
    ax2.set_title(f'Final Feature Masks ({int(masks.sum())}/{len(masks)} active)')
    ax2.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    wandb.log({"final_summary": wandb.Image(fig)})
    plt.close(fig)
    
    # Finish wandb run
    wandb.finish()