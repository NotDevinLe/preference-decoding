import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import matplotlib.pyplot as plt


class FeatureSelector(pl.LightningModule):
    def __init__(self, input_dim, lr=1e-3, sparsity_weight=0.1, target_k=None):
        """
        Args:
            input_dim: number of original attributes (D)
            lr: learning rate
            sparsity_weight: weight for sparsity penalty
            target_k: optional target number of selected features
        """
        super().__init__()
        self.save_hyperparameters()

        D = input_dim
        k = target_k if target_k is not None else max(1, int(0.1 * D))
        self.k_target = k

        # Initialize logits so p ~ k/D initially
        init = torch.full((D,), torch.logit(torch.tensor(float(k) / float(D))))
        self.mask_logits = nn.Parameter(init)

        # Simple reconstruction model: W maps masked x -> reconstructed x
        self.W = nn.Linear(D, D, bias=False)

        self.lr = lr
        self.sparsity_weight = sparsity_weight

    def forward(self, x):
        """
        Forward pass:
        - Apply soft mask probs to input
        - Reconstruct input
        """
        probs = torch.sigmoid(self.mask_logits)   # [D]
        x_masked = x * probs.view(1, -1)          # expectation masking
        x_hat = self.W(x_masked)
        return x_hat, probs

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = F.normalize(x, p=2, dim=1)

        x_hat, probs = self.forward(x)

        # Reconstruction loss
        recon_loss = F.mse_loss(x_hat, x)

        # Sparsity loss = expected number of active features
        exp_L0 = probs.sum()
        budget_loss = (exp_L0 - self.k_target) ** 2  # optional

        loss = recon_loss + self.sparsity_weight * exp_L0 + 1e-3 * budget_loss

        self.log('train_loss', loss, prog_bar=True)
        self.log('recon_loss', recon_loss)
        self.log('sparsity_loss', exp_L0)
        self.log('avg_mask_prob', probs.mean())
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = F.normalize(x, p=2, dim=1)
        x_hat, _ = self.forward(x)
        val_loss = F.mse_loss(x_hat, x)
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_selected_features(self, threshold=0.5):
        """Return indices of selected attributes after training."""
        probs = torch.sigmoid(self.mask_logits).detach().cpu().numpy()
        selected = np.where(probs > threshold)[0]
        return selected, probs


# -------------------------
# Example usage
# -------------------------
def create_dataloader(X, batch_size=32, shuffle=True):
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    # Fake dataset: 100 samples, 20 attributes
    np.random.seed(0)
    X = np.random.randn(100, 20)

    train_loader = create_dataloader(X[:80], batch_size=16)
    val_loader = create_dataloader(X[80:], batch_size=16, shuffle=False)

    model = FeatureSelector(
        input_dim=X.shape[1],
        lr=1e-2,
        sparsity_weight=0.05,
        target_k=5  # want ~5 features kept
    )

    trainer = pl.Trainer(max_epochs=50, accelerator='cpu')
    trainer.fit(model, train_loader, val_loader)

    # After training: get selected features
    selected, probs = model.get_selected_features(threshold=0.5)
    print("Selected feature indices:", selected)
    print("Mask probabilities:", probs.round(3))
