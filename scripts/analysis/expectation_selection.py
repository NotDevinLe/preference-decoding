import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import wandb


class FeatureSelector(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        lr: float = 1e-3,
        sparsity_weight: float = 0.1,   # λ
        alpha: float = 1.0,              # weight for variance term
        weight_decay: float = 0.01,
        budget_weight: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        D = input_dim
        self.lr = lr
        self.sparsity_weight = sparsity_weight
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.k_target = max(1, int(0.1 * D))
        self.budget_weight = budget_weight


        p0 = torch.clamp(torch.tensor(float(self.k_target) / float(D)), 1e-4, 1 - 1e-4)
        init = torch.full((D,), torch.logit(p0))
        init = init + 1e-3 * torch.randn_like(init)
        self.mask_logits = nn.Parameter(init)

        # Linear reconstructor X -> X (acts like W)
        self.encoder = nn.Linear(D, D, bias=False)  # identity latent (keeps things simple)

    def forward(self, x):
        """Expectation masking forward (used for inference/analysis if needed)."""
        p = torch.sigmoid(self.mask_logits)          # [D]
        x_masked = x * p.view(1, -1)                 # [B,D]
        x_hat = self.encoder(x_masked)               # [B,D]
        return x_hat, p

    def _create_probability_plot(self, probs: np.ndarray):
        """Create a bar plot of feature selection probabilities p_i."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.bar(range(len(probs)), probs, alpha=0.8)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Selection Probability p_i')
        ax.set_title('Feature Selection Probabilities')
        ax.set_ylim(0.0, 1.0)
        plt.tight_layout()
        return fig

    def _create_weight_heatmap(self, weights: np.ndarray):
        """Create a heatmap of the encoder weights (D x D)."""
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(weights, cmap='RdBu_r', aspect='auto')
        ax.set_xlabel('Input Features')
        ax.set_ylabel('Output Features')
        ax.set_title('Encoder Weights Heatmap')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        return fig

    def get_loss(self, X: torch.Tensor, include_variance: bool = True):
        """
        Exact expectation (MSE): bias + variance, plus sparsity & budget.
        """
        B, D = X.shape
        p = torch.sigmoid(self.mask_logits)                         # [D]

        # Bias term: E[z ⊙ x] = p ⊙ x
        X_mask = X * p.view(1, -1)                                  # [B,D]
        X_hat = self.encoder(X_mask)                                # [B,D]
        bias = F.mse_loss(X_hat, X)                                 # mean over batch & dims

        var_term = X.new_tensor(0.0)
        if include_variance:
            # Effective linear map W_eff \in R^{D x D}; here it's just encoder.weight
            # encoder.weight shape is [D, D] (out x in)
            W_eff = self.encoder.weight                             # [D,D]
            Wi2 = (W_eff ** 2).sum(dim=0)                           # sum_j w_{j,i}^2 -> [D]
            x2 = (X ** 2).mean(dim=0)                               # E[x_i^2] over batch -> [D]
            var_term = (Wi2 * p * (1 - p) * x2).sum()

        exp_L0 = p.sum()
        budget_pen = (exp_L0 - self.k_target) ** 2

        loss = bias + self.alpha * var_term + self.sparsity_weight * exp_L0 + self.budget_weight * budget_pen
        logs = {
            "loss": loss,
            "bias": bias.detach(),
            "var_term": var_term.detach(),
            "exp_L0": exp_L0.detach(),
            "budget_pen": budget_pen.detach(),
            "avg_p": p.mean().detach(),
            "sum_p": exp_L0.detach()
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch

        loss, logs = self.get_loss(x, include_variance=True)

        # Lightning logs
        self.log("train_loss", logs["loss"], prog_bar=True)
        self.log("train_bias", logs["bias"])
        self.log("train_var_term", logs["var_term"])
        self.log("train_exp_L0", logs["exp_L0"])
        self.log("train_avg_p", logs["avg_p"])
        self.log("train_sum_p", logs["sum_p"])
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch

        loss, logs = self.get_loss(x, include_variance=True)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_bias", logs["bias"])
        self.log("val_var_term", logs["var_term"])
        self.log("val_exp_L0", logs["exp_L0"])
        self.log("val_budget_pen", logs["budget_pen"])
        self.log("val_avg_p", logs["avg_p"])
        self.log("val_sum_p", logs["sum_p"])
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def on_train_end(self):
        """Log final probability plot, weights heatmap, and summary stats to wandb."""
        with torch.no_grad():
            p = torch.sigmoid(self.mask_logits).detach().cpu().numpy()
            weights = self.encoder.weight.detach().cpu().numpy()

            prob_fig = self._create_probability_plot(p)
            wandb.log({"probabilities": wandb.Image(prob_fig)})
            plt.close(prob_fig)

            w_fig = self._create_weight_heatmap(weights)
            wandb.log({"encoder_weights": wandb.Image(w_fig)})
            plt.close(w_fig)

            wandb.log({
                "final_avg_p": float(p.mean()),
                "final_std_p": float(p.std()),
                "final_sum_p": float(p.sum()),
                "final_num_features": int(p.size),
            })

    def get_selected_features(self, threshold=0.5, topk: int | None = None):
        """
        After training, return selected feature indices.
        If topk is provided, take the top-k by probability; otherwise threshold at 0.5.
        """
        p = torch.sigmoid(self.mask_logits).detach().cpu().numpy()
        if topk is not None:
            k = min(topk, p.size)
            idx = np.argpartition(-p, k-1)[:k]
            idx = idx[np.argsort(-p[idx])]
            return idx, p
        else:
            idx = np.where(p > threshold)[0]
            return idx, p


# -------------------------
# Example usage
# -------------------------
def create_dataloader(X, batch_size=32, shuffle=True):
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    np.random.seed(0)
    X = torch.load('data/rewards.pt')

    mean = X.mean(dim=0)
    var = X.var(dim=0)

    X = (X - mean) / np.sqrt(var + 1e-6)

    train_loader = create_dataloader(X[:int(0.8*X.shape[0])], batch_size=2048)
    val_loader = create_dataloader(X[int(0.8*X.shape[0]):], batch_size=2048, shuffle=False)

    sparsity_weights = [5e-8, 7e-8, 1e-7, 3e-7]
    learning_rates = [1e-2, 1e-3, 1e-4]
    weight_decays = [0.01, 0.001, 0.0001]
    budget_weights = [0.1, 0.01, 0.001]
    
    for sparsity_weight in sparsity_weights:
        for learning_rate in learning_rates:
            for weight_decay in weight_decays:
                for budget_weight in budget_weights:
                    wandb.init(
                        project="expectation-selection",
                        name="feature-selector",
                        config={
                            "model": "FeatureSelector",
                            "lr": learning_rate,
                            "sparsity_weight": sparsity_weight,
                            "alpha": 1e-6,
                            "batch_size": 2048,
                            "max_epochs": 400,
                            "weight_decay": weight_decay,
                            "budget_weight": budget_weight
                        }
                    )

                    # Log dataset statistics
                    wandb.log({
                        "data_shape": tuple(X.shape),
                        "data_mean": float(X.mean()),
                        "data_std": float(X.std()),
                        "data_min": float(X.min()),
                        "data_max": float(X.max()),
                        "budget_weight": budget_weight
                    })

                    model = FeatureSelector(
                        input_dim=X.shape[1],
                        lr=learning_rate,
                        sparsity_weight=sparsity_weight,   # λ
                        alpha=1e-6,               # variance term weight
                        weight_decay=weight_decay,
                        budget_weight=budget_weight
                    )

                    trainer = pl.Trainer(
                        max_epochs=400,
                        accelerator='cpu',
                        logger=pl.loggers.WandbLogger(project="expectation-selection")
                    )
                    trainer.fit(model, train_loader, val_loader)

                    # Hard subset by threshold
                    idx_thr, p = model.get_selected_features(threshold=0.5)
                    print(p.mean())
                    print(p.std())
                    print("Threshold>0.5 selected:", idx_thr, "count:", len(idx_thr))

                    # Log final artifacts
                    wandb.log({
                        "final_avg_p": float(p.mean()),
                        "final_std_p": float(p.std()),
                        "threshold_count": int(len(idx_thr)),
                        "survived_features": idx_thr,
                        "budget_weight": budget_weight
                    })

                    wandb.finish()
