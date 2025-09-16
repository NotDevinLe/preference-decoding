from scripts.analysis.gumbel import SparsePCALightning, create_dataloader, TemperatureScheduler
import numpy as np
import wandb
import pytorch_lightning as pl
import torch.nn.functional as F
import json
import torch
import torch.nn.functional as F

lrs = 10 ** np.linspace(-6, -3, 10)
sparsity_weights = 10 ** np.linspace(-6, -2, 10)
components = [5, 10, 20, 50, 100]

results = []

for lr in lrs:
    for sparsity_weight in sparsity_weights:
        for n_components in components:

            wandb.init(
                project="gumbel",
                name=f"lr_{lr}_sparsity_weight_{sparsity_weight}_n_components_{n_components}",
                config={
                    "model": "SparsePCALightning",
                    "n_components": n_components,
                    "sparsity_weight": sparsity_weight,
                    "lr": lr,
                    "initial_temperature": 1.0,
                    "final_temperature": 0.1,
                    "anneal_rate": 0.99,
                    "batch_size": 32,
                    "max_epochs": 100
                }
            )

            reward_matrix = np.load('data/reward_matrix_flexible.npz')['Y_chosen']

            model = SparsePCALightning(
                input_dim=reward_matrix.shape[1],
                n_components=n_components,
                sparsity_weight=sparsity_weight,
                temperature=1.0,
                hard_gumbel=True
            )

            train_loader = create_dataloader(reward_matrix[:int(0.8 * reward_matrix.shape[0])], batch_size=32)
            val_loader = create_dataloader(reward_matrix[int(0.8 * reward_matrix.shape[0]):], batch_size=32)

            trainer = pl.Trainer(
                max_epochs=100,
                accelerator='auto',
                logger=pl.loggers.WandbLogger(project="gumbel"),
                callbacks=[TemperatureScheduler(initial_temp=1.0, final_temp=0.1)]
            )

            trainer.fit(model, train_loader, val_loader)
            components, masks, mask_probs = model.get_sparse_components()

            wandb.finish()

            results.append({
                "lr": lr,
                "sparsity_weight": sparsity_weight,
                "n_selected": masks.sum(),
                "sparsity_ratio": float(masks.sum() / len(masks)),
                "reconstruction_error": float(F.mse_loss(model.forward(torch.from_numpy(reward_matrix))[1], torch.from_numpy(reward_matrix)).item()),
                "components": components.tolist(),
                "masks": masks.tolist(),
                "mask_probs": mask_probs,
                "n_components": n_components
            })

with open('data/parameter_search_gumbel.json', 'w') as f:
    json.dump(results, f)