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

for lr_val in lrs:
    for sparsity_weight_val in sparsity_weights:
        for n_components_val in components:
            
            # Ensure parameters are proper scalar types
            lr = float(lr_val)
            sparsity_weight = float(sparsity_weight_val)
            n_components = int(n_components_val)

            wandb.init(
                project="gumbel",
                name=f"lr_{lr:.2e}_sparsity_weight_{sparsity_weight:.2e}_n_components_{n_components}",
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
                lr=lr,
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

            model.eval()

            reconstruction_error = F.mse_loss(model.forward(torch.from_numpy(reward_matrix))[1], torch.from_numpy(reward_matrix)).item()
    
            results.append({
                "lr": float(lr),
                "sparsity_weight": float(sparsity_weight),
                "n_selected": int(masks.sum()),
                "sparsity_ratio": float(masks.sum() / len(masks)),
                "reconstruction_error": float(reconstruction_error),
                "components": components.tolist(),
                "masks": masks.tolist(),
                "mask_probs": mask_probs.tolist(),
                "n_components": int(n_components)
            })
            
            print(f"Completed: lr={lr:.2e}, sparsity_weight={sparsity_weight:.2e}, n_components={n_components}")
            print(f"  Selected features: {int(masks.sum())}/{len(masks)} ({100*masks.sum()/len(masks):.1f}%)")
            print(f"  Reconstruction error: {reconstruction_error:.4f}")
            print()

with open('data/parameter_search_gumbel.json', 'w') as f:
    json.dump(results, f)