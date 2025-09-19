from scripts.analysis.gumbel import SparsePCALightning, create_dataloader, TemperatureScheduler
import numpy as np
import wandb
import pytorch_lightning as pl
import torch.nn.functional as F
import json
import torch
import torch.nn.functional as F

lrs = 10 ** np.linspace(-6, -2, 5)
sparsity_weights = [1e-5]
component_sizes = [5, 10, 20, 50, 100]
print(f"Component sizes list: {component_sizes}")
print(f"Component sizes types: {[type(c) for c in component_sizes]}")

best = {}

for sparsity_weight_val in sparsity_weights:
    best_loss = float('inf')
    best_model = None
    for lr_val in lrs:
        for n_components_val in component_sizes:
            print(f"Training with sparsity weight {sparsity_weight_val}, lr {lr_val}, and n_components {n_components_val}")
            
            # Ensure parameters are proper scalar types
            lr = float(lr_val)
            sparsity_weight = float(sparsity_weight_val)
            n_components = int(n_components_val)

            reward_matrix = np.load('reward_matrix_flexible.npz')['Y_chosen']

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
                max_epochs=500,
                accelerator='auto',
                logger=pl.loggers.WandbLogger(project="gumbel",
                    name=f"lr_{lr:.2e}_sparsity_weight_{sparsity_weight:.2e}_n_components_{n_components}",
                    config={
                        "model": "SparsePCALightning",
                        "n_components": n_components,
                        "sparsity_weight": sparsity_weight,
                        "lr": lr,
                    }),
                callbacks=[TemperatureScheduler(initial_temp=1.0, final_temp=0.1)]
            )

            trainer.fit(model, train_loader, val_loader)

            model.eval()

            validation_matrix = torch.from_numpy(reward_matrix[int(0.8) * reward_matrix.shape[0]:])
            validation_loss = model.validation_step(validation_matrix, 0, log=False).item()

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_model = model

            wandb.finish()
    best[sparsity_weight_val] = best_model

component_results = {}

for sparsity_weight_val in sparsity_weights:
    components, _, _ = best[sparsity_weight_val].get_sparse_components()
    component_results[sparsity_weight_val] = components

with open('data/parameter_search_gumbel.json', 'w') as f:
    json.dump(component_results, f)

