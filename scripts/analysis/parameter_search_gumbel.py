from scripts.analysis.gumbel import SparsePCALightning, create_dataloader, TemperatureScheduler
import numpy as np
import wandb
import pytorch_lightning as pl
import torch.nn.functional as F
import json
import torch
import torch.nn.functional as F

lrs = [1e-2, 1e-3]
sparsity_weights = [1e-3, 1e-4, 1e-5]

results = []

for sparsity_weight_val in sparsity_weights:
    for lr_val in lrs:
        print(f"Training with sparsity weight {sparsity_weight_val}, lr {lr_val}")
        
        # Ensure parameters are proper scalar types
        lr = float(lr_val)
        sparsity_weight = float(sparsity_weight_val)
        n_components = 50

        reward_matrix = torch.load('rewards/rewards.pt').numpy()

        model = SparsePCALightning(
            input_dim=reward_matrix.shape[1],
            n_components=n_components,
            lr=lr,
            sparsity_weight=sparsity_weight,
            temperature=1.0,
            hard_gumbel=True
        )

        train_loader = create_dataloader(reward_matrix[:int(0.8 * reward_matrix.shape[0])], batch_size=512, num_workers=8)
        val_loader = create_dataloader(reward_matrix[int(0.8 * reward_matrix.shape[0]):], batch_size=512, num_workers=8)

        trainer = pl.Trainer(
            max_epochs=100,
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

        val_metrics = trainer.validate(model, dataloaders=val_loader, verbose=False)
        val_loss = float(val_metrics[0].get("val_loss", np.nan))

        components, mask, _ = model.get_sparse_components()
        results.append([val_loss, sparsity_weight_val, lr, mask.tolist()])

        wandb.finish()

results.sort()
with open('data/parameter_search_gumbel.json', 'w') as f:
    json.dump(results, f)

