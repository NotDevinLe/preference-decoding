import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import RelaxedBernoulli
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class SparsePCAEvaluator:
    """Comprehensive evaluation suite for Sparse PCA approach"""
    
    def __init__(self, model, X_train, X_val, attribute_names=None):
        self.model = model
        self.X_train = X_train
        self.X_val = X_val
        self.attribute_names = attribute_names or [f"Attr_{i}" for i in range(X_train.shape[1])]
        
    def evaluate_selection_quality(self):
        """Evaluate the quality of sparse feature selection"""
        components, masks, mask_probs = self.model.get_sparse_components()
        
        results = {
            'n_selected': int(masks.sum()),
            'sparsity_ratio': 1 - (masks.sum() / len(masks)),
            'selected_indices': np.where(masks)[0].tolist(),
            'selected_names': [self.attribute_names[i] for i in np.where(masks)[0]],
            'mask_probabilities': mask_probs,
            'components': components
        }
        
        # Reconstruction quality on validation set
        self.model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(self.X_val)
            X_val_norm = F.normalize(X_val_tensor, p=2, dim=1)
            z, x_hat, _ = self.model(X_val_norm)
            
            # Reconstruction error
            recon_error = F.mse_loss(x_hat, X_val_norm).item()
            
            # Explained variance
            var_original = X_val_norm.var(dim=0).sum().item()
            var_residual = (X_val_norm - x_hat).var(dim=0).sum().item()
            explained_var = 1 - (var_residual / var_original)
            
        results['reconstruction_error'] = recon_error
        results['explained_variance'] = explained_var
        
        return results
    
    def compare_with_pca(self, n_components=None):
        """Compare sparse selection with standard PCA"""
        if n_components is None:
            n_components = self.model.hparams.n_components
            
        # Standard PCA
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_val_scaled = scaler.transform(self.X_val)
        
        pca = PCA(n_components=n_components)
        pca.fit(X_train_scaled)
        
        # PCA reconstruction
        X_val_pca = pca.transform(X_val_scaled)
        X_val_recon_pca = pca.inverse_transform(X_val_pca)
        pca_recon_error = np.mean((X_val_scaled - X_val_recon_pca) ** 2)
        
        # Sparse PCA reconstruction
        sparse_results = self.evaluate_selection_quality()
        
        return {
            'pca_explained_variance': pca.explained_variance_ratio_.sum(),
            'pca_reconstruction_error': pca_recon_error,
            'sparse_explained_variance': sparse_results['explained_variance'],
            'sparse_reconstruction_error': sparse_results['reconstruction_error'],
            'pca_components': pca.components_,
            'sparse_components': sparse_results['components'],
            'sparse_n_features': sparse_results['n_selected'],
            'total_features': len(self.attribute_names)
        }
    
    def stability_analysis(self, n_runs=10, noise_std=0.01):
        """Test stability of feature selection across multiple runs with noise"""
        selected_features_runs = []
        
        for run in range(n_runs):
            # Add small amount of noise to training data
            X_noisy = self.X_train + np.random.normal(0, noise_std, self.X_train.shape)
            
            # Create and train new model
            model_copy = SparsePCALightning(
                input_dim=self.model.hparams.input_dim,
                n_components=self.model.hparams.n_components,
                sparsity_weight=self.model.hparams.sparsity_weight,
                temperature=self.model.hparams.temperature,
                hard_gumbel=self.model.hparams.hard_gumbel
            )
            
            # Quick training (fewer epochs for stability test)
            train_loader = create_dataloader(X_noisy, batch_size=32)
            trainer = pl.Trainer(max_epochs=20, enable_progress_bar=False, logger=False)
            trainer.fit(model_copy, train_loader)
            
            # Get selected features
            _, masks, _ = model_copy.get_sparse_components()
            selected_features_runs.append(np.where(masks)[0])
        
        # Analyze stability
        all_features = set()
        for features in selected_features_runs:
            all_features.update(features)
        
        feature_stability = {}
        for feature in all_features:
            count = sum(1 for features in selected_features_runs if feature in features)
            feature_stability[feature] = count / n_runs
        
        # Jaccard similarity between runs
        jaccard_similarities = []
        for i in range(n_runs):
            for j in range(i+1, n_runs):
                set_i = set(selected_features_runs[i])
                set_j = set(selected_features_runs[j])
                jaccard = len(set_i & set_j) / len(set_i | set_j) if len(set_i | set_j) > 0 else 0
                jaccard_similarities.append(jaccard)
        
        return {
            'selected_features_runs': selected_features_runs,
            'feature_stability': feature_stability,
            'avg_jaccard_similarity': np.mean(jaccard_similarities),
            'std_jaccard_similarity': np.std(jaccard_similarities),
            'most_stable_features': sorted(feature_stability.items(), key=lambda x: x[1], reverse=True)
        }
    
    def hyperparameter_sensitivity(self, sparsity_weights=None, temperatures=None):
        """Test sensitivity to hyperparameters"""
        if sparsity_weights is None:
            sparsity_weights = [0.001, 0.01, 0.1, 0.5, 1.0]
        if temperatures is None:
            temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        results = {'sparsity_weight_sweep': [], 'temperature_sweep': []}
        
        # Sparsity weight sweep
        for sparsity_weight in sparsity_weights:
            model = SparsePCALightning(
                input_dim=self.model.hparams.input_dim,
                n_components=self.model.hparams.n_components,
                sparsity_weight=sparsity_weight,
                temperature=1.0,
                hard_gumbel=True
            )
            
            train_loader = create_dataloader(self.X_train, batch_size=32)
            val_loader = create_dataloader(self.X_val, batch_size=32, shuffle=False)
            trainer = pl.Trainer(max_epochs=50, enable_progress_bar=False, logger=False)
            trainer.fit(model, train_loader, val_loader)
            
            eval_results = SparsePCAEvaluator(model, self.X_train, self.X_val, self.attribute_names).evaluate_selection_quality()
            results['sparsity_weight_sweep'].append({
                'sparsity_weight': sparsity_weight,
                'n_selected': eval_results['n_selected'],
                'reconstruction_error': eval_results['reconstruction_error'],
                'explained_variance': eval_results['explained_variance']
            })
        
        # Temperature sweep (initial temperature)
        for temp in temperatures:
            model = SparsePCALightning(
                input_dim=self.model.hparams.input_dim,
                n_components=self.model.hparams.n_components,
                sparsity_weight=0.01,
                temperature=temp,
                hard_gumbel=True
            )
            
            train_loader = create_dataloader(self.X_train, batch_size=32)
            val_loader = create_dataloader(self.X_val, batch_size=32, shuffle=False)
            trainer = pl.Trainer(max_epochs=50, enable_progress_bar=False, logger=False)
            trainer.fit(model, train_loader, val_loader)
            
            eval_results = SparsePCAEvaluator(model, self.X_train, self.X_val, self.attribute_names).evaluate_selection_quality()
            results['temperature_sweep'].append({
                'temperature': temp,
                'n_selected': eval_results['n_selected'],
                'reconstruction_error': eval_results['reconstruction_error'],
                'explained_variance': eval_results['explained_variance']
            })
        
        return results
    
    def plot_results(self, save_path='sparse_pca_analysis.png'):
        """Create comprehensive plots of results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Feature selection probabilities
        components, masks, mask_probs = self.model.get_sparse_components()
        selected_idx = np.where(masks)[0]
        
        axes[0,0].bar(range(len(mask_probs)), mask_probs)
        axes[0,0].axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
        axes[0,0].set_title('Feature Selection Probabilities')
        axes[0,0].set_xlabel('Feature Index')
        axes[0,0].set_ylabel('Selection Probability')
        
        # Highlight selected features
        for idx in selected_idx:
            axes[0,0].bar(idx, mask_probs[idx], color='red', alpha=0.7)
        
        # 2. Component loadings heatmap
        im = axes[0,1].imshow(components, aspect='auto', cmap='RdBu_r')
        axes[0,1].set_title('Learned Components')
        axes[0,1].set_xlabel('Feature Index')
        axes[0,1].set_ylabel('Component')
        plt.colorbar(im, ax=axes[0,1])
        
        # 3. Reconstruction error over training
        # (Would need to modify model to track this)
        axes[0,2].text(0.5, 0.5, 'Training curves\n(requires logging)', 
                      ha='center', va='center', transform=axes[0,2].transAxes)
        axes[0,2].set_title('Training Progress')
        
        # 4. Selected vs non-selected feature importance
        feature_importance = np.abs(components).mean(axis=0)
        selected_importance = feature_importance[masks.astype(bool)]
        unselected_importance = feature_importance[~masks.astype(bool)]
        
        axes[1,0].hist(unselected_importance, alpha=0.7, label='Unselected', bins=20)
        axes[1,0].hist(selected_importance, alpha=0.7, label='Selected', bins=20)
        axes[1,0].set_title('Feature Importance Distribution')
        axes[1,0].set_xlabel('Average Absolute Loading')
        axes[1,0].set_ylabel('Count')
        axes[1,0].legend()
        
        # 5. Sparsity vs reconstruction trade-off
        sparsity_results = self.hyperparameter_sensitivity()
        sparsity_data = sparsity_results['sparsity_weight_sweep']
        
        sparsity_weights = [r['sparsity_weight'] for r in sparsity_data]
        n_selected = [r['n_selected'] for r in sparsity_data]
        recon_errors = [r['reconstruction_error'] for r in sparsity_data]
        
        ax1 = axes[1,1]
        ax2 = ax1.twinx()
        
        ax1.plot(sparsity_weights, n_selected, 'b-o', label='Features Selected')
        ax2.plot(sparsity_weights, recon_errors, 'r-s', label='Reconstruction Error')
        
        ax1.set_xlabel('Sparsity Weight')
        ax1.set_ylabel('Number of Features Selected', color='b')
        ax2.set_ylabel('Reconstruction Error', color='r')
        ax1.set_xscale('log')
        axes[1,1].set_title('Sparsity vs Quality Trade-off')
        
        # 6. Comparison with PCA
        pca_comparison = self.compare_with_pca()
        
        methods = ['Standard PCA', 'Sparse PCA']
        explained_vars = [pca_comparison['pca_explained_variance'], 
                         pca_comparison['sparse_explained_variance']]
        recon_errors = [pca_comparison['pca_reconstruction_error'],
                       pca_comparison['sparse_reconstruction_error']]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax1 = axes[1,2]
        ax2 = ax1.twinx()
        
        ax1.bar(x - width/2, explained_vars, width, label='Explained Variance', alpha=0.8)
        ax2.bar(x + width/2, recon_errors, width, label='Reconstruction Error', alpha=0.8, color='orange')
        
        ax1.set_ylabel('Explained Variance')
        ax2.set_ylabel('Reconstruction Error')
        ax1.set_xlabel('Method')
        axes[1,2].set_title('PCA vs Sparse PCA')
        axes[1,2].set_xticks(x)
        axes[1,2].set_xticklabels(methods)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

def create_dataloader(X, batch_size=32, shuffle=True):
    """Create a simple DataLoader for training"""
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Usage example and improvements
def evaluate_sparse_pca_model(model, X_train, X_val, attribute_names):
    """Complete evaluation pipeline"""
    
    evaluator = SparsePCAEvaluator(model, X_train, X_val, attribute_names)
    
    print("=== Sparse PCA Evaluation Results ===")
    
    # 1. Basic selection quality
    selection_results = evaluator.evaluate_selection_quality()
    print(f"Selected {selection_results['n_selected']} out of {len(attribute_names)} features")
    print(f"Sparsity: {selection_results['sparsity_ratio']:.2%}")
    print(f"Reconstruction error: {selection_results['reconstruction_error']:.4f}")
    print(f"Explained variance: {selection_results['explained_variance']:.4f}")
    
    print("\nSelected attributes:")
    for name in selection_results['selected_names']:
        print(f"  - {name}")
    
    # 2. Compare with PCA
    print("\n=== Comparison with Standard PCA ===")
    pca_comparison = evaluator.compare_with_pca()
    print(f"PCA explained variance: {pca_comparison['pca_explained_variance']:.4f}")
    print(f"Sparse PCA explained variance: {pca_comparison['sparse_explained_variance']:.4f}")
    print(f"PCA uses all {pca_comparison['total_features']} features")
    print(f"Sparse PCA uses only {pca_comparison['sparse_n_features']} features")
    
    # 3. Stability analysis
    print("\n=== Stability Analysis ===")
    stability_results = evaluator.stability_analysis(n_runs=5)  # Reduced for speed
    print(f"Average Jaccard similarity: {stability_results['avg_jaccard_similarity']:.3f} ± {stability_results['std_jaccard_similarity']:.3f}")
    
    print("Most stable features:")
    for feature_idx, stability in stability_results['most_stable_features'][:5]:
        print(f"  {attribute_names[feature_idx]}: {stability:.2f}")
    
    # 4. Generate plots
    evaluator.plot_results()
    
    return {
        'selection_results': selection_results,
        'pca_comparison': pca_comparison,
        'stability_results': stability_results
    }

# Model improvements
class ImprovedSparsePCA(pl.LightningModule):
    """Enhanced version with additional features"""
    
    def __init__(self, input_dim, n_components, sparsity_weight=0.1,
                 temperature=1.0, hard_gumbel=True, orthogonal_penalty=0.01,
                 diversity_penalty=0.01):
        super().__init__()
        self.save_hyperparameters()
        
        # Main networks
        self.encoder = torch.nn.Linear(input_dim, n_components, bias=False)
        self.decoder = torch.nn.Linear(n_components, input_dim, bias=False)
        
        # Learnable mask parameters
        self.mask_logits = torch.nn.Parameter(torch.zeros(input_dim))
        
        # Hyperparameters
        self.sparsity_weight = sparsity_weight
        self.temperature = temperature
        self.hard_gumbel = hard_gumbel
        self.orthogonal_penalty = orthogonal_penalty
        self.diversity_penalty = diversity_penalty
        
        # Track metrics
        self.train_losses = []
        self.val_losses = []
    
    def get_masks(self, training=True):
        """Same as before"""
        if training:
            dist = RelaxedBernoulli(temperature=self.temperature, logits=self.mask_logits)
            if self.hard_gumbel:
                soft_sample = dist.rsample()
                hard_sample = (soft_sample > 0.5).float()
                masks = (hard_sample - soft_sample).detach() + soft_sample
            else:
                masks = dist.rsample()
        else:
            masks = (torch.sigmoid(self.mask_logits) > 0.5).float()
        return masks
    
    def forward(self, x):
        """Same as before"""
        masks = self.get_masks(training=self.training)
        x_masked = x * masks
        z = self.encoder(x_masked)
        x_hat = self.decoder(z)
        return z, x_hat, masks
    
    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = F.normalize(x, p=2, dim=1)
        
        z, x_hat, masks = self.forward(x)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(x_hat, x)
        
        # Sparsity loss
        mask_probs = torch.sigmoid(self.mask_logits)
        sparsity_loss = mask_probs.mean()
        
        # Orthogonality penalty for decoder (components should be orthogonal)
        decoder_weights = self.decoder.weight  # [input_dim, n_components]
        gram_matrix = torch.mm(decoder_weights.t(), decoder_weights)  # [n_components, n_components]
        identity = torch.eye(gram_matrix.size(0), device=gram_matrix.device)
        orthogonal_loss = F.mse_loss(gram_matrix, identity)
        
        # Diversity penalty (encourage different components to use different features)
        # This is more complex and might not be necessary
        
        # Total loss
        loss = (recon_loss + 
                self.sparsity_weight * sparsity_loss + 
                self.orthogonal_penalty * orthogonal_loss)
        
        # Enhanced logging
        self.log('train_loss', loss, prog_bar=True)
        self.log('recon_loss', recon_loss, prog_bar=True)
        self.log('sparsity_loss', sparsity_loss, prog_bar=True)
        self.log('orthogonal_loss', orthogonal_loss, prog_bar=True)
        self.log('avg_mask_prob', mask_probs.mean(), prog_bar=True)
        self.log('active_features', masks.sum().float(), prog_bar=True)
        self.log('temperature', self.temperature, prog_bar=True)
        
        self.train_losses.append(loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = F.normalize(x, p=2, dim=1)
        z, x_hat, masks = self.forward(x)
        val_loss = F.mse_loss(x_hat, x)
        self.log('val_loss', val_loss, prog_bar=True)
        self.val_losses.append(val_loss.item())
        return val_loss

if __name__ == "__main__":
    import numpy as np
    import pytorch_lightning as pl
    from gumbel import SparsePCALightning, create_dataloader, TemperatureScheduler
    
    # Load data
    reward_data = np.load('data/reward-matrix.npz')
    A, P, Q = reward_data['Y'].shape
    
    # Reshape to (P*Q, A) - each row is one preference pair, columns are attributes
    X = reward_data['Y'].transpose(1, 2, 0).reshape(P * Q, A)
    
    print(f"Data shape: {X.shape}")
    print(f"A (attributes): {A}, P (pairs): {P}, Q (questions): {Q}")
    
    # Split data into train/validation
    n_samples = X.shape[0]
    n_train = int(0.8 * n_samples)
    
    X_train = X[:n_train]
    X_val = X[n_train:]
    
    print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")

    attribute_prompts = [
        # Tone & Stance
        "You are a concise assistant. Keep answers short and to the point.",
        "You are a verbose assistant. Provide detailed, expanded answers.",
        "You are a formal academic assistant. Use professional and scholarly tone.",
        "You are a casual conversational assistant. Write informally and with a friendly tone.",
        "You are a polite and diplomatic assistant. Maintain courteous phrasing throughout.",
        "You are a skeptical assistant. Verify claims and flag uncertainty.",
        "You are an optimistic assistant. Highlight positives and opportunities.",
        "You are a neutral assistant. Provide unbiased, objective answers.",
        "You are a directive assistant. Give clear, imperative instructions.",
        "You are a humorous assistant. Add light humor where appropriate.",
        "You are an empathetic assistant. Express care and support in your answers.",
        "You are a critical assistant. Evaluate and point out flaws where needed.",

        # Reasoning Style
        "You are a step-by-step assistant. Solve problems with enumerated steps.",
        "You are a hypothesis-driven assistant. State a hypothesis, test it, and give a conclusion.",
        "You are an answer-first assistant. Start with the final answer, then explain.",
        "You are a reasoning-first assistant. Show your reasoning before giving a conclusion.",
        "You are a verification assistant. Double-check each step before finalizing.",
        "You are a self-critical assistant. Critique your draft answer before finalizing.",
        "You are a comparative assistant. Present multiple solutions with pros and cons.",
        "You are a counterargument-first assistant. Present opposing views first, then respond.",
        "You are an analogy-driven assistant. Use analogies to explain concepts.",
        "You are an example-driven assistant. Use examples to support explanations.",
        "You are a proof-sketch assistant. Provide compact mathematical arguments.",
        "You are a checklist assistant. Provide information as checklists.",

        # Evidence & Citation
        "You are a quotation-heavy assistant. Use direct quotes from sources.",
        "You are a statistical assistant. Provide numeric estimates with confidence intervals.",
        "You are an attribution assistant. Attribute claims explicitly (“According to …”).",
        "You are an uncertainty-tagging assistant. Label uncertain claims explicitly.",
        "You are a cautious assistant. Avoid unverifiable claims.",
        "You are a non-citing assistant. Provide answers without citations.",

        # Creativity & Analogy
        "You are an analogy-heavy assistant. Use analogies in explanations.",
        "You are a metaphorical assistant. Use metaphors and figurative language.",
        "You are a storytelling assistant. Answer with stories.",
        "You are a Socratic assistant. Ask questions instead of answering directly.",
        "You are a brainstorming assistant. Generate many ideas quickly.",
        "You are a speculative assistant. Explore imaginative “what if” scenarios.",
        "You are a descriptive assistant. Use vivid visual descriptions.",
        "You are a humorous-analogy assistant. Explain concepts with funny analogies.",
        "You are a role-play assistant. Respond as if role-playing a scenario.",
        "You are a what-if assistant. Explore hypothetical situations.",

        # Domain-Specific Postures
        "You are an engineering-tradeoff assistant. Emphasize practical tradeoffs. Do note state your profession in your response.",
        "You are a scientific assistant. Cite theory and experiments. Do note state your profession in your response.",
        "You are a statistical assistant. Provide caveats and confidence intervals. Do note state your profession in your response.",
        "You are a medical assistant. Respond cautiously with disclaimers. Do note state your profession in your response.",
        "You are a legal assistant. Respond cautiously with disclaimers. Do note state your profession in your response.",
        "You are a business assistant. Provide executive summaries. Do note state your profession in your response.",
        "You are a policy-neutral assistant. Present politically neutral answers. Do note state your profession in your response.",
        "You are a pedagogical assistant. Teach step by step like a teacher. Do note state your profession in your response.",
        "You are a debugging assistant. Focus on code debugging. Do note state your profession in your response.",
        "You are a design-thinking assistant. Provide design-style solutions. Do note state your profession in your response.",

        # Interaction Controls
        "You are an options assistant. Present multiple alternatives.",
        "You are a next-steps assistant. Suggest action items or next steps when appropriate.",
        "You are a reflective assistant. Restate the user's question before answering.",
        "You are a pros-and-cons assistant. List pros and cons for each option.",
        "You are a resource assistant. Suggest related readings or resources.",
        "You are a question-back assistant. End with a question to the user.",
        "You are a recommendation assistant. Suggest what action to take.",
        "You are a perspective assistant. Provide multiple viewpoints.",
    ]

    
    # Ensure we have the right number of attribute names
    if len(attribute_prompts) != A:
        print(f"Warning: Expected {A} attributes, but have {len(attribute_prompts)} names")
        # Pad or truncate as needed
        if len(attribute_prompts) < A:
            attribute_prompts.extend([f"Attribute_{i}" for i in range(len(attribute_prompts), A)])
        else:
            attribute_prompts = attribute_prompts[:A]
    
    # Create model
    model = SparsePCALightning(
        input_dim=A,  # Number of attributes
        n_components=10,
        sparsity_weight=0.01,
        temperature=1.0,
        hard_gumbel=True
    )
    
    # Create data loaders
    train_loader = create_dataloader(X_train, batch_size=32, shuffle=True)
    val_loader = create_dataloader(X_val, batch_size=32, shuffle=False)
    
    # Create trainer with temperature scheduling
    temp_scheduler = TemperatureScheduler(initial_temp=1.0, final_temp=0.1)
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[temp_scheduler],
        accelerator='auto',
        enable_progress_bar=True
    )
    
    print("Starting training...")
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    print("Training completed! Analyzing results...")
    
    # Now evaluate the trained model
    
    evaluator = SparsePCAEvaluator(model, X_train, X_val, attribute_prompts)
    results = evaluate_sparse_pca_model(model, X_train, X_val, attribute_prompts)
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    # Print key results
    selection_results = results['selection_results']
    print(f"Selected {selection_results['n_selected']} out of {A} attributes")
    print(f"Sparsity: {selection_results['sparsity_ratio']:.2%}")
    print(f"Reconstruction error: {selection_results['reconstruction_error']:.4f}")
    print(f"Explained variance: {selection_results['explained_variance']:.4f}")
    
    print("\nSelected attributes:")
    for i, name in enumerate(selection_results['selected_names']):
        print(f"  {i+1}. {name}")
    
    # Get the basic sparse components without full evaluation if needed
    components, masks, mask_probs = model.get_sparse_components()
    print(f"\nQuick summary:")
    print(f"Selected features: {masks.sum()}/{len(masks)}")
    print(f"Average mask probability: {mask_probs.mean():.3f}")
    print(f"Active feature indices: {np.where(masks)[0].tolist()}")
    
    # Save results
    import json
    output_results = {
        'selected_indices': np.where(masks)[0].tolist(),
        'selected_attributes': selection_results['selected_names'],
        'mask_probabilities': mask_probs.tolist(),
        'reconstruction_error': selection_results['reconstruction_error'],
        'explained_variance': selection_results['explained_variance'],
        'sparsity_ratio': selection_results['sparsity_ratio']
    }
    
    with open('sparse_pca_results.json', 'w') as f:
        json.dump(output_results, f, indent=2)
    
    print(f"\nResults saved to sparse_pca_results.json")