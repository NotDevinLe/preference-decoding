import pickle
import argparse
import numpy as np
import torch
import json
from drift import get_log_probs, get_scores
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from attribute_prompts import attribute_prompts, base_prompt
from drift import get_scores
import wandb
from tqdm import tqdm

class MLE:
    def __init__(self, model, tokenizer, data, device, expectation_matrix=None, chosen_rewards=None, num_expectation_samples=100, use_wandb=True, wandb_project="mle-training"):
        self.model = model
        self.tokenizer = tokenizer
        self.data = data
        self.device = device
        self.p = torch.randn(len(attribute_prompts), device=device)
        self.num_expectation_samples = num_expectation_samples
        
        # Initialize expectation matrix state
        self.expectation = None
        self.chosen_rewards = None
        self._expectation_generated = False
        
        if expectation_matrix is not None:
            # Load pre-computed expectation matrix
            self.expectation = expectation_matrix.to(device)
            self.num_expectation_samples = expectation_matrix.shape[1]
            self._expectation_generated = True
            print(f"Loaded expectation matrix: {self.expectation.shape}")
            
            # Load or compute chosen rewards
            if chosen_rewards is not None:
                # Load pre-computed chosen rewards
                self.chosen_rewards = chosen_rewards.to(device)
                print(f"Loaded pre-computed chosen rewards: {self.chosen_rewards.shape}")
            else:
                # Compute chosen rewards for this specific user's data
                print("Computing chosen rewards...")
                self.precompute_chosen_rewards()
        
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self._wandb_initialized = False
        # Don't initialize wandb here - wait until training starts so we can include all parameters


    def generate_expectation_matrix(self, num_expectation_samples=None):
        """
        Generate expectation matrix for the current user's training data.
        This should be called before training.
        
        Args:
            num_expectation_samples: Number of expectation samples per prompt (overrides constructor value)
        """
        if num_expectation_samples is not None:
            self.num_expectation_samples = num_expectation_samples
            
        print(f"Generating expectation matrix for {len(self.data)} prompts with {self.num_expectation_samples} samples each...")
        
        # Extract prompts from data
        prompts = [item['prompt'] for item in self.data]
        
        # Prepare all prompts at once
        all_inputs = []
        for prompt in prompts:
            formatted_input = self.tokenizer.apply_chat_template([
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": prompt}
            ], tokenize=False, add_generation_prompt=True)
            all_inputs.append(formatted_input)
        
        # Generate all outputs at once - vLLM handles batching internally
        print(f"Generating {self.num_expectation_samples} outputs for {len(all_inputs)} prompts...")
        sampling_params = SamplingParams(
            temperature=1.0, 
            max_tokens=1024, 
            n=self.num_expectation_samples
        )
        
        all_outputs = self.model.generate(all_inputs, sampling_params)
        
        # Extract all generated texts and prepare for reward computation
        print("Preparing data for reward computation...")
        all_reward_data = []
        output_mapping = []  # Track which outputs belong to which prompt
        
        for prompt_idx, vllm_output in enumerate(all_outputs):
            prompt = prompts[prompt_idx]
            for sample_idx, output in enumerate(vllm_output.outputs):
                all_reward_data.append((prompt, output.text))
                output_mapping.append((prompt_idx, sample_idx))
        
        # Compute rewards
        print(f"Computing rewards for {len(all_reward_data)} generated outputs...")
        all_rewards = self.get_reward(all_reward_data)
        
        # Create expectation matrix
        self.expectation = torch.zeros((len(prompts), self.num_expectation_samples, len(attribute_prompts)), device=self.device)
        
        # Map rewards back to expectation matrix
        for idx, (prompt_idx, sample_idx) in enumerate(output_mapping):
            self.expectation[prompt_idx, sample_idx] = all_rewards[idx]
        
        self._expectation_generated = True
        print(f"Expectation matrix generated successfully: {self.expectation.shape}")
        
        # Now compute chosen rewards
        self.precompute_chosen_rewards()
        
        # Don't initialize wandb here - wait until training starts
    
    def train(self, max_epochs=10000, learning_rate=0.01, beta=1.0, num_mc_samples=10, 
              gradient_tolerance=1e-6, loss_tolerance=1e-6, patience=100, l1_lambda=0.0, run_name=None):
        """
        Train the MLE model using gradient descent with convergence criteria and L1 regularization.
        
        ∇_p log π(y|x) = (1/β) R^(i)(x,y) - (1/β) E_{y'~π(·|x)} [R^(i)(x,y')] - λ * sign(p)
        
        Args:
            max_epochs: Maximum number of training epochs
            learning_rate: Learning rate for gradient descent
            beta: Temperature parameter from derivation
            num_mc_samples: Number of Monte Carlo samples for expectation estimation
            gradient_tolerance: Stop when gradient norm is below this threshold
            loss_tolerance: Stop when loss change is below this threshold
            patience: Stop if no improvement for this many epochs
            l1_lambda: L1 regularization coefficient (0.0 = no regularization)
            run_name: Custom name for wandb run
        """
        
        # Check if expectation matrix has been generated
        if not self._expectation_generated:
            print("WARNING: Expectation matrix has not been generated yet!")
            print("Please call mle.generate_expectation_matrix() before training.")
            print("Generating expectation matrix now with default parameters...")
            self.generate_expectation_matrix()
        
        # Initialize wandb with run name if not already initialized
        if self.use_wandb and not self._wandb_initialized:
            config = {
                "num_expectation_samples": self.num_expectation_samples,
                "num_mc_samples": num_mc_samples,
                "num_data_points": len(self.data),
                "num_attributes": len(attribute_prompts),
                "initial_p_norm": torch.norm(self.p).item(),
                "beta": beta,
                "learning_rate": learning_rate,
                "max_epochs": max_epochs,
                "gradient_tolerance": gradient_tolerance,
                "loss_tolerance": loss_tolerance,
                "patience": patience,
                "l1_lambda": l1_lambda
            }
            
            # Create descriptive run name
            if run_name is None:
                run_name = f"exp{self.num_expectation_samples}_mc{num_mc_samples}_data{len(self.data)}_lr{learning_rate}_beta{beta}"
                if l1_lambda > 0.0:
                    run_name += f"_l1{l1_lambda}"
            
            wandb.init(
                project=self.wandb_project,
                name=run_name,
                config=config
            )
            self._wandb_initialized = True
        
        # Initialize convergence tracking variables
        epoch = 0
        best_loss = float('inf')
        epochs_without_improvement = 0
        prev_loss = float('inf')
        
        # Training loop with convergence criteria
        while epoch < max_epochs:
            total_gradient = torch.zeros_like(self.p)
            epoch_log_likelihood = 0.0
            
            # Process each data point separately to compute proper expectation
            for data_idx, item in enumerate(self.data):
                # 1. Get pre-computed reward for chosen response: R^(i)(x,y)
                chosen_reward = self.chosen_rewards[data_idx]  # (num_attributes,)
                
                # 2. Compute expectation E_{y'~π(·|x)} [R^(i)(x,y')]
                # Use the precomputed expectation samples for this prompt
                expectation_rewards = self.expectation[data_idx]  # (num_expectation_samples, num_attributes)
                
                # Compute scores for each sample using current p: p^T * R(x,y')
                sample_scores = torch.sum(expectation_rewards * self.p.unsqueeze(0), dim=1)  # (num_expectation_samples,)
                
                # Compute softmax probabilities: π(y'|x) ∝ exp(p^T * R(x,y') / β)
                softmax_probs = torch.softmax(sample_scores / beta, dim=0)  # (num_expectation_samples,)
                
                # Sample from the distribution for Monte Carlo expectation
                sampled_indices = torch.multinomial(softmax_probs, num_mc_samples, replacement=True)
                sampled_rewards = expectation_rewards[sampled_indices]  # (num_mc_samples, num_attributes)
                
                # Compute Monte Carlo estimate of expectation: E_{y'~π(·|x)} [R^(i)(x,y')]
                expected_reward = torch.mean(sampled_rewards, dim=0)  # (num_attributes,)
                
                # 3. Compute gradient for this data point: ∇_p log π(y|x)
                data_gradient = (chosen_reward - expected_reward) / beta
                total_gradient += data_gradient
                
                # Compute log-likelihood contribution for this data point
                # log π(y|x) = p^T R(x,y) / β - log Z(x)
                chosen_score = torch.sum(self.p * chosen_reward) / beta
                # Approximate log Z(x) using all expectation samples
                all_scores = torch.sum(expectation_rewards * self.p.unsqueeze(0), dim=1) / beta
                log_Z = torch.logsumexp(all_scores, dim=0)
                data_log_likelihood = chosen_score - log_Z
                epoch_log_likelihood += data_log_likelihood.item()
            
            # Average gradient and log-likelihood across all data points
            total_gradient = total_gradient / len(self.data)
            avg_log_likelihood = epoch_log_likelihood / len(self.data)
            
            # Add L1 regularization gradient: -λ * sign(p)
            if l1_lambda > 0.0:
                l1_gradient = -l1_lambda * torch.sign(self.p)
                total_gradient += l1_gradient
            
            # 4. Update p using gradient ascent (maximizing log-likelihood - L1 penalty)
            with torch.no_grad():
                self.p += learning_rate * total_gradient
            
            # Compute additional metrics
            gradient_norm = torch.norm(total_gradient).item()
            p_norm = torch.norm(self.p).item()
            l1_penalty = l1_lambda * torch.sum(torch.abs(self.p)).item() if l1_lambda > 0.0 else 0.0
            current_loss = -avg_log_likelihood + l1_penalty  # Negative log-likelihood + L1 penalty
            
            # Check convergence criteria
            loss_change = abs(current_loss - prev_loss)
            
            # Update best loss and patience counter
            if current_loss < best_loss:
                best_loss = current_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "avg_log_likelihood": avg_log_likelihood,
                    "negative_log_likelihood_loss": -avg_log_likelihood,
                    "l1_penalty": l1_penalty,
                    "total_loss": current_loss,
                    "gradient_norm": gradient_norm,
                    "p_norm": p_norm,
                    "learning_rate": learning_rate,
                    "loss_change": loss_change,
                    "epochs_without_improvement": epochs_without_improvement
                })
            
            # Console logging
            if epoch % 100 == 0:
                print(f"Epoch {epoch}")
                print(f"  Avg Log-Likelihood: {avg_log_likelihood:.4f}")
                print(f"  Negative LL Loss: {-avg_log_likelihood:.4f}")
                if l1_lambda > 0.0:
                    print(f"  L1 Penalty: {l1_penalty:.4f}")
                print(f"  Total Loss: {current_loss:.4f}")
                print(f"  Gradient norm: {gradient_norm:.4f}")
                print(f"  P norm: {p_norm:.4f}")
                print(f"  Loss change: {loss_change:.6f}")
                if l1_lambda == 0.0:
                    print(f"  Epochs without improvement: {epochs_without_improvement}")
                else:
                    print(f"  Epochs without improvement: {epochs_without_improvement} (patience disabled with L1)")
                print(f"  Current p: {self.p.cpu().numpy()}")
            
            # Check convergence criteria
            if gradient_norm < gradient_tolerance:
                print(f"\nConverged: Gradient norm ({gradient_norm:.2e}) < tolerance ({gradient_tolerance:.2e})")
                if self.use_wandb:
                    wandb.log({
                        "converged_epoch": epoch,
                        "convergence_reason": "gradient_tolerance"
                    })
                break
            
            if loss_change < loss_tolerance and epoch > 0:
                print(f"\nConverged: Loss change ({loss_change:.2e}) < tolerance ({loss_tolerance:.2e})")
                if self.use_wandb:
                    wandb.log({
                        "converged_epoch": epoch,
                        "convergence_reason": "loss_tolerance"
                    })
                break
            
            if epochs_without_improvement >= patience and l1_lambda == 0.0:
                print(f"\nConverged: No improvement for {patience} epochs")
                if self.use_wandb:
                    wandb.log({
                        "converged_epoch": epoch,
                        "convergence_reason": "patience"
                    })
                break
            
            # Update previous loss and increment epoch
            prev_loss = current_loss
            epoch += 1
        
        # If we hit max_epochs without converging
        if epoch >= max_epochs:
            print(f"\nReached maximum epochs ({max_epochs}) without full convergence")
            if self.use_wandb:
                wandb.log({
                    "converged_epoch": epoch,
                    "convergence_reason": "max_epochs"
                })
        
        print(f"\nTraining completed at epoch {epoch}")
        print(f"Final gradient norm: {gradient_norm:.2e}")
        print(f"Final loss: {current_loss:.4f}")
        print(f"Final p norm: {p_norm:.4f}")

    
    def precompute_chosen_rewards(self):
        """
        Pre-compute reward vectors for all chosen responses to avoid recomputation during training.
        """
        print("Pre-computing chosen rewards...")
        
        # Prepare data for batch processing
        chosen_data = [(item['prompt'], item['chosen']) for item in self.data]
        
        # Compute rewards for all chosen responses at once
        self.chosen_rewards = self.get_reward(chosen_data)  # (num_data, num_attributes)
        
        print(f"Pre-computed rewards for {len(self.data)} chosen responses")


    def get_reward(self, data):

        """
        Compute the reward for a given set of data.

        Args:
            data: List of (prompt, output) tuples

        Returns:
            torch.Tensor: m x n matrix where m = number of prompts, n = number of attributes
        """

        m = len(data)  # number of prompts
        
        # Flatten all data for batch processing
        flat_questions = []
        flat_outputs = []
        
        for prompt, output in data:
            flat_questions.append(prompt)
            flat_outputs.append(output)
        
        # Get base log probabilities for all flattened items
        print("Computing base log probabilities...")
        base_probs, base_counts = get_log_probs(
            self.model, self.tokenizer, [base_prompt] * m, 
            flat_questions, flat_outputs, self.device
        )
        base_tensor = torch.tensor(base_probs, device=self.device) / torch.tensor(base_counts, device=self.device)
        
        # Initialize drift scores for all items
        drift_scores = torch.zeros((m, len(attribute_prompts)), device=self.device)
        
        # Process each attribute prompt individually with progress bar
        print(f"Computing attribute log probabilities for {len(attribute_prompts)} attributes...")
        for i, attribute_prompt in enumerate(tqdm(attribute_prompts, desc="Processing attributes")):
            # Get log probabilities for this attribute prompt
            attr_probs, attr_counts = get_log_probs(
                self.model, self.tokenizer, [attribute_prompt] * m, 
                flat_questions, flat_outputs, self.device
            )
            
            # Convert to tensors
            attr_tensor = torch.tensor(attr_probs, device=self.device) / torch.tensor(attr_counts, device=self.device)
            
            # Compute drift contribution for this attribute
            attribute_drift = (attr_tensor - base_tensor)
            
            # Add to total drift scores
            drift_scores[:, i] += attribute_drift
        
        return drift_scores
    
    def save_results(self, save_path, num_mc_samples=None, l1_lambda=None):
        """Save the learned p vector and training results."""
        results = {
            "p_vector": self.p.cpu().numpy().tolist(),
            "p_norm": torch.norm(self.p).item(),
            "num_attributes": len(attribute_prompts),
            "num_data_points": len(self.data),
        }
        
        # Add optional parameters if provided
        if num_mc_samples is not None:
            results["num_mc_samples"] = num_mc_samples
        if l1_lambda is not None:
            results["l1_lambda"] = l1_lambda
        
        with open(save_path, "a") as f:
            json.dump(results, f)
            f.write("\n")
        
        if self.use_wandb:
            try:
                # Try to save with wandb, but don't fail if path issues
                import os
                abs_path = os.path.abspath(save_path)
                wandb.save(abs_path)
            except Exception as e:
                print(f"Warning: Could not save to wandb: {e}")
            wandb.finish()
    
    def save_expectation_matrix(self, save_path):
        """Save the expectation matrix to disk for reuse across users."""
        print(f"Saving expectation matrix to {save_path}")
        torch.save({
            'expectation_matrix': self.expectation.cpu(),
            'num_expectation_samples': self.num_expectation_samples,
            'num_attributes': len(attribute_prompts)
        }, save_path)
        print(f"Expectation matrix saved successfully")
    
    def save_chosen_rewards(self, save_path):
        """Save the chosen rewards separately for reuse."""
        print(f"Saving chosen rewards to {save_path}")
        torch.save({
            'chosen_rewards': self.chosen_rewards.cpu(),
            'num_data_points': len(self.data),
            'num_attributes': len(attribute_prompts)
        }, save_path)
        print(f"Chosen rewards saved successfully")
    
    @staticmethod
    def load_expectation_matrix(load_path):
        """Load a pre-computed expectation matrix."""
        print(f"Loading expectation matrix from {load_path}")
        checkpoint = torch.load(load_path)
        print(f"Loaded expectation matrix with shape: {checkpoint['expectation_matrix'].shape}")
        return checkpoint
    
    @staticmethod
    def load_chosen_rewards(load_path):
        """Load pre-computed chosen rewards."""
        print(f"Loading chosen rewards from {load_path}")
        checkpoint = torch.load(load_path)
        print(f"Loaded chosen rewards with shape: {checkpoint['chosen_rewards'].shape}")
        return checkpoint