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
    def __init__(self, model, tokenizer, data, device, expectation_matrix, chosen_rewards, use_wandb=True, wandb_project="mle-training"):
        self.model = model
        self.tokenizer = tokenizer
        self.data = data
        self.device = device
        self.p = torch.randn(len(attribute_prompts), device=device)
        
        # Load pre-computed matrices
        self.expectation = expectation_matrix.to(device)
        self.chosen_rewards = chosen_rewards.to(device)
        self.num_expectation_samples = expectation_matrix.shape[1]
        
        print(f"Loaded expectation matrix: {self.expectation.shape}")
        print(f"Loaded chosen rewards: {self.chosen_rewards.shape}")
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=wandb_project, config={
                "num_expectation_samples": self.num_expectation_samples,
                "num_data_points": len(data),
                "num_attributes": len(attribute_prompts),
                "initial_p_norm": torch.norm(self.p).item()
            })


    def train(self, num_epochs=1000, learning_rate=0.01, beta=1.0, num_mc_samples=10):
        """
        Train the MLE model using gradient descent following the mathematical derivation.
        
        ∇_p log π(y|x) = (1/β) R^(i)(x,y) - (1/β) E_{y'~π(·|x)} [R^(i)(x,y')]
        
        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            beta: Temperature parameter from derivation
            num_mc_samples: Number of Monte Carlo samples for expectation estimation
        """
        
        for epoch in range(num_epochs):
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
            
            # 4. Update p using gradient ascent (maximizing log-likelihood)
            with torch.no_grad():
                self.p += learning_rate * total_gradient
            
            # Compute additional metrics
            gradient_norm = torch.norm(total_gradient).item()
            p_norm = torch.norm(self.p).item()
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "avg_log_likelihood": avg_log_likelihood,
                    "negative_log_likelihood_loss": -avg_log_likelihood,  # Loss is negative log-likelihood
                    "gradient_norm": gradient_norm,
                    "p_norm": p_norm,
                    "learning_rate": learning_rate,
                })
                
                # Log individual p values
                for i, p_val in enumerate(self.p.cpu().numpy()):
                    wandb.log({f"p_{i}": p_val})
            
            # Console logging
            if epoch % 100 == 0:
                print(f"Epoch {epoch}")
                print(f"  Avg Log-Likelihood: {avg_log_likelihood:.4f}")
                print(f"  Negative LL Loss: {-avg_log_likelihood:.4f}")
                print(f"  Gradient norm: {gradient_norm:.4f}")
                print(f"  P norm: {p_norm:.4f}")
                print(f"  Current p: {self.p.cpu().numpy()}")
                
            # Optional: early stopping if gradient is very small
            if gradient_norm < 1e-6:
                print(f"Converged at epoch {epoch}")
                if self.use_wandb:
                    wandb.log({"converged_epoch": epoch})
                break

    def generate_expectation_outputs(self):
        """
        Generate num_expectation_samples outputs for each prompt in the data.
        Compute their reward vectors and store them.
        Uses vLLM's built-in batching for maximum efficiency.
        """
        print("Generating expectation outputs...")
        
        # Prepare all prompts at once
        all_inputs = []
        for item in self.data:
            prompt = item['prompt']
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
            prompt = self.data[prompt_idx]['prompt']
            for sample_idx, output in enumerate(vllm_output.outputs):
                all_reward_data.append((prompt, output.text))
                output_mapping.append((prompt_idx, sample_idx))
        
        # Compute all rewards at once
        print(f"Computing rewards for {len(all_reward_data)} generated outputs...")
        all_rewards = self.get_reward(all_reward_data)  # (total_outputs, num_attributes)
        
        # Map rewards back to expectation matrix
        for idx, (prompt_idx, sample_idx) in enumerate(output_mapping):
            self.expectation[prompt_idx, sample_idx] = all_rewards[idx]
        
        print("Finished generating expectation outputs")
    
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
    
    def save_results(self, save_path):
        """Save the learned p vector and training results."""
        results = {
            "p_vector": self.p.cpu().numpy().tolist(),
            "p_norm": torch.norm(self.p).item(),
            "num_attributes": len(attribute_prompts),
            "num_data_points": len(self.data),
        }
        
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        
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
        """Save the expectation matrix and chosen rewards to disk for reuse."""
        print(f"Saving expectation matrix to {save_path}")
        torch.save({
            'expectation_matrix': self.expectation.cpu(),
            'chosen_rewards': self.chosen_rewards.cpu(),
            'num_data_points': len(self.data),
            'num_expectation_samples': self.num_expectation_samples,
            'num_attributes': len(attribute_prompts)
        }, save_path)
        print(f"Expectation matrix and chosen rewards saved successfully")
    
    @staticmethod
    def load_expectation_matrix(load_path):
        """Load a pre-computed expectation matrix."""
        print(f"Loading expectation matrix from {load_path}")
        checkpoint = torch.load(load_path)
        print(f"Loaded expectation matrix with shape: {checkpoint['expectation_matrix'].shape}")
        return checkpoint