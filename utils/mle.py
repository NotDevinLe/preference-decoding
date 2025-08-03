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

class MLE:
    def __init__(self, model, tokenizer, num_expectation_samples, data, device):
        self.model = model
        self.tokenizer = tokenizer
        self.data = data
        self.device = device
        self.p = torch.randn(len(attribute_prompts), device=device)
        self.num_expectation_samples = num_expectation_samples
        self.expectation = torch.zeros((len(data), self.num_expectation_samples, len(attribute_prompts)), device=device)

        self.generate_expectation_outputs()


    def train(self, num_epochs=1000, learning_rate=0.01, n=10):
        """
        Train the MLE model using gradient descent.
        
        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            n: Number of samples to draw from expectation for Monte Carlo estimation
        """
        # Convert data once
        training_data = [(item['prompt'], item['chosen']) for item in self.data]
        
        for epoch in range(num_epochs):
            # 1. Get rewards for ALL training data at once
            curr_reward = self.get_reward(training_data)  # (batch_size, num_attributes)
            batch_avg_reward = torch.mean(curr_reward, dim=0)  # (num_attributes,)
            
            # 2. Compute expected reward using current p
            # self.expectation is (num_data, num_expectation_samples, num_attributes)
            # We want to compute E[R] across all data points and samples
            
            # Compute scores for each expectation sample: sum over attributes weighted by p
            # expectation_scores: (num_data, num_expectation_samples)
            expectation_scores = torch.sum(
                self.expectation * self.p.unsqueeze(0).unsqueeze(0), 
                dim=2
            )  # Broadcasting: (num_data, num_expectation_samples, num_attributes) * (1, 1, num_attributes)
            
            # Flatten to sample across all data points and expectation samples
            flat_scores = expectation_scores.view(-1)  # (num_data * num_expectation_samples,)
            flat_expectation = self.expectation.view(-1, len(attribute_prompts))  # (num_data * num_expectation_samples, num_attributes)
            
            # Sample n indices using softmax probabilities
            softmax_probs = torch.softmax(flat_scores, dim=0)
            indices = torch.multinomial(softmax_probs, n, replacement=True)
            
            # 3. Get expected reward from sampled expectation
            selected_expectation_rewards = flat_expectation[indices]  # (n, num_attributes)
            expected_reward = torch.mean(selected_expectation_rewards, dim=0)  # (num_attributes,)
            
            # 4. Compute gradient and update p
            gradient = (batch_avg_reward - expected_reward) / 1.0  # Divide by beta if you have one
            
            with torch.no_grad():
                self.p += learning_rate * gradient  # Note: += for gradient ascent (maximizing likelihood)
            
            # Optional: logging
            if epoch % 100 == 0:
                likelihood_diff = torch.dot(gradient, self.p).item()
                print(f"Epoch {epoch}")
                print(f"  Gradient norm: {torch.norm(gradient).item():.4f}")
                print(f"  P norm: {torch.norm(self.p).item():.4f}")
                print(f"  Likelihood direction: {likelihood_diff:.4f}")
                print(f"  Current p: {self.p.cpu().numpy()}")
                
            # Optional: early stopping if gradient is very small
            if torch.norm(gradient).item() < 1e-6:
                print(f"Converged at epoch {epoch}")
                break

    def generate_expectation_outputs(self):
        """
        Generate num_expectation_samples outputs for each prompt in the data.
        Compute their reward vectors and store them.
        """
        print("Generating expectation outputs...")
        
        for i, item in enumerate(self.data):
            prompt = item['prompt']
            
            # Format prompt for generation
            inputs = self.tokenizer.apply_chat_template([
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": prompt}
            ], tokenize=False, add_generation_prompt=True)
            
            # Generate multiple outputs
            sampling_params = SamplingParams(
                temperature=1.0, 
                max_tokens=1024, 
                n=self.num_expectation_samples
            )
            
            outputs = self.model.generate(inputs, sampling_params)
            
            # Extract generated texts
            generated_texts = []
            for output in outputs[0].outputs:  # outputs is a list with one RequestOutput
                generated_texts.append(output.text)
            
            # Compute rewards for all generated outputs for this prompt
            data_for_reward = [(prompt, text) for text in generated_texts]
            reward_matrix = self.get_reward(data_for_reward)  # (num_expectation_samples, num_attributes)
            
            # Store the reward vectors
            self.expectation[i] = reward_matrix
            
            if i % 10 == 0:
                print(f"Processed {i}/{len(self.data)} prompts")

        print("Finished generating expectation outputs")


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
        base_probs, base_counts = get_log_probs(
            self.model, self.tokenizer, [base_prompt] * m, 
            flat_questions, flat_outputs
        )
        base_tensor = torch.tensor(base_probs, device=self.device) / torch.tensor(base_counts, device=self.device)
        
        # Initialize drift scores for all items
        drift_scores = torch.zeros((m, len(attribute_prompts)), device=self.device)
        
        # Process each attribute prompt individually
        for i, attribute_prompt in enumerate(attribute_prompts):
            # Get log probabilities for this attribute prompt
            attr_probs, attr_counts = get_log_probs(
                self.model, self.tokenizer, [attribute_prompt] * m, 
                flat_questions, flat_outputs
            )
            
            # Convert to tensors
            attr_tensor = torch.tensor(attr_probs, device=self.device) / torch.tensor(attr_counts, device=self.device)
            
            # Compute drift contribution for this attribute
            attribute_drift = (attr_tensor - base_tensor)
            
            # Add to total drift scores
            drift_scores[:, i] += attribute_drift
        
        return drift_scores