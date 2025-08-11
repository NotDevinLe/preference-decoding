#!/usr/bin/env python3
"""
QAlign generators using MCMC sampling with MBR ROUGE-L selection.
Based on the QAlign algorithm from the paper.
"""

import torch
import numpy as np
import random
import math
from typing import List
from tqdm import tqdm

from src.evaluation.generation_evaluator import Generator


class QAlignDriftGenerator(Generator):
    """Generator using QAlign MCMC with drift scoring and MBR ROUGE-L selection."""
    
    def __init__(
        self,
        base_model,
        drift_model,
        p_vector,
        base_prompt,
        attribute_prompts,
        tokenizer,
        num_steps: int = 100,
        beta: float = 1.0,
        temperature: float = 0.8,
        max_length: int = 256,
        device=None
    ):
        """
        Initialize QAlign with drift generator.
        
        Args:
            base_model: Base language model for generation
            drift_model: Model for drift scoring
            p_vector: Preference vector
            base_prompt: Base system prompt
            attribute_prompts: List of attribute prompts
            tokenizer: Tokenizer
            num_steps: Number of MCMC steps (default 100)
            beta: Temperature parameter for acceptance probability (default 1.0)
            temperature: Sampling temperature for generation (default 0.8)
            max_length: Maximum generation length (default 256)
            device: Device for computation
        """
        self.base_model = base_model
        self.drift_model = drift_model
        self.base_prompt = base_prompt
        self.tokenizer = tokenizer
        self.num_steps = num_steps
        self.beta = beta
        self.temperature = temperature
        self.max_length = max_length
        self.device = device or 'cpu'
        
        # Sparsify p_vector to keep only top 7 attributes by absolute value
        top_k = 7
        if top_k < len(p_vector):
            abs_values = np.abs(p_vector)
            top_indices = np.argpartition(abs_values, -top_k)[-top_k:]
            
            # Create sparse vector with only top k values
            self.p_vector = p_vector[top_indices]
            self.attribute_prompts = [attribute_prompts[i] for i in top_indices]
            
            print(f"QAlignDrift sparsified: keeping {top_k} attributes with indices {sorted(top_indices.tolist())}")
            print(f"Active weights: {[f'{i}:{p_vector[i]:.4f}' for i in sorted(top_indices.tolist())]}")
        else:
            self.p_vector = p_vector
            self.attribute_prompts = attribute_prompts
    
    def generate(self, prompt: str) -> str:
        """Generate using QAlign MCMC with drift scoring and MBR ROUGE-L selection."""
        from src.core.drift import get_scores
        
        # Define reward function using drift scores
        def reward_fn(question: str, response: str) -> float:
            scores = get_scores(
                [(question, [response])],
                self.drift_model,
                self.p_vector,
                self.base_prompt,
                self.attribute_prompts,
                self.device,
                self.tokenizer
            )[0]
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()
            return float(scores[0])
        
        # Run QAlign MCMC algorithm
        accepted_samples = self._run_qalign_mcmc(prompt, reward_fn)
        
        # Use MBR ROUGE-L to select best response
        best_response = self._mbr_rouge_l_selection(accepted_samples)
        return best_response
    
    def _run_qalign_mcmc(self, prompt: str, reward_fn) -> list:
        """Run QAlign MCMC sampling."""
        # Generate initial response
        initial_response = self._generate_response(prompt)
        current_response = initial_response
        current_reward = reward_fn(prompt, current_response)
        
        accepted_samples = [current_response]
        acceptance_count = 0
        
        for step in range(self.num_steps):
            # Generate proposal using QUEST method
            proposal_response = self._generate_proposal(prompt, current_response)
            
            if proposal_response is None or len(proposal_response.strip()) == 0:
                continue
            
            # Compute acceptance probability
            proposal_reward = reward_fn(prompt, proposal_response)
            
            # Compute acceptance probability using Eq. 9 from QAlign paper
            reward_diff = proposal_reward - current_reward
            current_len = len(self.tokenizer.encode(current_response))
            proposal_len = len(self.tokenizer.encode(proposal_response))
            
            if proposal_len > 0:
                length_ratio = current_len / proposal_len
                log_acceptance_prob = min(0, reward_diff / self.beta + math.log(length_ratio))
                acceptance_prob = math.exp(log_acceptance_prob)
            else:
                acceptance_prob = 0
            
            # Accept or reject
            if random.random() < acceptance_prob:
                current_response = proposal_response
                current_reward = proposal_reward
                accepted_samples.append(current_response)
                acceptance_count += 1
            else:
                accepted_samples.append(current_response)
        
        return accepted_samples
    
    def _generate_response(self, prompt: str) -> str:
        """Generate a complete response."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        if prompt in response:
            response = response.split(prompt)[-1].strip()
        return response
    
    def _generate_proposal(self, prompt: str, current_response: str) -> str:
        """Generate proposal using QUEST method."""
        # Tokenize current response
        current_tokens = self.tokenizer.encode(current_response, add_special_tokens=False)
        
        if len(current_tokens) == 0:
            return self._generate_response(prompt)
        
        # Sample index uniformly
        i = random.randint(0, len(current_tokens) - 1)
        
        # Create prefix
        prefix_tokens = current_tokens[:i]
        prefix_text = self.tokenizer.decode(prefix_tokens, skip_special_tokens=True)
        
        # Generate completion from prefix
        full_prompt = prompt
        if prefix_text:
            full_prompt = prompt + " " + prefix_text
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        
        try:
            with torch.no_grad():
                outputs = self.base_model.generate(
                    **inputs,
                    max_new_tokens=max(1, self.max_length - i),
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract the completion part
            if full_prompt in full_response:
                completion = full_response.split(full_prompt)[-1].strip()
            else:
                completion = full_response[len(full_prompt):].strip()
            
            return prefix_text + " " + completion if prefix_text else completion
        except Exception:
            return None
    
    def _mbr_rouge_l_selection(self, candidates: list) -> str:
        """Select best candidate using MBR ROUGE-L."""
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]
        
        best_candidate = None
        best_score = -1
        
        for candidate in candidates:
            total_score = 0
            for other in candidates:
                score = self._rouge_l_f1(candidate, other)
                total_score += score
            
            avg_score = total_score / len(candidates)
            if avg_score > best_score:
                best_score = avg_score
                best_candidate = candidate
        
        return best_candidate
    
    def _rouge_l_f1(self, text1: str, text2: str) -> float:
        """Compute ROUGE-L F1 score."""
        def lcs_length(x, y):
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        tokens1 = text1.lower().split()
        tokens2 = text2.lower().split()
        
        if len(tokens1) == 0 or len(tokens2) == 0:
            return 0.0
        
        lcs_len = lcs_length(tokens1, tokens2)
        precision = lcs_len / len(tokens1) if len(tokens1) > 0 else 0.0
        recall = lcs_len / len(tokens2) if len(tokens2) > 0 else 0.0
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


class QAlignMLEGenerator(Generator):
    """Generator using QAlign MCMC with MLE scoring and MBR ROUGE-L selection."""
    
    def __init__(
        self,
        base_model,
        mle_model,
        p_vector_mle,
        base_prompt,
        attribute_prompts,
        tokenizer,
        num_steps: int = 100,
        beta: float = 1.0,
        temperature: float = 0.8,
        max_length: int = 256,
        device=None
    ):
        """
        Initialize QAlign with MLE generator.
        
        Args:
            base_model: Base language model
            mle_model: Model for MLE scoring
            p_vector_mle: MLE-optimized preference vector
            base_prompt: Base system prompt
            attribute_prompts: List of attribute prompts
            tokenizer: Tokenizer
            num_steps: Number of MCMC steps (default 100)
            beta: Temperature parameter for acceptance probability (default 1.0)
            temperature: Sampling temperature for generation (default 0.8)
            max_length: Maximum generation length (default 256)
            device: Device for computation
        """
        self.base_model = base_model
        self.mle_model = mle_model
        self.base_prompt = base_prompt
        self.tokenizer = tokenizer
        self.num_steps = num_steps
        self.beta = beta
        self.temperature = temperature
        self.max_length = max_length
        self.device = device or 'cpu'
        
        # Sparsify p_vector to keep only top 7 attributes by absolute value
        top_k = 7
        if top_k < len(p_vector_mle):
            abs_values = np.abs(p_vector_mle)
            top_indices = np.argpartition(abs_values, -top_k)[-top_k:]
            
            # Create sparse vector with only top k values
            self.p_vector_mle = p_vector_mle[top_indices]
            self.attribute_prompts = [attribute_prompts[i] for i in top_indices]
            
            print(f"QAlignMLE sparsified: keeping {top_k} attributes with indices {sorted(top_indices.tolist())}")
            print(f"Active weights: {[f'{i}:{p_vector_mle[i]:.4f}' for i in sorted(top_indices.tolist())]}")
        else:
            self.p_vector_mle = p_vector_mle
            self.attribute_prompts = attribute_prompts
    
    def generate(self, prompt: str) -> str:
        """Generate using QAlign MCMC with MLE scoring and MBR ROUGE-L selection."""
        from src.core.drift import get_scores
        
        # Define reward function using MLE scores
        def reward_fn(question: str, response: str) -> float:
            scores = get_scores(
                [(question, [response])],
                self.mle_model,
                self.p_vector_mle,
                self.base_prompt,
                self.attribute_prompts,
                self.device,
                self.tokenizer
            )[0]
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()
            return float(scores[0])
        
        # Use same MCMC algorithm as drift version
        drift_gen = QAlignDriftGenerator(
            self.base_model,
            self.mle_model,
            self.p_vector_mle,
            self.base_prompt,
            self.attribute_prompts,
            self.tokenizer,
            self.num_steps,
            self.beta,
            self.temperature,
            self.max_length,
            self.device
        )
        
        # Run QAlign MCMC algorithm
        accepted_samples = drift_gen._run_qalign_mcmc(prompt, reward_fn)
        
        # Use MBR ROUGE-L to select best response
        best_response = drift_gen._mbr_rouge_l_selection(accepted_samples)
        return best_response