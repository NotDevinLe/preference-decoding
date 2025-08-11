#!/usr/bin/env python3
"""
Unified QAlign generator using MCMC sampling with MBR ROUGE-L selection.
Based on the QAlign algorithm from the paper.
"""

import torch
import numpy as np
import random
import math
from typing import List, Optional, Callable
from tqdm import tqdm

from src.evaluation.generation_evaluator import Generator


class QAlignGenerator(Generator):
    """Generator using QAlign MCMC with customizable scoring and MBR ROUGE-L selection."""
    
    def __init__(
        self,
        base_model,
        scoring_model,
        p_vector,
        base_prompt,
        attribute_prompts,
        tokenizer,
        num_steps: int = 100,
        beta: float = 1.0,
        temperature: float = 0.8,
        max_length: int = 256,
        device=None,
        memory_cleanup_frequency: int = 10,
        method_name: str = "QAlign",
        sparsify_k: Optional[int] = 7
    ):
        """
        Initialize unified QAlign generator.
        
        Args:
            base_model: Base language model for generation
            scoring_model: Model for scoring (drift or MLE model)
            p_vector: Preference vector
            base_prompt: Base system prompt
            attribute_prompts: List of attribute prompts
            tokenizer: Tokenizer
            num_steps: Number of MCMC steps (default 100)
            beta: Temperature parameter for acceptance probability (default 1.0)
            temperature: Sampling temperature for generation (default 0.8)
            max_length: Maximum generation length (default 256)
            device: Device for computation
            memory_cleanup_frequency: Clean memory every N steps (default 10)
            method_name: Name of the method for logging (default "QAlign")
            sparsify_k: Number of top attributes to keep (default 7, None to disable)
        """
        self.base_model = base_model
        self.scoring_model = scoring_model
        self.base_prompt = base_prompt
        self.tokenizer = tokenizer
        self.num_steps = num_steps
        self.beta = beta
        self.temperature = temperature
        self.max_length = max_length
        self.device = device or 'cpu'
        self.memory_cleanup_frequency = memory_cleanup_frequency
        self.method_name = method_name
        
        # Sparsify p_vector if requested
        if sparsify_k is not None and sparsify_k < len(p_vector):
            abs_values = np.abs(p_vector)
            top_indices = np.argpartition(abs_values, -sparsify_k)[-sparsify_k:]
            
            # Create sparse vector with only top k values
            self.p_vector = p_vector[top_indices]
            self.attribute_prompts = [attribute_prompts[i] for i in top_indices]
            
            print(f"{method_name} sparsified: keeping {sparsify_k} attributes with indices {sorted(top_indices.tolist())}")
            print(f"Active weights: {[f'{i}:{p_vector[i]:.4f}' for i in sorted(top_indices.tolist())]}")
        else:
            self.p_vector = p_vector
            self.attribute_prompts = attribute_prompts
    
    def generate(self, prompt: str) -> str:
        """Generate using QAlign MCMC with scoring and MBR ROUGE-L selection."""
        # Define reward function using the scoring model
        reward_fn = self._create_reward_function()
        
        # Run QAlign MCMC algorithm
        accepted_samples = self._run_qalign_mcmc(prompt, reward_fn)
        
        # Use MBR ROUGE-L to select best response
        best_response = self._mbr_rouge_l_selection(accepted_samples)
        return best_response
    
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for multiple prompts with parallel MCMC chains."""
        print(f"Running {len(prompts)} parallel {self.method_name} MCMC chains...")
        
        # Initialize chains for all prompts with batched initial generation
        print("Generating initial responses in batch...")
        initial_responses = self._generate_responses_batch(prompts)
        
        chains = []
        for i, (prompt, initial_response) in enumerate(zip(prompts, initial_responses)):
            chains.append({
                'prompt_idx': i,
                'prompt': prompt,
                'current_response': initial_response,
                'accepted_samples': [initial_response],
                'acceptance_count': 0
            })
        
        # Run MCMC steps with batched scoring
        with tqdm(total=self.num_steps, desc=f"{self.method_name} MCMC steps for {len(prompts)} prompts") as pbar:
            for step in range(self.num_steps):
                # Generate proposals for all chains in batch
                proposals = self._generate_proposals_batch(chains)
                
                # Batch score all current responses and proposals
                current_responses = [chain['current_response'] for chain in chains]
                valid_proposals = [p if p and len(p.strip()) > 0 else current_responses[i] 
                                for i, p in enumerate(proposals)]
                
                # Score all responses at once - VLLM handles batching
                all_prompts = [chain['prompt'] for chain in chains]
                
                # Score current responses
                current_scores_matrix = self._batch_score_multiple_prompts(
                    all_prompts, [[resp] for resp in current_responses]
                )
                current_scores = [scores[0] for scores in current_scores_matrix]
                
                # Score proposals
                proposal_scores_matrix = self._batch_score_multiple_prompts(
                    all_prompts, [[resp] for resp in valid_proposals]
                )
                proposal_scores = [scores[0] for scores in proposal_scores_matrix]
                
                # Update each chain based on acceptance
                accepted_count = 0
                for i, chain in enumerate(chains):
                    if proposals[i] is None or len(proposals[i].strip()) == 0:
                        chain['accepted_samples'].append(chain['current_response'])
                        continue
                    
                    current_reward = current_scores[i]
                    proposal_reward = proposal_scores[i]
                    
                    # Compute acceptance probability using Eq. 9 from QAlign paper
                    reward_diff = proposal_reward - current_reward
                    current_len = len(self.tokenizer.encode(chain['current_response']))
                    proposal_len = len(self.tokenizer.encode(valid_proposals[i]))
                    
                    if proposal_len > 0:
                        length_ratio = current_len / proposal_len
                        log_acceptance_prob = min(0, reward_diff / self.beta + math.log(length_ratio))
                        acceptance_prob = math.exp(log_acceptance_prob)
                    else:
                        acceptance_prob = 0
                    
                    # Accept or reject
                    if random.random() < acceptance_prob:
                        chain['current_response'] = valid_proposals[i]
                        chain['accepted_samples'].append(valid_proposals[i])
                        chain['acceptance_count'] += 1
                        accepted_count += 1
                    else:
                        chain['accepted_samples'].append(chain['current_response'])
                
                # Update progress bar with acceptance info
                acceptance_rate = accepted_count / len(chains) if len(chains) > 0 else 0
                pbar.set_postfix({
                    'step': f"{step+1}/{self.num_steps}", 
                    'accept_rate': f"{acceptance_rate:.2%}"
                })
                pbar.update(1)
                
                # Clean up memory periodically
                if step % self.memory_cleanup_frequency == 0:
                    import gc
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
                
                # Clear intermediate variables
                del current_scores, proposal_scores, proposals, current_responses, valid_proposals
        
        # Select best response for each chain using MBR ROUGE-L
        results = []
        for chain in chains:
            best_response = self._mbr_rouge_l_selection(chain['accepted_samples'])
            results.append(best_response)
        
        return results
    
    def _create_reward_function(self) -> Callable:
        """Create reward function using the scoring model."""
        from src.core.drift import get_scores
        
        def reward_fn(question: str, response: str) -> float:
            scores = get_scores(
                [(question, [response])],
                self.scoring_model,
                self.p_vector,
                self.base_prompt,
                self.attribute_prompts,
                self.device,
                self.tokenizer
            )[0]
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()
            return float(scores[0])
        
        return reward_fn
    
    def _run_qalign_mcmc(self, prompt: str, reward_fn) -> list:
        """Run QAlign MCMC sampling for a single prompt."""
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
    
    def _generate_responses_batch(self, prompts: List[str]) -> List[str]:
        """Generate initial responses for multiple prompts using VLLM batching."""
        from vllm import SamplingParams
        
        # VLLM handles batching automatically - just pass all prompts
        sampling_params = SamplingParams(
            max_tokens=self.max_length,
            temperature=self.temperature,
            top_p=0.9
        )
        
        # Generate all responses in one batch
        outputs = self.base_model.generate(prompts, sampling_params)
        
        # Extract the generated text
        responses = [output.outputs[0].text.strip() for output in outputs]
        
        return responses
    
    def _generate_proposals_batch(self, chains: List[dict]) -> List[str]:
        """Generate proposals for all chains using VLLM batching and QUEST method."""
        from vllm import SamplingParams
        
        # Prepare all prompts for QUEST proposals
        full_prompts = []
        prefix_info = []  # Store prefix information for reconstruction
        
        for chain in chains:
            prompt = chain['prompt']
            current_response = chain['current_response']
            
            # QUEST method: sample index and create prefix
            current_tokens = self.tokenizer.encode(current_response, add_special_tokens=False)
            
            if len(current_tokens) == 0:
                # If no tokens, generate from scratch
                full_prompts.append(prompt)
                prefix_info.append((prompt, ""))
            else:
                # Sample index uniformly
                idx = random.randint(0, len(current_tokens) - 1)
                prefix_tokens = current_tokens[:idx]
                prefix_text = self.tokenizer.decode(prefix_tokens, skip_special_tokens=True)
                full_prompt = prompt + " " + prefix_text if prefix_text else prompt
                full_prompts.append(full_prompt)
                prefix_info.append((prompt, prefix_text))
        
        # VLLM handles batching automatically - just pass all prompts
        sampling_params = SamplingParams(
            max_tokens=max(1, self.max_length // 2),
            temperature=self.temperature,
            top_p=0.9
        )
        
        # Generate all proposals in one batch
        outputs = self.base_model.generate(full_prompts, sampling_params)
        
        # Extract and reconstruct proposals
        proposals = []
        for output, (_, prefix_text) in zip(outputs, prefix_info):
            completion = output.outputs[0].text.strip()
            
            # Reconstruct full proposal
            if prefix_text:
                proposal = prefix_text + " " + completion
            else:
                proposal = completion
            
            proposals.append(proposal)
        
        return proposals
    
    def _batch_score_multiple_prompts(self, prompts: List[str], responses_lists: List[List[str]]) -> List[List[float]]:
        """Score responses for multiple prompts using VLLM batching."""
        from src.core.drift import get_scores
        
        # Prepare batch data: (prompt, responses) tuples
        batch_data = [(prompt, responses) for prompt, responses in zip(prompts, responses_lists)]
        
        # Score all prompts and their responses - VLLM handles batching internally
        scores_matrix = get_scores(
            batch_data,
            self.scoring_model,
            self.p_vector,
            self.base_prompt,
            self.attribute_prompts,
            self.device,
            self.tokenizer
        )
        
        # Convert to list of lists
        results = []
        for scores in scores_matrix:
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()
            results.append(scores.tolist())
        
        return results
    
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


# Convenience function to create specific QAlign variants
def create_qalign_drift(base_model, drift_model, p_vector, base_prompt, attribute_prompts, tokenizer, **kwargs):
    """Create QAlign generator with drift scoring."""
    return QAlignGenerator(
        base_model=base_model,
        scoring_model=drift_model,
        p_vector=p_vector,
        base_prompt=base_prompt,
        attribute_prompts=attribute_prompts,
        tokenizer=tokenizer,
        method_name="QAlign-Drift",
        **kwargs
    )


def create_qalign_mle(base_model, mle_model, p_vector_mle, base_prompt, attribute_prompts, tokenizer, **kwargs):
    """Create QAlign generator with MLE scoring."""
    return QAlignGenerator(
        base_model=base_model,
        scoring_model=mle_model,
        p_vector=p_vector_mle,
        base_prompt=base_prompt,
        attribute_prompts=attribute_prompts,
        tokenizer=tokenizer,
        method_name="QAlign-MLE",
        **kwargs
    )