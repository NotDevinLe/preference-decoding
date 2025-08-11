#!/usr/bin/env python3
"""
BON (Best-of-N) evaluator for selecting from pre-generated responses.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from abc import ABC, abstractmethod

from src.evaluation.base_evaluator import BaseEvaluator


class Selector(ABC):
    """Abstract base class for selection strategies."""
    
    @abstractmethod
    def select(self, prompt: str, candidates: List[str]) -> str:
        """
        Select the best candidate from a list.
        
        Args:
            prompt: The prompt
            candidates: List of candidate responses
            
        Returns:
            Selected response
        """
        raise NotImplementedError


class BONEvaluator(BaseEvaluator):
    """
    Evaluator for BON (Best-of-N) methods.
    Selects from pre-generated candidate responses.
    """
    
    def __init__(
        self,
        judge,
        selector: Selector,
        bon_data: Dict[str, List[str]],
        method_name: str = "BON"
    ):
        """
        Initialize BON evaluator.
        
        Args:
            judge: Judge instance for scoring
            selector: Selector instance for choosing best response
            bon_data: Dictionary mapping prompts to lists of candidate responses
            method_name: Name of the evaluation method
        """
        super().__init__(judge, method_name)
        self.selector = selector
        self.bon_data = bon_data
    
    def get_responses(self, prompts: List[str], n: int = 100, **kwargs) -> List[str]:
        """
        Get responses by selecting from BON candidates.
        
        Args:
            prompts: List of prompts
            n: Number of candidates to consider (best-of-n)
            **kwargs: Additional parameters
            
        Returns:
            List of selected responses
        """
        # Check if selector supports batch processing
        if hasattr(self.selector, 'select_batch'):
            # Use batch processing for better performance
            candidates_lists = []
            
            for prompt in prompts:
                if prompt not in self.bon_data:
                    raise ValueError(f"Prompt not found in BON data: {prompt[:50]}...")
                
                # Get n candidates
                candidates = self.bon_data[prompt][:n]
                
                if not candidates:
                    raise ValueError(f"No candidates found for prompt: {prompt[:50]}...")
                
                candidates_lists.append(candidates)
            
            # Use batch selection
            selected_responses = self.selector.select_batch(prompts, candidates_lists)
        else:
            # Fall back to sequential processing
            selected_responses = []
            
            for prompt in prompts:
                if prompt not in self.bon_data:
                    raise ValueError(f"Prompt not found in BON data: {prompt[:50]}...")
                
                # Get n candidates
                candidates = self.bon_data[prompt][:n]
                
                if not candidates:
                    raise ValueError(f"No candidates found for prompt: {prompt[:50]}...")
                
                # Use selector to choose best response
                selected = self.selector.select(prompt, candidates)
                selected_responses.append(selected)
        
        return selected_responses
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get method-specific metadata."""
        metadata = super().get_metadata()
        metadata.update({
            "selector_type": type(self.selector).__name__,
            "bon_data_size": len(self.bon_data),
            "max_candidates": max(len(v) for v in self.bon_data.values()) if self.bon_data else 0
        })
        return metadata


# Concrete Selector Implementations

class PreferenceVectorSelector(Selector):
    """Unified selector using preference vector scores for selection."""
    
    def __init__(self, model, p_vector, base_prompt, attribute_prompts, tokenizer, device=None):
        """
        Initialize preference vector selector.
        
        Args:
            model: VLLM model for computing scores
            p_vector: Preference vector (numpy array or torch tensor)
            base_prompt: Base system prompt
            attribute_prompts: List of attribute prompts
            tokenizer: Tokenizer for the model
            device: Device for computation
        """
        self.model = model
        self.p_vector = p_vector
        self.base_prompt = base_prompt
        self.attribute_prompts = attribute_prompts
        self.tokenizer = tokenizer
        self.device = device
    
    def select(self, prompt: str, candidates: List[str]) -> str:
        """Select using preference vector scores."""
        # Import here to avoid circular dependency
        from src.core.drift import get_scores
        
        # Compute preference vector scores for all candidates
        scores = get_scores(
            [(prompt, candidates)],
            self.model,
            self.p_vector,
            self.base_prompt,
            self.attribute_prompts,
            self.device,
            self.tokenizer
        )[0]  # Get first (and only) result
        
        # Select candidate with highest score
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()
        best_idx = np.argmax(scores)
        return candidates[best_idx]
    
    def select_batch(self, prompts: List[str], candidates_lists: List[List[str]]) -> List[str]:
        """Select using preference vector scores for multiple prompts at once."""
        # Import here to avoid circular dependency
        from src.core.drift import get_scores
        
        # Prepare data for batch processing
        batch_data = [(prompt, candidates) for prompt, candidates in zip(prompts, candidates_lists)]
        
        # Compute preference vector scores for all prompts and candidates
        scores_matrix = get_scores(
            batch_data,
            self.model,
            self.p_vector,
            self.base_prompt,
            self.attribute_prompts,
            self.device,
            self.tokenizer
        )
        
        # Select best candidate for each prompt
        selected_responses = []
        for i, (prompt, candidates) in enumerate(zip(prompts, candidates_lists)):
            scores = scores_matrix[i]
            
            # Convert to numpy if needed
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()
            
            best_idx = np.argmax(scores)
            selected_responses.append(candidates[best_idx])
        
        return selected_responses


# Legacy aliases for backward compatibility
DriftSelector = PreferenceVectorSelector
MLESelector = PreferenceVectorSelector


class RewardModelSelector(Selector):
    """Selector using a trained reward model."""
    
    def __init__(self, reward_model, tokenizer, device=None):
        """
        Initialize reward model selector.
        
        Args:
            reward_model: Trained reward model
            tokenizer: Tokenizer for the reward model
            device: Device for computation
        """
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = device or 'cpu'
    
    def select(self, prompt: str, candidates: List[str]) -> str:
        """Select using reward model scores."""
        import torch
        
        scores = []
        for candidate in candidates:
            # Format prompt and response for reward model
            text = self.format_for_reward_model(prompt, candidate)
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Get reward score
            with torch.no_grad():
                outputs = self.reward_model(**inputs)
                if hasattr(outputs, 'logits'):
                    score = outputs.logits[0].item()
                else:
                    score = outputs[0].item()
            
            scores.append(score)
        
        # Select best
        best_idx = np.argmax(scores)
        return candidates[best_idx]
    
    def format_for_reward_model(self, prompt: str, response: str) -> str:
        """Format prompt and response for reward model."""
        # This should match the format used during reward model training
        # Adjust based on your specific format
        return f"<|user|>\n{prompt}\n<|assistant|>\n{response}"


class FirstOutputSelector(Selector):
    """Simple selector that always chooses the first output (for baseline)."""
    
    def select(self, prompt: str, candidates: List[str]) -> str:
        """Always select the first candidate."""
        return candidates[0]