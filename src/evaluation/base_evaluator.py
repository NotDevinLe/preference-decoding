#!/usr/bin/env python3
"""
Base evaluator class for unified evaluation framework.
All evaluation methods (BON, QAlign, Drift Decoding) inherit from this.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    method_name: str
    num_prompts: int
    scores: List[float]
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    median_score: float
    percentile_25: float
    percentile_75: float
    evaluation_time: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def summary(self) -> str:
        """Get summary string."""
        return (f"{self.method_name}: "
                f"Mean={self.mean_score:.3f} (±{self.std_score:.3f}), "
                f"Median={self.median_score:.3f}, "
                f"Min={self.min_score:.3f}, Max={self.max_score:.3f}")


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluation methods.
    
    The evaluation process is:
    1. Get responses (either select from BON or generate)
    2. Judge responses using the same judge for all methods
    3. Compute metrics and return results
    """
    
    def __init__(self, judge, method_name: str = "Base"):
        """
        Initialize evaluator.
        
        Args:
            judge: Judge instance (PersonaJudge, GoldenRewardJudge, etc.)
            method_name: Name of the evaluation method
        """
        self.judge = judge
        self.method_name = method_name
        self.evaluation_cache = {}
    
    @abstractmethod
    def get_responses(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Get responses for given prompts.
        
        This is the key method that differs between evaluation types:
        - BON methods: Select from pre-generated candidates
        - Generation methods: Generate new responses
        
        Args:
            prompts: List of prompts to get responses for
            **kwargs: Method-specific parameters
            
        Returns:
            List of responses, one per prompt
        """
        raise NotImplementedError("Subclasses must implement get_responses")
    
    def judge_responses(
        self, 
        prompts: List[str], 
        responses: List[str],
        use_cache: bool = True
    ) -> List[float]:
        """
        Judge responses using the configured judge.
        
        Args:
            prompts: List of prompts
            responses: List of responses to judge
            use_cache: Whether to use cached judgments
            
        Returns:
            List of scores
        """
        scores = []
        
        for prompt, response in zip(prompts, responses):
            # Create cache key
            cache_key = f"{prompt}||{response}"
            
            # Check cache
            if use_cache and cache_key in self.evaluation_cache:
                score = self.evaluation_cache[cache_key]
            else:
                # Judge the response
                score = self._judge_single(prompt, response)
                
                # Cache the result
                if use_cache:
                    self.evaluation_cache[cache_key] = score
            
            scores.append(score)
        
        return scores
    
    def _judge_single(self, prompt: str, response: str) -> float:
        """
        Judge a single response.
        
        Args:
            prompt: The prompt
            response: The response to judge
            
        Returns:
            Score (float)
        """
        # Use standard judge interface
        if hasattr(self.judge, 'score'):
            return self.judge.score(prompt, response)
        else:
            raise ValueError(f"Unknown judge type: {type(self.judge)}")
    
    def compute_metrics(self, scores: List[float]) -> EvaluationResult:
        """
        Compute evaluation metrics from scores.
        
        Args:
            scores: List of scores
            
        Returns:
            EvaluationResult object
        """
        scores_array = np.array(scores)
        
        return EvaluationResult(
            method_name=self.method_name,
            num_prompts=len(scores),
            scores=scores,
            mean_score=float(np.mean(scores_array)),
            std_score=float(np.std(scores_array)),
            min_score=float(np.min(scores_array)),
            max_score=float(np.max(scores_array)),
            median_score=float(np.median(scores_array)),
            percentile_25=float(np.percentile(scores_array, 25)),
            percentile_75=float(np.percentile(scores_array, 75)),
            evaluation_time=0.0,  # Will be set by evaluate()
            metadata={}
        )
    
    def evaluate(
        self,
        prompts: List[str],
        progress_callback: Optional[callable] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Main evaluation method.
        
        Args:
            prompts: List of prompts to evaluate
            progress_callback: Optional callback for progress updates
            **kwargs: Method-specific parameters
            
        Returns:
            EvaluationResult object
        """
        start_time = time.time()
        
        # Step 1: Get responses (method-specific)
        if progress_callback:
            progress_callback(0, len(prompts), "Getting responses...")
        
        responses = self.get_responses(prompts, **kwargs)
        
        if len(responses) != len(prompts):
            raise ValueError(f"Expected {len(prompts)} responses, got {len(responses)}")
        
        # Step 2: Judge responses (same for all methods)
        if progress_callback:
            progress_callback(len(prompts)//2, len(prompts), "Judging responses...")
        
        scores = self.judge_responses(prompts, responses)
        
        # Step 3: Compute metrics
        if progress_callback:
            progress_callback(len(prompts), len(prompts), "Computing metrics...")
        
        result = self.compute_metrics(scores)
        result.evaluation_time = time.time() - start_time
        
        # Add method-specific metadata
        result.metadata = self.get_metadata()
        
        return result
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get method-specific metadata.
        Subclasses can override to add custom metadata.
        
        Returns:
            Dictionary of metadata
        """
        return {
            "method_class": self.__class__.__name__,
            "judge_type": type(self.judge).__name__
        }
    
    def save_results(self, result: EvaluationResult, filepath: str):
        """
        Save evaluation results to file.
        
        Args:
            result: EvaluationResult object
            filepath: Path to save results
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f"Results saved to: {filepath}")
    
    def load_results(self, filepath: str) -> EvaluationResult:
        """
        Load evaluation results from file.
        
        Args:
            filepath: Path to results file
            
        Returns:
            EvaluationResult object
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert dict back to EvaluationResult
        return EvaluationResult(**data)


class OracleEvaluator(BaseEvaluator):
    """
    Oracle evaluator that always returns the best possible response.
    Used as an upper bound for comparison.
    """
    
    def __init__(self, judge, bon_data: Dict[str, List[str]]):
        """
        Initialize oracle evaluator.
        
        Args:
            judge: Judge instance
            bon_data: Dictionary mapping prompts to candidate responses
        """
        super().__init__(judge, method_name="Oracle")
        self.bon_data = bon_data
    
    def get_responses(self, prompts: List[str], n: int = 100, **kwargs) -> List[str]:
        """
        Get the best possible response for each prompt.
        
        Args:
            prompts: List of prompts
            n: Number of candidates to consider
            
        Returns:
            List of best responses
        """
        best_responses = []
        
        for prompt in prompts:
            if prompt not in self.bon_data:
                raise ValueError(f"Prompt not found in BON data: {prompt}")
            
            # Get candidates
            candidates = self.bon_data[prompt][:n]
            
            # Score all candidates
            scores = []
            for candidate in candidates:
                score = self._judge_single(prompt, candidate)
                scores.append(score)
            
            # Select best
            best_idx = np.argmax(scores)
            best_responses.append(candidates[best_idx])
        
        return best_responses


class RandomEvaluator(BaseEvaluator):
    """
    Random evaluator that selects responses randomly.
    Used as a lower bound baseline.
    """
    
    def __init__(self, judge, bon_data: Dict[str, List[str]], seed: Optional[int] = None):
        """
        Initialize random evaluator.
        
        Args:
            judge: Judge instance
            bon_data: Dictionary mapping prompts to candidate responses
            seed: Random seed for reproducibility
        """
        super().__init__(judge, method_name="Random")
        self.bon_data = bon_data
        self.rng = np.random.RandomState(seed)
    
    def get_responses(self, prompts: List[str], n: int = 100, **kwargs) -> List[str]:
        """
        Get random responses for each prompt.
        
        Args:
            prompts: List of prompts
            n: Number of candidates to consider
            
        Returns:
            List of randomly selected responses
        """
        random_responses = []
        
        for prompt in prompts:
            if prompt not in self.bon_data:
                raise ValueError(f"Prompt not found in BON data: {prompt}")
            
            # Get candidates
            candidates = self.bon_data[prompt][:n]
            
            # Select random
            random_idx = self.rng.randint(len(candidates))
            random_responses.append(candidates[random_idx])
        
        return random_responses