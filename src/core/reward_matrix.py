"""
Compute reward matrices for sparse coding from persona responses.
Adapts the drift-based reward computation to matrix format.
"""

import torch
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np

from vllm import LLM
from transformers import AutoTokenizer
from drift import get_log_probs


class RewardMatrixBuilder:
    """Build reward matrix Y for sparse coding from persona responses."""
    
    def __init__(
        self,
        model: LLM,
        tokenizer: AutoTokenizer,
        base_prompt: str,
        attribute_prompts: List[str],
        device: str = "cuda"
    ):
        """
        Initialize reward matrix builder.
        
        Args:
            model: vLLM model for computing log probabilities
            tokenizer: Tokenizer for the model
            base_prompt: Base system prompt (neutral)
            attribute_prompts: List of attribute/persona prompts
            device: Device to use for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.base_prompt = base_prompt
        self.attribute_prompts = attribute_prompts
        self.device = device
    
    def compute_drift_scores(
        self,
        questions: List[str],
        responses: List[str],
        attribute_prompt: str
    ) -> torch.Tensor:
        """
        Compute drift scores for a single attribute.
        
        Args:
            questions: List of questions
            responses: List of responses
            attribute_prompt: Single attribute prompt
            
        Returns:
            Tensor of drift scores (n_samples,)
        """
        n = len(questions)
        
        # Get log probabilities with base prompt
        base_probs, base_counts = get_log_probs(
            self.model,
            self.tokenizer,
            [self.base_prompt] * n,
            questions,
            responses,
            self.device
        )
        
        # Get log probabilities with attribute prompt
        attr_probs, attr_counts = get_log_probs(
            self.model,
            self.tokenizer,
            [attribute_prompt] * n,
            questions,
            responses,
            self.device
        )
        
        # Convert to tensors and normalize
        base_tensor = torch.tensor(base_probs, device=self.device) / torch.tensor(base_counts, device=self.device)
        attr_tensor = torch.tensor(attr_probs, device=self.device) / torch.tensor(attr_counts, device=self.device)
        
        # Compute drift: difference in log probabilities
        drift = attr_tensor - base_tensor
        
        return drift
    
    def compute_reward_matrix(
        self,
        persona_responses_dir: str,
        num_personas: Optional[int] = None,
        num_questions: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute full reward matrix Y from persona responses.
        
        Args:
            persona_responses_dir: Directory containing persona response files
            num_personas: Number of personas to process (None = all)
            num_questions: Number of questions per persona (None = all)
            
        Returns:
            Y: Reward matrix (d × U) where d = datapoints, U = personas
            metadata: Dictionary with matrix metadata
        """
        responses_dir = Path(persona_responses_dir)
        
        # Discover available personas
        persona_dirs = sorted([
            d for d in responses_dir.iterdir()
            if d.is_dir() and d.name.startswith("persona_")
        ])
        
        if num_personas:
            persona_dirs = persona_dirs[:num_personas]
        
        print(f"Processing {len(persona_dirs)} personas")
        
        # Load first persona to get dimensions
        with open(persona_dirs[0] / "responses.json", 'r') as f:
            first_data = json.load(f)
        
        all_questions = [r["question"] for r in first_data["responses"]]
        if num_questions:
            all_questions = all_questions[:num_questions]
        
        d = len(all_questions)  # Number of datapoints
        U = len(persona_dirs)    # Number of personas
        K = len(self.attribute_prompts)  # Number of attributes
        
        print(f"Matrix dimensions: {d} datapoints × {U} personas")
        print(f"Computing drift scores for {K} attributes")
        
        # Initialize reward matrix
        Y = torch.zeros((d, U), device=self.device)
        
        # Process each persona
        for persona_idx, persona_dir in enumerate(tqdm(persona_dirs, desc="Processing personas")):
            # Load persona responses
            with open(persona_dir / "responses.json", 'r') as f:
                persona_data = json.load(f)
            
            persona_prompt = persona_data["persona_prompt"]
            responses = [r["response"] for r in persona_data["responses"]]
            
            if num_questions:
                responses = responses[:num_questions]
            
            # Compute drift scores for this persona
            # We treat the persona prompt itself as the attribute prompt
            drift_scores = self.compute_drift_scores(
                all_questions,
                responses,
                persona_prompt
            )
            
            Y[:, persona_idx] = drift_scores
        
        # Create metadata
        metadata = {
            "num_datapoints": d,
            "num_personas": U,
            "num_attributes": K,
            "questions": all_questions,
            "persona_prompts": [json.load(open(pd / "responses.json"))["persona_prompt"] 
                               for pd in persona_dirs]
        }
        
        return Y, metadata
    
    def compute_attribute_reward_matrix(
        self,
        questions: List[str],
        responses_per_question: List[List[str]]
    ) -> torch.Tensor:
        """
        Compute reward matrix where each column is an attribute's drift scores.
        
        Args:
            questions: List of questions
            responses_per_question: List of response lists (one per question)
            
        Returns:
            Y: Reward matrix (d × K) where d = datapoints, K = attributes
        """
        # Flatten questions and responses for batch processing
        flat_questions = []
        flat_responses = []
        
        for question, response_list in zip(questions, responses_per_question):
            for response in response_list:
                flat_questions.append(question)
                flat_responses.append(response)
        
        d = len(flat_questions)  # Total datapoints
        K = len(self.attribute_prompts)  # Number of attributes
        
        print(f"Computing attribute reward matrix: {d} datapoints × {K} attributes")
        
        # Initialize reward matrix
        Y = torch.zeros((d, K), device=self.device)
        
        # Compute drift scores for each attribute
        for attr_idx, attr_prompt in enumerate(tqdm(self.attribute_prompts, desc="Processing attributes")):
            drift_scores = self.compute_drift_scores(
                flat_questions,
                flat_responses,
                attr_prompt
            )
            Y[:, attr_idx] = drift_scores
        
        return Y
    
    def save_reward_matrix(
        self,
        Y: torch.Tensor,
        metadata: Dict,
        save_path: str
    ):
        """Save reward matrix and metadata."""
        save_dict = {
            "reward_matrix": Y.cpu(),
            "metadata": metadata
        }
        
        torch.save(save_dict, save_path)
        print(f"Saved reward matrix to {save_path}")
        print(f"  Shape: {Y.shape}")
        print(f"  Size: {Y.element_size() * Y.nelement() / 1e6:.2f} MB")
    
    @staticmethod
    def load_reward_matrix(load_path: str) -> Tuple[torch.Tensor, Dict]:
        """Load reward matrix and metadata."""
        checkpoint = torch.load(load_path)
        return checkpoint["reward_matrix"], checkpoint["metadata"]