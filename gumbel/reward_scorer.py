import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging
import requests
import asyncio
import aiohttp

class VLLMRewardScorer:
    """
    HTTP client for VLLM preference scoring server.
    Makes requests to the standalone VLLM server.
    """
    
    def __init__(self, 
                 server_url: str = "http://localhost:8000",
                 base_prompt: str = "You are a helpful assistant.",
                 attribute_prompts: List[str] = None,
                 timeout: float = 30.0):
        """
        Initialize the reward scorer client.
        
        Args:
            server_url: URL of the VLLM server
            base_prompt: Base system prompt for neutral scoring
            attribute_prompts: List of attribute-specific system prompts
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip('/')
        self.base_prompt = base_prompt
        self.attribute_prompts = attribute_prompts or []
        self.timeout = timeout
        self.num_attributes = len(self.attribute_prompts)
        
        # Test connection
        self._test_connection()
        
        logging.info(f"Initialized VLLMRewardScorer client for {server_url}")
        logging.info(f"Base prompt: {base_prompt[:50]}...")
        logging.info(f"Number of attributes: {self.num_attributes}")
    
    def _test_connection(self):
        """Test connection to the VLLM server"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5.0)
            if response.status_code == 200:
                health_data = response.json()
                logging.info(f"VLLM server health: {health_data}")
            else:
                logging.warning(f"VLLM server health check failed: {response.status_code}")
        except Exception as e:
            logging.error(f"Cannot connect to VLLM server at {self.server_url}: {e}")
            raise ConnectionError(f"VLLM server not available: {e}")
    
    def score_batch(self, 
                    prompts: List[str], 
                    outputs: List[str],
                    attribute_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute reward scores for a batch of (prompt, output) pairs.
        
        Args:
            prompts: List of user prompts
            outputs: List of model outputs (one per prompt)
            attribute_weights: [num_attributes] tensor of weights p_i for each attribute
            
        Returns:
            torch.Tensor: [batch_size] tensor of reward scores
        """
        batch_size = len(prompts)
        assert len(outputs) == batch_size, f"Prompt/output length mismatch: {len(prompts)} vs {len(outputs)}"
        assert len(attribute_weights) == self.num_attributes, f"Weight dimension mismatch: {len(attribute_weights)} vs {self.num_attributes}"
        
        if batch_size == 0:
            return torch.empty(0)
        
        # Prepare request
        request_data = {
            "prompts": prompts,
            "outputs": outputs,
            "attribute_weights": attribute_weights.tolist(),
            "base_prompt": self.base_prompt,
            "attribute_prompts": self.attribute_prompts
        }
        
        try:
            # Make HTTP request to VLLM server
            response = requests.post(
                f"{self.server_url}/score_preferences",
                json=request_data,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logging.error(f"VLLM server request failed: {response.status_code}")
                logging.error(f"Response: {response.text}")
                # Return dummy scores on error
                return torch.randn(batch_size)
            
            result = response.json()
            
            if not result["success"]:
                logging.error(f"VLLM server error: {result.get('error', 'Unknown error')}")
                return torch.randn(batch_size)
            
            # Convert scores back to tensor
            scores = torch.tensor(result["scores"], dtype=torch.float32)
            
            logging.debug(f"Received {len(scores)} scores from VLLM server")
            logging.debug(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
            
            return scores
            
        except Exception as e:
            logging.error(f"Error communicating with VLLM server: {e}")
            # Return dummy scores on error
            return torch.randn(batch_size)


def score_rewards(X: torch.Tensor, m_hard: torch.Tensor, 
                 scorer: VLLMRewardScorer,
                 user_data: Dict[str, Any]) -> torch.Tensor:
    """
    Compute rewards for a batch of user data with attribute masking.
    This replaces the stub function in utils.py.
    
    Args:
        X: [batch_size, d] tensor of user features/embeddings
        m_hard: [d] binary mask indicating which attributes are active
        scorer: VLLMRewardScorer instance
        user_data: Dictionary containing:
            - 'prompts': List[str] of user prompts
            - 'outputs': List[str] of model outputs
            - 'user_ids': List[str] of user identifiers (optional)
    
    Returns:
        torch.Tensor: [batch_size] tensor of reward scores
    """
    batch_size = X.shape[0]
    d = X.shape[1]  # number of attributes
    
    # Extract data
    prompts = user_data['prompts']
    outputs = user_data['outputs'] 
    
    assert len(prompts) == batch_size, f"Prompt batch size mismatch: {len(prompts)} vs {batch_size}"
    assert len(outputs) == batch_size, f"Output batch size mismatch: {len(outputs)} vs {batch_size}"
    assert len(m_hard) == d, f"Mask dimension mismatch: {len(m_hard)} vs {d}"
    
    # Create attribute weights from the hard mask
    # Only active attributes (m_hard[i] == 1) get weight, others get 0
    attribute_weights = m_hard.float()  # [d]
    
    # Compute rewards using the scorer
    rewards = scorer.score_batch(prompts, outputs, attribute_weights)
    
    logging.debug(f"Computed rewards for {batch_size} items, mask sparsity: {m_hard.sum().item()}/{d}")
    
    return rewards


def create_dummy_user_data(batch_size: int, device: str = "cpu") -> Dict[str, Any]:
    """
    Create dummy user data for testing purposes.
    In practice, this would come from your actual dataset.
    """
    import random
    
    # Generate dummy prompts and outputs
    dummy_prompts = [
        f"What are your thoughts on topic {i}?" 
        for i in range(batch_size)
    ]
    
    dummy_outputs = [
        f"I think topic {i} is very interesting and worth discussing further." 
        for i in range(batch_size)
    ]
    
    return {
        'prompts': dummy_prompts,
        'outputs': dummy_outputs,
        'user_ids': [f"user_{i}" for i in range(batch_size)]
    }


# Example usage and testing
if __name__ == "__main__":
    # This is for testing - in practice, the scorer would be created in the main process
    logging.basicConfig(level=logging.DEBUG)
    
    # Dummy test
    batch_size = 4
    d = 10  # number of attributes
    
    # Create dummy data
    X = torch.randn(batch_size, d)  # User feature embeddings
    m_hard = torch.randint(0, 2, (d,)).float()  # Random binary mask
    user_data = create_dummy_user_data(batch_size)
    
    print(f"Testing reward scoring:")
    print(f"Batch size: {batch_size}")
    print(f"Number of attributes: {d}")
    print(f"Active attributes: {m_hard.sum().item()}")
    print(f"Mask: {m_hard}")
    
    # Note: This would fail without actual VLLM model
    # scorer = VLLMRewardScorer(model=None, tokenizer=None, 
    #                          base_prompt="", attribute_prompts=[""] * d)
    # rewards = score_rewards(X, m_hard, scorer, user_data)
    # print(f"Rewards: {rewards}")