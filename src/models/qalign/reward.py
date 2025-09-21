from typing import List, Callable
import numpy as np
from typing import List, Optional, Dict
import requests
import aiohttp
import asyncio
from src.models.qalign.list_utils import chunked
from itertools import islice
import os
from typing import List
import gc

import numpy as np

from transformers import AutoModelForSequenceClassification
#fix_loggers(name="transformers")

import torch
from typing import Dict
from torch import nn

from src.core.drift import compute_drift_rewards


DEBUG = os.getenv("DEBUG", "False").lower() == "true"
class Reward:
    """
    The base class for reward evaluation.

    Attributes:
        None

    Methods:
        evaluate: Evaluates the reward for a list of candidates.

    """

    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name.replace("/", "-").split(".")[0]

    def evaluate(
        self,
       conversations: List[List[Dict[str, str]]],
        **kwargs,
    ) -> List[float]:
        """
        Evaluates the reward for a list of candidates.

        Args:
            candidates (List[str]): A list of candidate strings.
            **kwargs: Additional keyword arguments.

        Returns:
            List[float]: A list of reward values for each candidate.

        Raises:
            NotImplementedError: This method should be implemented in the derived classes.

        """
        raise NotImplementedError


class ConstantReward(Reward):
    """
    A class for a constant reward.

    Attributes:
        reward (float): The reward value.

    Methods:
        evaluate: Evaluates the reward for a list of candidates.

    """

    def __init__(self, reward: float):
        """
        The constructor for ConstantReward class.

        Args:
            reward (float): The reward value.

        """
        self.reward = reward
        super().__init__(f"constant:{self.reward}")

    def evaluate(
        self,
        conversations: List[List[Dict[str, str]]],
        
        **kwargs,
    ) -> List[float]:
        """
        Evaluates the reward for a list of candidates.

        Args:
            candidates (List[str]): A list of candidate strings.
            **kwargs: Additional keyword arguments.

        Returns:
            List[float]: A list of reward values for each candidate.

        """
 

        return [self.reward for _ in range(len(conversations))]

    def set_context(self, *args, **kwargs):
        pass

class VectorReward(Reward):
    """
    A class for a vector reward.
    """
    def __init__(self, vector: List[float], attribute_prompts: List[str]):
        self.vector = vector
        self.attribute_prompts = attribute_prompts
        super().__init__(f"vector:{self.vector}")
        
    def evaluate(self, conversations: List[List[Dict[str, str]]],
        gateway_url,
        tokenizer,
        base_prompt,
        model_name,
        device,
    ) -> List[float]:
        """
        Evaluates drift rewards for conversations.
        
        Args:
            conversations: List of conversations, each containing [user_msg, assistant_msg]
            gateway_url: URL of the VLLM-compatible gateway
            tokenizer: Tokenizer for the model
            base_prompt: Base system prompt
            attribute_prompts: List of attribute system prompts
            model_name: Model identifier
            device: PyTorch device
            
        Returns:
            List[float]: Scalar rewards (one per conversation), preserving input order
        """
        import asyncio
        import torch
        
        # Extract prompts and outputs while preserving order
        prompts = []
        outputs = []
        for i, conversation in enumerate(conversations):
            if len(conversation) >= 2:
                prompts.append(conversation[0]['content'])  # User message
                outputs.append(conversation[1]['content'])  # Assistant message
            else:
                # Handle incomplete conversations gracefully
                prompts.append(conversation[0]['content'] if len(conversation) > 0 else "")
                outputs.append("")
        
        async def _async_compute_rewards():
            # Call the async compute_drift_rewards function
            reward_matrix = await compute_drift_rewards(
                gateway_url=gateway_url,
                tokenizer=tokenizer,
                prompts=prompts,
                outputs=outputs,
                base_prompt=base_prompt,
                attribute_prompts=self.attribute_prompts,
                model_name=model_name,
                device=device,
            )
            
            vector_tensor = torch.tensor(self.vector, device=device, dtype=torch.float32)
            scalar_rewards = torch.sum(reward_matrix * vector_tensor, dim=1)
            
            return scalar_rewards.tolist()
        
        # Handle async execution from sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, use a thread pool
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _async_compute_rewards())
                    result = future.result()
            else:
                result = loop.run_until_complete(_async_compute_rewards())
        except RuntimeError:
            # No event loop exists, create a new one
            result = asyncio.run(_async_compute_rewards())
        
        # Verify result integrity - throw exception if unexpected behavior
        if len(result) != len(conversations):
            raise ValueError(
                f"Reward computation returned {len(result)} results for {len(conversations)} conversations. "
                f"This indicates a critical error in order preservation or computation."
            )
        
        # Verify all results are valid numbers
        for i, reward in enumerate(result):
            if not isinstance(reward, (int, float)) or not torch.isfinite(torch.tensor(reward)):
                raise ValueError(
                    f"Invalid reward value {reward} at index {i}. "
                    f"All rewards must be finite numbers."
                )
        
        return result
