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
    def __init__(self, vector: List[float]):
        self.vector = vector
        super().__init__(f"vector:{self.vector}")
        
    def evaluate(self, conversations: List[List[Dict[str, str]]],
        gateway_url,
        tokenizer,
        base_prompt,
        attribute_prompts,
        model_name,
        device,
    ) -> List[float]:


        prompts = [conversation[0]['content'] for conversation in conversations]
        outputs = [conversation[1]['content'] for conversation in conversations]
        
        return await compute_drift_rewards(
            gateway_url=gateway_url,
            tokenizer=tokenizer,
            prompts=prompts,
            outputs=outputs,
            base_prompt=base_prompt,
            attribute_prompts=attribute_prompts,
            model_name=model_name,
            device=device,
        )
