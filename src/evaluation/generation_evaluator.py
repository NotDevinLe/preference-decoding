#!/usr/bin/env python3
"""
Generation evaluator for methods that generate new responses.
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import torch
import numpy as np
from tqdm import tqdm

from src.evaluation.base_evaluator import BaseEvaluator


class Generator(ABC):
    """Abstract base class for generation strategies."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: The prompt to generate a response for
            
        Returns:
            Generated response
        """
        raise NotImplementedError


class GenerationEvaluator(BaseEvaluator):
    """
    Evaluator for generation methods.
    Generates new responses instead of selecting from candidates.
    """
    
    def __init__(
        self,
        judge,
        generator: Generator,
        method_name: str = "Generation"
    ):
        """
        Initialize generation evaluator.
        
        Args:
            judge: Judge instance for scoring
            generator: Generator instance for creating responses
            method_name: Name of the evaluation method
        """
        super().__init__(judge, method_name)
        self.generator = generator
    
    def get_responses(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Get responses by generating them.
        
        Args:
            prompts: List of prompts
            **kwargs: Additional parameters for generation
            
        Returns:
            List of generated responses
        """
        # Check if generator supports batch generation
        if hasattr(self.generator, 'generate_batch'):
            print(f"Using batch generation for {len(prompts)} prompts...")
            return self.generator.generate_batch(prompts)
        else:
            # Fall back to sequential generation
            generated_responses = []
            
            for prompt in tqdm(prompts, desc="Generating responses"):
                # Generate response
                response = self.generator.generate(prompt)
                generated_responses.append(response)
            
            return generated_responses
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get method-specific metadata."""
        metadata = super().get_metadata()
        metadata.update({
            "generator_type": type(self.generator).__name__
        })
        return metadata
    

class DriftDecodingGenerator(Generator):
    """Generator using drift decoding with logits processor."""
    
    def __init__(
        self,
        base_model,
        drift_logits_processor,
        tokenizer,
        max_length: int = 512,
        temperature: float = 0.7,
        device=None
    ):
        """
        Initialize drift decoding generator.
        
        Args:
            base_model: Base language model
            drift_logits_processor: DriftLogitsProcessor instance
            tokenizer: Tokenizer
            max_length: Maximum generation length
            temperature: Generation temperature
            device: Device for computation
        """
        self.base_model = base_model
        self.drift_logits_processor = drift_logits_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.temperature = temperature
        self.device = device or 'cpu'
    
    def generate(self, prompt: str) -> str:
        """Generate using drift decoding."""
        from transformers import LogitsProcessorList
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Create logits processor list
        logits_processor = LogitsProcessorList([self.drift_logits_processor])
        
        # Generate with drift logits processor
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_length=self.max_length,
                temperature=self.temperature,
                do_sample=True,
                logits_processor=logits_processor,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove prompt from response
        response = response[len(prompt):].strip()
        
        return response


class VLLMGenerator(Generator):
    """Generator using VLLM for fast inference."""
    
    def __init__(
        self,
        model_name: str,
        sampling_params=None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize VLLM generator.
        
        Args:
            model_name: Name/path of the model
            sampling_params: VLLM SamplingParams
            system_prompt: Optional system prompt
        """
        from vllm import LLM, SamplingParams
        
        self.model = LLM(model=model_name, tensor_parallel_size=1)
        self.sampling_params = sampling_params or SamplingParams(
            temperature=0.7,
            max_tokens=512,
            top_p=0.9
        )
        self.system_prompt = system_prompt
    
    def generate(self, prompt: str) -> str:
        """Generate using VLLM."""
        # Add system prompt if provided
        if self.system_prompt:
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        # Generate
        outputs = self.model.generate([full_prompt], self.sampling_params)
        
        # Extract response
        response = outputs[0].outputs[0].text
        return response