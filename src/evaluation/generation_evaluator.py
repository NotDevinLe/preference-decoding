#!/usr/bin/env python3
"""
Generation evaluator for methods that generate new responses.
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import torch
import numpy as np

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
        generated_responses = []
        
        for i, prompt in enumerate(prompts):
            # Generate response
            response = self.generator.generate(prompt)
            generated_responses.append(response)
            
            # Optional progress tracking
            if i % 10 == 0:
                print(f"Generated {i+1}/{len(prompts)} responses")
        
        return generated_responses
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get method-specific metadata."""
        metadata = super().get_metadata()
        metadata.update({
            "generator_type": type(self.generator).__name__
        })
        return metadata


# Concrete Generator Implementations

class QAlignDriftGenerator(Generator):
    """Generator using QAlign with drift scoring."""
    
    def __init__(
        self,
        base_model,
        drift_model,
        p_vector,
        base_prompt,
        attribute_prompts,
        tokenizer,
        num_samples: int = 32,
        temperature: float = 1.0,
        max_length: int = 512,
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
            num_samples: Number of samples to generate for QAlign
            temperature: Sampling temperature
            max_length: Maximum generation length
            device: Device for computation
        """
        self.base_model = base_model
        self.drift_model = drift_model
        self.p_vector = p_vector
        self.base_prompt = base_prompt
        self.attribute_prompts = attribute_prompts
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.temperature = temperature
        self.max_length = max_length
        self.device = device or 'cpu'
    
    def generate(self, prompt: str) -> str:
        """Generate using QAlign with drift scoring."""
        from src.core.drift import get_scores
        
        # Generate multiple samples
        samples = []
        for _ in range(self.num_samples):
            # Generate sample with base model
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.base_model.generate(
                    **inputs,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the response
            response = response[len(prompt):].strip()
            samples.append(response)
        
        # Score all samples with drift
        scores = get_scores(
            [(prompt, samples)],
            self.drift_model,
            self.p_vector,
            self.base_prompt,
            self.attribute_prompts,
            self.device,
            self.tokenizer
        )[0]
        
        # Select best sample
        best_idx = np.argmax(scores)
        return samples[best_idx]


class QAlignMLEGenerator(Generator):
    """Generator using QAlign with MLE scoring."""
    
    def __init__(
        self,
        base_model,
        mle_model,
        p_vector_mle,
        base_prompt,
        attribute_prompts,
        tokenizer,
        num_samples: int = 32,
        temperature: float = 1.0,
        max_length: int = 512,
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
            num_samples: Number of samples for QAlign
            temperature: Sampling temperature
            max_length: Maximum generation length
            device: Device for computation
        """
        self.base_model = base_model
        self.mle_model = mle_model
        self.p_vector_mle = p_vector_mle
        self.base_prompt = base_prompt
        self.attribute_prompts = attribute_prompts
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.temperature = temperature
        self.max_length = max_length
        self.device = device or 'cpu'
    
    def generate(self, prompt: str) -> str:
        """Generate using QAlign with MLE scoring."""
        from src.core.drift import get_scores
        
        # Generate multiple samples
        samples = []
        for _ in range(self.num_samples):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.base_model.generate(
                    **inputs,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            samples.append(response)
        
        # Score with MLE-optimized vector
        scores = get_scores(
            [(prompt, samples)],
            self.mle_model,
            self.p_vector_mle,
            self.base_prompt,
            self.attribute_prompts,
            self.device,
            self.tokenizer
        )[0]
        
        best_idx = np.argmax(scores)
        return samples[best_idx]


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


class SimpleGenerator(Generator):
    """Simple generator for testing with transformers."""
    
    def __init__(
        self,
        model,
        tokenizer,
        max_length: int = 512,
        temperature: float = 0.7,
        device=None
    ):
        """
        Initialize simple generator.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            max_length: Maximum generation length
            temperature: Generation temperature
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.temperature = temperature
        self.device = device or 'cpu'
    
    def generate(self, prompt: str) -> str:
        """Generate a simple response."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        return response