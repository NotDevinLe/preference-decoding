#!/usr/bin/env python3
"""
LLM Judge for persona evaluation using VLLM endpoint.
"""

import os
import sys
import json
import requests
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed

from persona_rubric import (
    PersonaScore,
    create_evaluation_prompt,
    create_comparison_prompt,
    parse_evaluation_response,
    parse_comparison_response,
    extract_persona_from_prompt
)


class PersonaJudge:
    """LLM-based judge for evaluating persona adherence."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        cache_dir: str = "cache/persona_judge",
        temperature: float = 0.1,
        max_tokens: int = 512,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize PersonaJudge.
        
        Args:
            base_url: VLLM endpoint URL (defaults to env var VLLM_BASE_URL)
            model: Model name (defaults to env var VLLM_MODEL)
            cache_dir: Directory for caching evaluations
            temperature: Temperature for LLM sampling (lower = more consistent)
            max_tokens: Maximum tokens for response
            max_retries: Number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.model = model or os.getenv("VLLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Statistics
        self.stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "failures": 0
        }
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key for a prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """Load cached response if available."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    self.stats["cache_hits"] += 1
                    return data["response"]
            except:
                pass
        return None
    
    def _save_to_cache(self, cache_key: str, response: str):
        """Save response to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    "response": response,
                    "timestamp": time.time()
                }, f)
        except:
            pass
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Make a single LLM API call with retries."""
        url = f"{self.base_url}/chat/completions"
        
        messages = [
            {"role": "system", "content": "You are an expert at evaluating how well responses match specific personas. Be precise and critical in your evaluations."},
            {"role": "user", "content": prompt}
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy"
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                response.raise_for_status()
                result = response.json()
                self.stats["api_calls"] += 1
                return result["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"Error calling LLM after {self.max_retries} attempts: {e}")
                    self.stats["failures"] += 1
                    return None
    
    def score_response(
        self,
        persona: str,
        question: str,
        response: str,
        use_cache: bool = True
    ) -> Optional[PersonaScore]:
        """
        Score a single response for persona adherence.
        
        Args:
            persona: Persona description
            question: The question/prompt
            response: The response to evaluate
            use_cache: Whether to use caching
            
        Returns:
            PersonaScore object or None if evaluation fails
        """
        self.stats["total_calls"] += 1
        
        # Create evaluation prompt
        eval_prompt = create_evaluation_prompt(persona, question, response)
        
        # Check cache
        cache_key = self._get_cache_key(eval_prompt)
        llm_response = None
        
        if use_cache:
            llm_response = self._load_from_cache(cache_key)
        
        # Call LLM if not cached
        if llm_response is None:
            llm_response = self._call_llm(eval_prompt)
            if llm_response and use_cache:
                self._save_to_cache(cache_key, llm_response)
        
        # Parse response
        if llm_response:
            try:
                return parse_evaluation_response(llm_response)
            except Exception as e:
                print(f"Error parsing evaluation response: {e}")
                return None
        
        return None
    
    def compare_responses(
        self,
        persona: str,
        question: str,
        response_a: str,
        response_b: str,
        use_cache: bool = True
    ) -> Optional[Tuple[PersonaScore, PersonaScore, str, str]]:
        """
        Compare two responses for persona adherence.
        
        Args:
            persona: Persona description
            question: The question/prompt
            response_a: First response
            response_b: Second response
            use_cache: Whether to use caching
            
        Returns:
            Tuple of (score_a, score_b, winner, reason) or None if evaluation fails
        """
        self.stats["total_calls"] += 1
        
        # Create comparison prompt
        comp_prompt = create_comparison_prompt(persona, question, response_a, response_b)
        
        # Check cache
        cache_key = self._get_cache_key(comp_prompt)
        llm_response = None
        
        if use_cache:
            llm_response = self._load_from_cache(cache_key)
        
        # Call LLM if not cached
        if llm_response is None:
            llm_response = self._call_llm(comp_prompt)
            if llm_response and use_cache:
                self._save_to_cache(cache_key, llm_response)
        
        # Parse response
        if llm_response:
            try:
                return parse_comparison_response(llm_response)
            except Exception as e:
                print(f"Error parsing comparison response: {e}")
                return None
        
        return None
    
    def batch_evaluate(
        self,
        evaluations: List[Dict[str, str]],
        max_workers: int = 4,
        progress_callback: Optional[callable] = None
    ) -> List[Optional[PersonaScore]]:
        """
        Evaluate multiple responses in parallel.
        
        Args:
            evaluations: List of dicts with keys 'persona', 'question', 'response'
            max_workers: Number of parallel workers
            progress_callback: Optional callback function(completed, total)
            
        Returns:
            List of PersonaScore objects (None for failed evaluations)
        """
        results = [None] * len(evaluations)
        
        def evaluate_single(idx, eval_data):
            score = self.score_response(
                eval_data['persona'],
                eval_data['question'],
                eval_data['response']
            )
            return idx, score
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(evaluate_single, i, eval_data): i
                for i, eval_data in enumerate(evaluations)
            }
            
            completed = 0
            for future in as_completed(futures):
                idx, score = future.result()
                results[idx] = score
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, len(evaluations))
        
        return results
    
    def batch_compare(
        self,
        comparisons: List[Dict[str, str]],
        max_workers: int = 4,
        progress_callback: Optional[callable] = None
    ) -> List[Optional[Tuple[PersonaScore, PersonaScore, str, str]]]:
        """
        Compare multiple response pairs in parallel.
        
        Args:
            comparisons: List of dicts with keys 'persona', 'question', 'response_a', 'response_b'
            max_workers: Number of parallel workers
            progress_callback: Optional callback function(completed, total)
            
        Returns:
            List of comparison results (None for failed comparisons)
        """
        results = [None] * len(comparisons)
        
        def compare_single(idx, comp_data):
            result = self.compare_responses(
                comp_data['persona'],
                comp_data['question'],
                comp_data['response_a'],
                comp_data['response_b']
            )
            return idx, result
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(compare_single, i, comp_data): i
                for i, comp_data in enumerate(comparisons)
            }
            
            completed = 0
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, len(comparisons))
        
        return results
    
    def score(self, prompt: str, response: str, persona: str = "A helpful assistant") -> float:
        """
        Score interface for BaseEvaluator - returns average of 5 rubric dimensions.
        
        Args:
            prompt: The prompt/question
            response: The response to evaluate  
            persona: Persona description
            
        Returns:
            Average score from 1-5 rubric dimensions
        """
        persona_score = self.score_response(persona, prompt, response)
        if persona_score:
            return persona_score.get_overall()
        else:
            return 3.0  # Default middle score if evaluation fails
    
    def get_statistics(self) -> Dict[str, int]:
        """Get usage statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset usage statistics."""
        self.stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "failures": 0
        }


class AsyncPersonaJudge(PersonaJudge):
    """Async version of PersonaJudge for higher throughput."""
    
    async def _call_llm_async(self, session: aiohttp.ClientSession, prompt: str) -> Optional[str]:
        """Make an async LLM API call."""
        url = f"{self.base_url}/chat/completions"
        
        messages = [
            {"role": "system", "content": "You are an expert at evaluating how well responses match specific personas. Be precise and critical in your evaluations."},
            {"role": "user", "content": prompt}
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy"
        }
        
        for attempt in range(self.max_retries):
            try:
                async with session.post(url, json=payload, headers=headers, timeout=30) as response:
                    response.raise_for_status()
                    result = await response.json()
                    self.stats["api_calls"] += 1
                    return result["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"Error calling LLM after {self.max_retries} attempts: {e}")
                    self.stats["failures"] += 1
                    return None
    
    async def score_response_async(
        self,
        session: aiohttp.ClientSession,
        persona: str,
        question: str,
        response: str,
        use_cache: bool = True
    ) -> Optional[PersonaScore]:
        """Async version of score_response."""
        self.stats["total_calls"] += 1
        
        eval_prompt = create_evaluation_prompt(persona, question, response)
        cache_key = self._get_cache_key(eval_prompt)
        llm_response = None
        
        if use_cache:
            llm_response = self._load_from_cache(cache_key)
        
        if llm_response is None:
            llm_response = await self._call_llm_async(session, eval_prompt)
            if llm_response and use_cache:
                self._save_to_cache(cache_key, llm_response)
        
        if llm_response:
            try:
                return parse_evaluation_response(llm_response)
            except Exception as e:
                print(f"Error parsing evaluation response: {e}")
                return None
        
        return None
    
    async def batch_evaluate_async(
        self,
        evaluations: List[Dict[str, str]],
        max_concurrent: int = 10,
        progress_callback: Optional[callable] = None
    ) -> List[Optional[PersonaScore]]:
        """
        Evaluate multiple responses asynchronously.
        
        Args:
            evaluations: List of dicts with keys 'persona', 'question', 'response'
            max_concurrent: Maximum concurrent requests
            progress_callback: Optional async callback function(completed, total)
            
        Returns:
            List of PersonaScore objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = [None] * len(evaluations)
        completed = 0
        
        async def evaluate_with_semaphore(session, idx, eval_data):
            nonlocal completed
            async with semaphore:
                score = await self.score_response_async(
                    session,
                    eval_data['persona'],
                    eval_data['question'],
                    eval_data['response']
                )
                results[idx] = score
                completed += 1
                if progress_callback:
                    await progress_callback(completed, len(evaluations))
                return idx, score
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                evaluate_with_semaphore(session, i, eval_data)
                for i, eval_data in enumerate(evaluations)
            ]
            await asyncio.gather(*tasks)
        
        return results


if __name__ == "__main__":
    # Test the PersonaJudge
    print("=== TESTING PERSONA JUDGE ===")
    
    judge = PersonaJudge()
    
    # Test single evaluation
    test_persona = "A grumpy old wizard who speaks in riddles"
    test_question = "How do I learn magic?"
    test_response = "Bah! Magic, you say? *waves staff irritably* The path to power lies not in seeking, but in finding what was never lost."
    
    print(f"\nPersona: {test_persona}")
    print(f"Question: {test_question}")
    print(f"Response: {test_response[:100]}...")
    
    score = judge.score_response(test_persona, test_question, test_response)
    if score:
        print(f"\nScores:")
        print(f"  Speaking Style: {score.speaking_style}")
        print(f"  Personality: {score.personality}")
        print(f"  Knowledge: {score.knowledge}")
        print(f"  Behavioral: {score.behavioral}")
        print(f"  Emotional: {score.emotional}")
        print(f"  Overall: {score.get_overall():.2f}")
    else:
        print("Evaluation failed!")
    
    # Print statistics
    print(f"\nStatistics: {judge.get_statistics()}")