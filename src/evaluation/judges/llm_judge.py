#!/usr/bin/env python3
"""
Simple async LLM Judge for persona evaluation - just does comparisons.
"""

import os
import hashlib
import asyncio
import aiohttp
from pathlib import Path
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class PersonaJudge:
    """Simple async LLM judge for comparing persona adherence."""
    
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://g3101:8000/v1")
        self.model = model or os.getenv("VLLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
        self.cache_dir = Path("cache/persona_judge")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, persona: str, question: str, response_a: str, response_b: str) -> str:
        """Generate cache key from all inputs for better cache hits."""
        combined = f"{persona}|{question}|{response_a}|{response_b}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    async def _call_llm(self, session: aiohttp.ClientSession, prompt: str) -> Optional[str]:
        """Make async LLM API call."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that compares responses. Answer only with 'A' or 'B'."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 10,
        }
        
        # Set up headers - support both VLLM and OpenAI
        headers = {"Content-Type": "application/json"}
        
        # Check if we're using OpenAI API
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and ("openai.com" in self.base_url or not self.base_url.startswith("http://localhost")):
            headers["Authorization"] = f"Bearer {openai_key}"
        else:
            # Default for VLLM (dummy auth)
            headers["Authorization"] = "Bearer dummy"
        
        try:
            async with session.post(url, json=payload, headers=headers, timeout=30) as response:
                response.raise_for_status()
                result = await response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"LLM call failed: {e}")
            return None
    
    async def compare_responses(self, persona: str, question: str, response_a: str, response_b: str) -> Optional[str]:
        """Compare two responses. Returns "A" or "B"."""
        
        # Check cache
        cache_key = self._get_cache_key(persona, question, response_a, response_b)
        cache_file = self.cache_dir / f"{cache_key}.txt"
        
        if cache_file.exists():
            try:
                result = cache_file.read_text().strip()
                if result in ["A", "B"]:
                    return result
            except:
                pass
        
        # Simplified prompt with just one example
        prompt = f"""Compare which response better follows the given persona.

Example:
Persona: You are a formal AI assistant.
Question: What's the weather like?
Response A: Yo, it's pretty nice out there!
Response B: I apologize, but I do not have access to real-time weather information.
Better response: B

Now judge this:
Persona: {persona}
Question: {question}
Response A: {response_a}
Response B: {response_b}
Better response (just write A or B):"""
        
        # Call LLM
        async with aiohttp.ClientSession() as session:
            response = await self._call_llm(session, prompt)
            if response:
                # Parse response - much simpler parsing
                response_clean = response.strip().upper()
                if "A" in response_clean:
                    result = "A"
                elif "B" in response_clean:
                    result = "B"
                else:
                    return None
                
                # Cache result
                try:
                    cache_file.write_text(result)
                except:
                    pass
                
                return result
        
        return None
    
    async def batch_compare(self, comparisons: List[Dict[str, str]], max_concurrent: int = 10) -> List[Optional[str]]:
        """Compare multiple pairs in parallel."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def compare_one(comp):
            async with semaphore:
                return await self.compare_responses(
                    comp['persona'], comp['question'], 
                    comp['response_a'], comp['response_b']
                )
        
        tasks = [compare_one(comp) for comp in comparisons]
        return await asyncio.gather(*tasks)


if __name__ == "__main__":
    # Simple test
    judge = PersonaJudge()
    
    async def test():
        persona = "You are a grumpy old wizard who speaks in riddles"
        question = "How do I learn magic?"
        response_a = "You should read some books about magic and practice simple spells."
        response_b = "Bah! *waves staff* The path lies not in seeking, but in finding what was never lost."
        
        winner = await judge.compare_responses(persona, question, response_a, response_b)
        print(f"Winner: Response {winner}")
        print(f"Expected: Response B (the grumpy wizard response)")
    
    asyncio.run(test())