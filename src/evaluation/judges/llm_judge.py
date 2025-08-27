#!/usr/bin/env python3
"""
Simple async LLM Judge for persona evaluation - just does comparisons.
"""

import os
import hashlib
import json
import asyncio
import aiohttp
from pathlib import Path
from typing import Optional, List, Dict


class PersonaJudge:
    """Simple async LLM judge for comparing persona adherence."""
    
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://g3101:8000/v1")
        self.model = model or os.getenv("VLLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
        self.cache_dir = Path("cache/persona_judge")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()
    
    async def _call_llm(self, session: aiohttp.ClientSession, prompt: str) -> Optional[str]:
        """Make async LLM API call."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a pattern following assistant that can only answer with Output (a) or Output (b). Your goal is to select the preferred (least wrong) output for a given instruction."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 100,
        }
        
        try:
            async with session.post(url, json=payload, timeout=30) as response:
                response.raise_for_status()
                result = await response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"LLM call failed: {e}")
            return None
    
    async def compare_responses(self, persona: str, question: str, response_a: str, response_b: str) -> Optional[str]:
        """Compare two responses. Returns "A" or "B"."""
        prompt = f"""

### Instruction:
Respond to the following input using this persona: You are a humorous AI assistant.

### Input:
How do I cook pasta?

### Output (a):
Boil water in a large pot, add salt, then add pasta and cook according to package directions. Drain and serve with your favorite sauce.

### Output (b):
Step 1: Convince the pasta it wants to be cooked. Step 2: Throw it in boiling water and hope it doesn't hold a grudge. Step 3: Fish it out before it gets too soggy and disappointing, like my last relationship!

### Which response better follows the given persona, Output (a) or Output (b)?:

Output (b)

Good job! Indeed, Output (b) better follows the given persona than Output (a).

### Instruction:
Respond to the following input using this persona: You are a formal AI assistant.

### Input:
What's the weather like?

### Output (a):
Yo, it's pretty nice out there! Sunny and warm, perfect for chilling outside.

### Output (b):
I apologize, but I do not have access to real-time weather information. I recommend consulting a reliable weather service or application for current meteorological conditions in your area.

### Which response better follows the given persona, Output (a) or Output (b)?:

Output (b)

Great! Indeed, Output (b) better follows the given persona than Output (a).

Let's do one more, remember to answer with one-word either "Output (a)" or "Output (b)".

### Instruction:
Respond to the following input using this persona: {persona}

### Input:
{question}

### Output (a):
{response_a}

### Output (b):
{response_b}

### Which response better follows the given persona, Output (a) or Output (b)?:

"""
        
        # Check cache
        cache_key = self._get_cache_key(prompt)
        cache_file = self.cache_dir / f"{cache_key}.txt"
        
        if cache_file.exists():
            try:
                result = cache_file.read_text().strip()
                if result in ["A", "B"]:
                    return result
            except:
                pass
        
        # Call LLM
        async with aiohttp.ClientSession() as session:
            response = await self._call_llm(session, prompt)
            if response:
                # Parse "Output (a)" or "Output (b)" format
                response_lower = response.strip().lower()
                if "output (a)" in response_lower or "output a" in response_lower:
                    result = "A"
                elif "output (b)" in response_lower or "output b" in response_lower:
                    result = "B"
                else:
                    # Fallback: check if starts with A or B
                    response_upper = response.strip().upper()
                    if response_upper.startswith('A'):
                        result = "A"
                    elif response_upper.startswith('B'):
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
    # Test
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