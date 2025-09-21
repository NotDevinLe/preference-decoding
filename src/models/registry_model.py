import torch
from typing import List, Union


class RegistryModelWrapper:
    """
    Wrapper to adapt registry-based model calls to QAlign's expected interface.
    
    QAlign expects a model with these methods:
    - encode(input_data: List[str]) -> torch.Tensor (tokenized prompt ids)
    - continuation(prompt: torch.Tensor, prefix: List[torch.Tensor]) -> List[torch.Tensor] 
    - decode_tokenize(completions: List[torch.Tensor]) -> List[str]
    - tokenize(text: List[str]) -> List[torch.Tensor]
    - tokenizer attribute with apply_chat_template method
    """
    
    def __init__(self, gateway_url, tokenizer, model_name):
        self.gateway_url = gateway_url
        self.tokenizer = tokenizer
        self.model_name = model_name
    
    def encode(self, input_data: List[str]) -> torch.Tensor:
        """
        Encode a list of prompt strings into token IDs.
        
        Args:
            input_data: List of prompt strings (already formatted with chat template)
            
        Returns:
            torch.Tensor: Tokenized prompt IDs, shape (batch_size, seq_len)
        """
        # Tokenize the input strings
        encoded = self.tokenizer(
            input_data,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        return encoded['input_ids']
    
    def continuation(self, prompt: torch.Tensor, prefix: Union[List[torch.Tensor], None]) -> List[torch.Tensor]:
        """
        Generate continuations from the model given a prompt and optional prefix.
        
        Args:
            prompt: Tokenized prompt, shape (batch_size, prompt_len)
            prefix: List of tokenized prefixes, one per batch item. 
                   Each prefix is shape (prefix_len,). Can be None.
        
        Returns:
            List[torch.Tensor]: List of generated continuation token sequences
        """
        batch_size = prompt.shape[0]
        continuations = []
        
        for i in range(batch_size):
            # Get the prompt for this batch item
            prompt_tokens = prompt[i]
            
            # If prefix is provided, concatenate it with the prompt
            if prefix is not None and prefix[i] is not None:
                input_tokens = torch.cat([prompt_tokens, prefix[i]])
            else:
                input_tokens = prompt_tokens
            
            # Decode back to text for registry call
            input_text = self.tokenizer.decode(input_tokens, skip_special_tokens=True)
            
            # Generate continuation using registry
            # Note: This is a synchronous call - you may need to adapt for async
            generated_text = self._generate_with_registry(input_text)
            
            # Tokenize the generated text
            generated_tokens = self.tokenizer.encode(
                generated_text, 
                return_tensors="pt",
                add_special_tokens=False
            ).squeeze(0)
            
            continuations.append(generated_tokens)
        
        return continuations
    
    def decode_tokenize(self, completions: List[torch.Tensor]) -> List[str]:
        """
        Decode a list of token sequences back to text strings.
        
        Args:
            completions: List of token tensors, each shape (seq_len,)
            
        Returns:
            List[str]: Decoded text strings
        """
        texts = []
        for tokens in completions:
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            texts.append(text)
        return texts
    
    def tokenize(self, text_list: List[str]) -> List[torch.Tensor]:
        """
        Tokenize a list of text strings.
        
        Args:
            text_list: List of text strings
            
        Returns:
            List[torch.Tensor]: List of tokenized sequences
        """
        tokenized = []
        for text in text_list:
            tokens = self.tokenizer.encode(
                text,
                return_tensors="pt",
                add_special_tokens=False
            ).squeeze(0)
            tokenized.append(tokens)
        return tokenized
    
    def _generate_with_gateway(self, input_text: str) -> str:
        """
        Generate text using VLLM-compatible gateway.
        
        Args:
            input_text: Input prompt text
            
        Returns:
            str: Generated completion text
        """
        import asyncio
        import aiohttp
        
        async def _async_generate():
            timeout = aiohttp.ClientTimeout(total=120)  # 2 minute timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Prepare the generation payload
                payload = {
                    "model": self.model_name,
                    "prompt": input_text,
                    "max_tokens": 512,
                    "temperature": 1.0,
                    "stop": None  # Let the model decide when to stop
                }
                
                # Make the request to gateway
                async with session.post(f"{self.gateway_url}/v1/completions", json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()
                
                # Extract the generated text from the response
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0].get("text", "")
                elif "text" in result:
                    return result["text"]
                else:
                    # Fallback: return the whole result as string if format is unexpected
                    return str(result)
        
        # Run the async function and return the result
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we need to use a different approach
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _async_generate())
                    return future.result()
            else:
                return loop.run_until_complete(_async_generate())
        except RuntimeError:
            # If no event loop exists, create a new one
            return asyncio.run(_async_generate())


class BonVoyageRewardWrapper:
    """
    Simple wrapper to adapt BonVoyageVector for QAlign's reward interface.
    """
    
    def __init__(self, bonvoyage_vector, tokenizer, gateway_url):
        self.bonvoyage = bonvoyage_vector
        self.tokenizer = tokenizer
        self.gateway_url = gateway_url
    
    def evaluate(self, conversations):
        """
        QAlign-compatible evaluate method.
        
        Args:
            conversations: List of conversations (as expected by QAlign)
            
        Returns:
            List[float]: Scalar reward scores
        """
        return self.bonvoyage.evaluate_for_qalign(
            conversations, 
            self.tokenizer, 
            self.gateway_url
        )


def create_qalign_with_gateway(gateway_url, tokenizer, model_name, bonvoyage_vector, beta=0.1):
    """
    Create a QAlign instance using your gateway infrastructure.
    
    Args:
        gateway_url: URL of the VLLM-compatible gateway
        tokenizer: HuggingFace tokenizer
        model_name: Model name for the gateway
        bonvoyage_vector: Your BonVoyageVector instance
        beta: Temperature parameter for QAlign
        
    Returns:
        QAlign instance ready to use
    """
    from src.models.qalign.qalign_generator import QAlign
    
    # Create wrappers
    model_wrapper = RegistryModelWrapper(gateway_url, tokenizer, model_name)
    reward_wrapper = BonVoyageRewardWrapper(bonvoyage_vector, tokenizer, gateway_url)
    
    # Create QAlign instance
    qalign = QAlign(
        model=model_wrapper,
        reward=reward_wrapper,
        beta=beta
    )
    
    return qalign