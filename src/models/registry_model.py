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
    
    def __init__(self, registry_client, tokenizer, model_name):
        self.registry_client = registry_client
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
    
    def _generate_with_registry(self, input_text: str) -> str:
        """
        Generate text using your registry client.
        
        Args:
            input_text: Input prompt text
            
        Returns:
            str: Generated completion text
            
        Note: You'll need to implement this based on your registry API
        """
        # TODO: Replace this with your actual registry generation call
        # Example:
        # response = self.registry_client.generate(
        #     model_name=self.model_name,
        #     prompt=input_text,
        #     max_tokens=512,
        #     temperature=1.0
        # )
        # return response.text
        
        # Placeholder - replace with your registry call
        return "Generated text placeholder"


class BonVoyageRewardWrapper:
    """
    Simple wrapper to adapt BonVoyageVector for QAlign's reward interface.
    """
    
    def __init__(self, bonvoyage_vector, tokenizer, registry_client):
        self.bonvoyage = bonvoyage_vector
        self.tokenizer = tokenizer
        self.registry_client = registry_client
    
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
            self.registry_client
        )


def create_qalign_with_registry(registry_client, tokenizer, model_name, bonvoyage_vector, beta=0.1):
    """
    Create a QAlign instance using your registry infrastructure.
    
    Args:
        registry_client: Your registry client for model calls
        tokenizer: HuggingFace tokenizer
        model_name: Model name for registry
        bonvoyage_vector: Your BonVoyageVector instance
        beta: Temperature parameter for QAlign
        
    Returns:
        QAlign instance ready to use
    """
    from src.models.qalign.qalign_generator import QAlign
    
    # Create wrappers
    model_wrapper = RegistryModelWrapper(registry_client, tokenizer, model_name)
    reward_wrapper = BonVoyageRewardWrapper(bonvoyage_vector, tokenizer, registry_client)
    
    # Create QAlign instance
    qalign = QAlign(
        model=model_wrapper,
        reward=reward_wrapper,
        beta=beta
    )
    
    return qalign