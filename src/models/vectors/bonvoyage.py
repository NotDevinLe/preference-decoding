import torch
import json
from src.core.drift import compute_drift_rewards
from src.core.attribute_prompts import attribute_prompts, base_prompt
from src.models.vectors.base import BaseVector

class BonVoyageVector(BaseVector):
    def __init__(self, device, mc_samples, model_name, qalign_generator=None):
        self.device = device
        num_attrs = len(attribute_prompts)
        self.p = torch.randn(num_attrs, device=device)
        self.qalign_generator = qalign_generator
        self.mc_samples = mc_samples
        self.model_name = model_name

    async def compute_chosen_rewards(self, data, tokenizer, gateway_url):
        """Compute reward vectors for chosen responses in training data."""
        print("Computing chosen rewards...")
        chosen_data = [(item['prompt'], item['chosen']) for item in data]
        chosen_rewards = await self.get_reward(chosen_data, tokenizer, gateway_url)
        print(f"Computed rewards for {len(data)} chosen responses")
        return chosen_rewards
    
    def train(self, data, chosen_rewards, learning_rate=0.01, max_epochs=1000, beta=1.0):
        """Simple training loop for learning preference vector p."""
        print(f"Training with {len(data)} data points for {max_epochs} epochs...")
        
        for epoch in range(max_epochs):
            total_gradient = torch.zeros_like(self.p)
            
            for data_idx in range(len(data)):
                # Get chosen reward
                chosen_reward = chosen_rewards[data_idx]
                
                # Generate expectation using QAlign if available
                if self.qalign_generator is not None:
                    # Note: QAlign expectation estimation requires async - 
                    # For now using placeholder. You'll need to either:
                    # 1. Make train() async and await this call, or
                    # 2. Precompute expectations before training
                    expected_reward = torch.randn_like(chosen_reward) * 0.1  # Placeholder
                else:
                    # Simple random baseline expectation
                    expected_reward = torch.randn_like(chosen_reward) * 0.1
                
                # Compute gradient
                data_gradient = (chosen_reward - expected_reward) / beta
                total_gradient += data_gradient
            
            # Update parameters
            total_gradient = total_gradient / len(data)
            with torch.no_grad():
                self.p += learning_rate * total_gradient
            
            if epoch % 100 == 0:
                gradient_norm = torch.norm(total_gradient).item()
                print(f"Epoch {epoch}: gradient_norm={gradient_norm:.4f}, p_norm={torch.norm(self.p).item():.4f}")
        
        print("Training completed!")

    
    async def _estimate_expectation_with_qalign(self, data_item, steps, tokenizer, gateway_url, beta):
        """Estimate expectation using QAlign generator."""
        if self.qalign_generator is None:
            return torch.zeros(self.p.shape[0], device=self.device)
        
        # Create conversation format for QAlign
        conversation = [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": data_item['prompt']}
        ]
        
        # Generate samples using QAlign
        qalign_output = self.qalign_generator.run(
            conversations=[conversation],  # QAlign expects list of conversations
            steps=steps
        )
        
        # Extract generated texts from QAlign output
        # QAlign returns Output object with .texts attribute
        generated_texts = []
        for output_dict in qalign_output.texts:
            # Extract the generated completions
            for output_item in output_dict['outputs']:
                generated_texts.append(output_item['text'])
                if len(generated_texts) >= self.mc_samples:
                    break
            if len(generated_texts) >= self.mc_samples:
                break
        
        # Prepare data for reward computation
        reward_data = [(data_item['prompt'], text) for text in generated_texts[:self.mc_samples]]
        
        # Compute rewards for generated samples
        sample_rewards = await self.get_reward(reward_data, tokenizer, gateway_url)
        
        # Compute expected reward (mean across samples)
        expected_reward = torch.mean(sample_rewards, dim=0)
        
        return expected_reward

    async def get_reward(self, data, tokenizer, gateway_url):
        """Compute reward for data using drift rewards."""
        flat_questions = [prompt for prompt, _ in data]
        flat_outputs = [output for _, output in data]
        
        reward_matrix = await compute_drift_rewards(
            gateway_url=gateway_url,
            tokenizer=tokenizer,
            prompts=flat_questions,
            outputs=flat_outputs,
            base_prompt=base_prompt,
            attribute_prompts=attribute_prompts,
            model_name=self.model_name,
            device=self.device
        )
        
        return reward_matrix
    
    def save_results(self, save_path, num_data_points=None):
        """Save the learned preference vector."""
        results = {
            "p_vector": self.p.cpu().numpy().tolist(),
            "p_norm": torch.norm(self.p).item(),
            "num_attributes": self.p.shape[0],
        }
        
        if num_data_points is not None:
            results["num_data_points"] = num_data_points
        
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {save_path}")
    
    def set_qalign_generator(self, qalign_generator):
        """Set the QAlign generator for expectation estimation."""
        self.qalign_generator = qalign_generator
    
    def generate_with_qalign(self, conversations, steps=100):
        """Generate samples using QAlign for the given conversations."""
        if self.qalign_generator is None:
            raise ValueError("QAlign generator not set. Use set_qalign_generator() first.")
        
        return self.qalign_generator.run(
            conversations=conversations,
            steps=steps
        )
    
    def evaluate_for_qalign(self, conversations, tokenizer, gateway_url):
        """
        QAlign-compatible reward evaluation method.
        
        Args:
            conversations: List of conversations with system/user/assistant messages
            tokenizer: Tokenizer for reward computation
            gateway_url: URL of the VLLM-compatible gateway
            
        Returns:
            List[float]: Scalar reward scores (one per conversation)
        """
        import asyncio
        
        # Extract prompt/completion pairs from conversations
        reward_data = []
        for conv in conversations:
            prompt = None
            completion = None
            
            for msg in conv:
                if msg["role"] == "user":
                    prompt = msg["content"]
                elif msg["role"] == "assistant":
                    completion = msg["content"]
            
            if prompt is not None and completion is not None:
                reward_data.append((prompt, completion))
        
        # Handle async reward computation synchronously for QAlign
        if not reward_data:
            return []
        
        try:
            # Try to get the current event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in an async context, we need to use a different approach
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self.get_reward(reward_data, tokenizer, gateway_url))
                        reward_matrix = future.result()
                else:
                    reward_matrix = loop.run_until_complete(
                        self.get_reward(reward_data, tokenizer, gateway_url)
                    )
            except RuntimeError:
                # No event loop exists, create a new one
                reward_matrix = asyncio.run(
                    self.get_reward(reward_data, tokenizer, gateway_url)
                )
            
            # Convert to scalar rewards using learned preference vector
            scalar_rewards = torch.sum(reward_matrix * self.p, dim=1).tolist()
            return scalar_rewards
            
        except Exception as e:
            print(f"Error in reward evaluation: {e}")
            # Return zero rewards as fallback
            return [0.0] * len(reward_data)
