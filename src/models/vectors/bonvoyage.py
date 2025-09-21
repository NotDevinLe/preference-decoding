import torch
import json
from src.core.drift import compute_drift_rewards
from src.core.attribute_prompts import base_prompt
from src.models.vectors.base import BaseVector
from src.models.qalign.qalign import QAlign

class BonVoyageVector(BaseVector):
    def __init__(self, device, mc_samples, attribute_prompts, model_name, qalign_generator=None):
        self.device = device
        num_attrs = len(attribute_prompts)
        self.p = torch.randn(num_attrs, device=device)
        self.qalign_generator = qalign_generator
        self.mc_samples = mc_samples
        self.model_name = model_name
        self.attribute_prompts = attribute_prompts

    async def compute_chosen_rewards(self, data, tokenizer, gateway_url):
        """Compute reward vectors for chosen responses in training data."""
        print("Computing chosen rewards...")
        chosen_data = [(item['prompt'], item['chosen']) for item in data]
        chosen_rewards = await self.get_reward(chosen_data, tokenizer, gateway_url)
        print(f"Computed rewards for {len(data)} chosen responses")
        return chosen_rewards
    
    async def train(self, data, chosen_rewards, learning_rate=0.01, max_epochs=1000, beta=1.0, tokenizer=None, gateway_url=None, qalign_steps=15):
        """Simple training loop for learning preference vector p."""
        print(f"Training with {len(data)} data points for {max_epochs} epochs...")
        
        for epoch in range(max_epochs):
            total_gradient = torch.zeros_like(self.p)
            
            # Generate expectations for all data items in parallel
            if self.qalign_generator is not None and tokenizer is not None and gateway_url is not None:
                expected_rewards = await self._estimate_expectations_with_qalign(
                    data, qalign_steps, tokenizer, gateway_url
                )
            else:
                # Simple random baseline expectations
                expected_rewards = [torch.randn_like(chosen_rewards[0]) * 0.1 for _ in data]
            
            for data_idx in range(len(data)):
                # Get chosen reward and expected reward
                chosen_reward = chosen_rewards[data_idx]
                expected_reward = expected_rewards[data_idx]
                
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

    
    async def _estimate_expectations_with_qalign(self, data, steps, tokenizer, gateway_url):
        """Estimate expectations for all data items using QAlign generator."""
        if self.qalign_generator is None:
            return [torch.zeros(self.p.shape[0], device=self.device) for _ in data]
        
        conversations = []

        for row in data:
            self.conversations.append([{'role': 'user', 'content': row['prompt']}])
        # Generate samples using QAlign for all conversations
        qalign_output = await self.qalign_generator.run(
            conversations=conversations,
            steps=steps
        )
        
        # Process outputs for each conversation
        expectations = []
        for i, data_item in enumerate(data):
            # Extract generated texts for this conversation
            generated_texts = []
            if i < len(qalign_output.texts):
                output_dict = qalign_output.texts[i]
                for output_item in output_dict['outputs']:
                    generated_texts.append(output_item['text'])
                    if len(generated_texts) >= self.mc_samples:
                        break
            
            if not generated_texts:
                raise Exception("Nothing was generated in the sampling stage.")
            
            # Prepare data for reward computation
            reward_data = [(data_item['prompt'], text) for text in generated_texts]
            
            # Compute rewards for generated samples
            sample_rewards = await self.get_reward(reward_data, tokenizer, gateway_url)
            
            # Compute expected reward (mean across samples)
            expected_reward = torch.mean(sample_rewards, dim=0)
            expectations.append(expected_reward)
        
        return expectations

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
            attribute_prompts=self.attribute_prompts,
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
    
    def get_vector(self):
        return VectorReward(self.p, self.attribute_prompts)
    
    def set_qalign_generator(self, qalign_generator):
        """Set the QAlign generator for expectation estimation."""
        self.qalign_generator = qalign_generator
    
    async def generate_with_qalign(self, conversations, steps=100):
        """Generate samples using QAlign for the given conversations."""
        if self.qalign_generator is None:
            raise ValueError("QAlign generator not set. Use set_qalign_generator() first.")
        
        return await self.qalign_generator.run(
            conversations=conversations,
            steps=steps
        )
    
if __name__ == '__main__':
    import asyncio
    from src.models.remote_vllm import RemoteVLLM
    from src.models.qalign.reward import VectorReward
    from transformers import AutoTokenizer

    async def main():
        with open('data/persona_pref/user11_train.json', 'r') as f:
            data = json.load(f)
        
        # Select specific attribute prompts to use
        from src.core.attribute_prompts import attribute_prompts
        
        # Example: Select a subset of interesting attribute prompts
        selected_prompts = [
            "You are a concise assistant. Keep answers short and to the point.",
            "You are a verbose assistant. Provide detailed, expanded answers.",
            "You are a formal academic assistant. Use professional and scholarly tone.",
            "You are a casual conversational assistant. Write informally and with a friendly tone.",
            "You are a step-by-step assistant. Solve problems with enumerated steps.",
            "You are an answer-first assistant. Start with the final answer, then explain.",
            "You are an empathetic assistant. Express care and support in your answers.",
            "You are a humorous assistant. Add light humor where appropriate.",
        ]
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        model = RemoteVLLM(
            server_url="http://g3103.hyak.local:8080",
            model_path="meta-llama/Llama-3.2-1B-Instruct",
            max_prompt_length=1000,
            max_new_tokens=1000,
        )
        
        # Initialize reward with placeholder vector
        initial_vector = [1.0] + [0.0] * (len(selected_prompts) - 1)  # Start with preference for first attribute
        reward = VectorReward(initial_vector, selected_prompts)
        
        # Create QAlign generator
        qalign = QAlign(model=model, reward=reward)
        
        # Initialize BonVoyage vector
        vector = BonVoyageVector('cuda', 8, selected_prompts, 'meta-llama/Llama-3.2-1B-Instruct', qalign_generator=qalign)
        
        # Compute chosen rewards
        chosen_rewards = await vector.compute_chosen_rewards(data, tokenizer, "http://g3124.hyak.local:8080")
        
        # Train the vector
        await vector.train(
            data=data,
            chosen_rewards=chosen_rewards,
            learning_rate=0.01,
            max_epochs=100,
            tokenizer=tokenizer,
            gateway_url="http://g3124.hyak.local:8080",
            qalign_steps=15
        )
        
        # Save results
        vector.save_results("bonvoyage_results.json", len(data))
        
        print(f"Final preference vector: {vector.p}")
        print(f"Vector norm: {torch.norm(vector.p).item():.4f}")

    asyncio.run(main())