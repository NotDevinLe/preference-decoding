import torch
import json
from src.core.drift import compute_drift_rewards
from src.core.attribute_prompts import base_prompt
from src.models.vectors.base import BaseVector
from src.models.qalign.qalign import QAlign
import time

class BonVoyageVector(BaseVector):
    def __init__(self, device, mc_samples, attribute_prompts, model_name, qalign_generator=None):
        self.device = device
        num_attrs = len(attribute_prompts)
        self.p = torch.randn(num_attrs, device=device)
        self.qalign_generator = qalign_generator
        self.mc_samples = mc_samples
        self.model_name = model_name
        self.attribute_prompts = attribute_prompts
        self.cache_p_vector = None
        self.expected_rewards_cache = None
        self.cached_samples = None  # Store generated samples for importance sampling
        self.warm_start = None
        self.chosen_rewards = None

    async def compute_chosen_rewards(self, data, tokenizer, gateway_url):
        """
        
        Compute reward vectors for chosen responses in training data.
        
        Input: [{'prompt', 'text'}], tokenizer, gateway url

        output: A matrix of the shape len(data) x len(attributes)
        
        """
        print("Computing chosen rewards...")
        chosen_data = [(item['prompt'], item['chosen']) for item in data]
        chosen_rewards = await self.get_reward(chosen_data, tokenizer, gateway_url)
        print(f"Computed rewards for {len(data)} chosen responses")
        self.chosen_rewards = chosen_rewards

        self.warm_start = [{'completion': item['chosen'], 'reward': chosen_rewards[i]} for i, item in enumerate(data)]
        return chosen_rewards
    
    async def train(self, data, chosen_rewards, learning_rate=0.01, max_epochs=1000, beta=1.0, tokenizer=None, gateway_url=None, qalign_steps=15, val_data=None, val_chosen_rewards=None, use_wandb=False):
        """Simple training loop for learning preference vector p."""
        import wandb
        
        if use_wandb:
            wandb.init(project="bonvoyage-training", config={
                "learning_rate": learning_rate,
                "max_epochs": max_epochs,
                "beta": beta,
                "qalign_steps": qalign_steps,
                "train_size": len(data),
                "val_size": len(val_data) if val_data else 0,
                "num_attributes": len(self.attribute_prompts),
            })
        
        print(f"Training with {len(data)} data points for {max_epochs} epochs...")
        
        for epoch in range(max_epochs):
            epoch_start = time.time()
            
            total_gradient = torch.zeros_like(self.p)

            if self.should_resample(0.01):
                print(f"Resampling needed: KL divergence exceeded threshold")
                
                self._update_qalign_reward(tokenizer, gateway_url)
                
                self.cache_p_vector = self.p.clone()

                # Generate new samples and expectations
                if self.qalign_generator is not None and tokenizer is not None and gateway_url is not None:
                    qalign_start = time.time()
                    self.expected_rewards_cache, self.cached_samples = await self._estimate_expectations_with_qalign_and_cache(
                        data, qalign_steps, tokenizer, gateway_url
                    )
                    qalign_time = time.time() - qalign_start
                    print(f"QAlign expectation estimation took: {qalign_time:.2f}s")
                else:
                    raise Exception("QAlign generator not set. Use set_qalign_generator() first.")
            else:
                # Use importance sampling to reweight cached samples
                print(f"Using importance sampling with cached samples")
                qalign_start = time.time()
                self.expected_rewards_cache = await self._reweight_cached_expectations(
                    data, tokenizer, gateway_url
                )
                qalign_time = time.time() - qalign_start
                print(f"Importance sampling reweighting took: {qalign_time:.2f}s")
                
            
            # Compute training loss using negative log-likelihood
            train_loss = 0.0
            for data_idx in range(len(data)):
                chosen_reward = chosen_rewards[data_idx]
                expected_reward = self.expected_rewards_cache[data_idx]
                
                log_chosen_prob = torch.dot(self.p, chosen_reward)
                log_partition = torch.dot(self.p, expected_reward)
                log_likelihood = log_chosen_prob - log_partition
                loss = -log_likelihood
                train_loss += loss.item()
                
                data_gradient = (chosen_reward - expected_reward)
                total_gradient += data_gradient
            
            train_loss /= len(data)
            
            total_gradient = total_gradient / len(data)
            with torch.no_grad():
                self.p += learning_rate * total_gradient
            
            val_loss = None
            if val_data is not None and val_chosen_rewards is not None:
                val_start = time.time()
                val_loss = await self._compute_validation_loss(
                    val_data, val_chosen_rewards, tokenizer, gateway_url, qalign_steps, beta
                )
                val_time = time.time() - val_start
                print(f"Validation computation took: {val_time:.2f}s")
            else:
                val_time = 0
            
            if use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "gradient_norm": torch.norm(total_gradient).item(),
                    "p_norm": torch.norm(self.p).item(),
                }
                if val_loss is not None:
                    log_dict["val_loss"] = val_loss
                wandb.log(log_dict)
            
            epoch_total_time = time.time() - epoch_start
            
            if epoch % 100 == 0:
                gradient_norm = torch.norm(total_gradient).item()
                val_str = f", val_loss={val_loss:.4f}" if val_loss is not None else ""
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}{val_str}, gradient_norm={gradient_norm:.4f}, p_norm={torch.norm(self.p).item():.4f}")
                print(f"  Timing - QAlign: {qalign_time:.2f}s, Validation: {val_time:.2f}s, Total: {epoch_total_time:.2f}s")
        
        if use_wandb:
            wandb.finish()
        
        print("Training completed!")

    async def _compute_validation_loss(self, val_data, val_chosen_rewards, tokenizer, gateway_url, qalign_steps, beta):
        """Compute validation loss using negative log-likelihood (no sample generation needed)."""
        
        val_loss = 0.0
        for i, val_item in enumerate(val_data):
            chosen_reward = val_chosen_rewards[i]
            
            expected_reward = torch.mean(val_chosen_rewards, dim=0)
            
            log_chosen_prob = torch.dot(self.p, chosen_reward)
            log_partition = torch.dot(self.p, expected_reward)
            log_likelihood = log_chosen_prob - log_partition
            loss = -log_likelihood
            val_loss += loss.item()
        
        return val_loss / len(val_data)
    
    async def _estimate_expectations_with_qalign(self, data, steps, tokenizer, gateway_url):
        """Estimate expectations for all data items using QAlign generator."""
        if self.qalign_generator is None:
            raise Exception("There is no QAlign generator")
        
        conversations = []

        for row in data:
            for i in range(self.mc_samples):
                conversations.append([{'role': 'user', 'content': row['prompt']}])
        import asyncio
        import concurrent.futures
        import time
        
        def run_qalign():
            return self.qalign_generator.run(
                conversations=conversations,
                steps=steps
            )
        
        print(f"Starting QAlign generation for {len(conversations)} conversations, {steps} steps each")
        qalign_gen_start = time.time()
        
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            qalign_output = await loop.run_in_executor(executor, run_qalign)
        
        qalign_gen_time = time.time() - qalign_gen_start
        print(f"QAlign generation completed in {qalign_gen_time:.2f}s")
        
        from collections import Counter
        expectations = []
        for i in range(0,len(data), self.mc_samples):
            response = qalign_output.texts[i:i+self.mc_samples]
            generated_texts = []
            counts = [Counter(text['outputs']) for text in response]

            for count in counts:
                generated_texts.append(count.most_common(1)[0][0])

            if not generated_texts:
                raise Exception("Nothing was generated in the sampling stage.")
            
            reward_data = [(data[i]['prompt'], text) for text in generated_texts]
            
            reward_start = time.time()
            sample_rewards = await self.get_reward(reward_data, tokenizer, gateway_url)
            reward_time = time.time() - reward_start
            print(f"  Reward computation for sample {i//self.mc_samples} took {reward_time:.2f}s ({len(reward_data)} requests)")
            
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
        from src.models.qalign.reward import VectorReward
        
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
        return VectorReward(
            vector=self.p.cpu().tolist(), 
            attribute_prompts=self.attribute_prompts,
        )
    
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
    
    def should_resample(self, threshold):
        if self.cache_p_vector is None:
            return True
        
        kl_div = self.compute_kl_div(self.cache_p_vector, self.p)
        return kl_div > threshold

    def compute_kl_div(self, old_p, new_p):
        """Compute approximate KL divergence between old and new preference vectors."""
        diff = (new_p - old_p)
        kl_approx = 0.5 * torch.sum(diff ** 2).item()
        return kl_approx
    
    def _update_qalign_reward(self, tokenizer, gateway_url):
        """Update QAlign's reward function with current preference vector."""
        from src.models.qalign.reward import VectorReward
        from src.core.attribute_prompts import base_prompt
        
        updated_reward = VectorReward(
            vector=self.p.cpu().tolist(),
            attribute_prompts=self.attribute_prompts,
            gateway_url=gateway_url,
            tokenizer=tokenizer,
            base_prompt=base_prompt,
            model_name=self.model_name,
            device=str(self.device)
        )
        self.qalign_generator.rm = updated_reward
        print(f"Updated QAlign reward function with p_norm={torch.norm(self.p).item():.4f}")
    
    async def _estimate_expectations_with_qalign_and_cache(self, data, steps, tokenizer, gateway_url):
        """Generate samples and cache them for importance sampling."""
        conversations = []
        for row in data:
            conversations.append([{'role': 'user', 'content': row['prompt']}])
            
        # Generate samples using QAlign
        import asyncio
        import concurrent.futures
        import time
        
        def run_qalign():
            return self.qalign_generator.run(
                conversations=conversations,
                steps=steps
            )
        
        print(f"Starting QAlign generation for {len(conversations)} conversations, {steps} steps each")
        qalign_gen_start = time.time()
        
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            qalign_output = await loop.run_in_executor(executor, run_qalign)
        
        qalign_gen_time = time.time() - qalign_gen_start
        print(f"QAlign generation completed in {qalign_gen_time:.2f}s")
        
        # Process outputs and cache samples
        expectations = []
        cached_samples = []  # Store (prompt, generated_texts, reward_vectors) for each data item
        
        for i, data_item in enumerate(data):
            generated_texts = []
            if i < len(qalign_output.texts):
                output_dict = qalign_output.texts[i]
                outputs_list = output_dict['outputs']
                
                for sample_idx in range(self.mc_samples):
                    if len(outputs_list) > 0:
                        output_idx = sample_idx % len(outputs_list)
                        generated_texts.append(outputs_list[output_idx])
            
            if not generated_texts:
                raise Exception("Nothing was generated in the sampling stage.")
            
            # Compute rewards for generated samples
            reward_data = [(data_item['prompt'], text) for text in generated_texts]
            reward_start = time.time()
            sample_rewards = await self.get_reward(reward_data, tokenizer, gateway_url)
            reward_time = time.time() - reward_start
            print(f"  Reward computation for sample {i} took {reward_time:.2f}s ({len(reward_data)} requests)")
            
            # Cache samples and their rewards
            cached_samples.append({
                'prompt': data_item['prompt'],
                'generated_texts': generated_texts,
                'sample_rewards': sample_rewards
            })
            
            # Compute expected reward (mean across samples)
            expected_reward = torch.mean(sample_rewards, dim=0)
            expectations.append(expected_reward)
        
        return expectations, cached_samples
    
    async def _reweight_cached_expectations(self, data, tokenizer, gateway_url):
        """Use importance sampling to reweight cached samples with new preference vector."""
        if self.cached_samples is None:
            raise Exception("No cached samples available for importance sampling")
        
        expectations = []
        
        for i, data_item in enumerate(data):
            cached_sample = self.cached_samples[i]
            sample_rewards = cached_sample['sample_rewards']
            
            # Compute importance weights
            # w_i = π_new(a_i|s) / π_old(a_i|s) = exp((p_new - p_old)^T φ(s,a_i))
            diff_p = self.p - self.cache_p_vector
            importance_weights = torch.exp(torch.matmul(sample_rewards, diff_p))
            
            # Normalize weights
            importance_weights = importance_weights / torch.sum(importance_weights)
            
            # Compute importance-weighted expectation
            weighted_rewards = sample_rewards * importance_weights.unsqueeze(1)
            expected_reward = torch.sum(weighted_rewards, dim=0)
            expectations.append(expected_reward)
        
        return expectations
    
if __name__ == '__main__':
    import asyncio
    from src.models.remote_vllm import RemoteVLLM
    from src.models.qalign.reward import VectorReward
    from transformers import AutoTokenizer

    async def main():
        # Load training and test data
        with open('data/persona_pref/user11_train.json', 'r') as f:
            data = json.load(f)
        
        with open('data/persona_pref/user11_test.json', 'r') as f:
            val_data = json.load(f)
        
        gateway_url = "http://localhost:8080"
        
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
            server_url=gateway_url,
            model_path="meta-llama/Llama-3.2-1B-Instruct",
            max_prompt_length=1000,
            max_new_tokens=1000,
        )
        
        # Initialize reward with placeholder vector
        initial_vector = [1.0] + [0.0] * (len(selected_prompts) - 1)  # Start with preference for first attribute
        reward = VectorReward(
            vector=initial_vector, 
            attribute_prompts=selected_prompts,
            gateway_url=gateway_url,
            tokenizer=tokenizer,
            base_prompt=base_prompt,
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            device="cuda"
        )
        
        # Create QAlign generator
        qalign = QAlign(model=model, reward=reward)
        
        # Initialize BonVoyage vector
        vector = BonVoyageVector('cuda', 4, selected_prompts, 'meta-llama/Llama-3.2-1B-Instruct', qalign_generator=qalign)
        
        # Compute chosen rewards for training and validation
        chosen_rewards = await vector.compute_chosen_rewards(data, tokenizer, gateway_url)
        val_chosen_rewards = await vector.compute_chosen_rewards(val_data, tokenizer, gateway_url)
        
        # Train the vector with wandb tracking
        await vector.train(
            data=data,
            chosen_rewards=chosen_rewards,
            learning_rate=0.01,
            max_epochs=30000,
            tokenizer=tokenizer,
            gateway_url=gateway_url,
            qalign_steps=8,
            val_data=val_data,
            val_chosen_rewards=val_chosen_rewards,
            use_wandb=True
        )
        
        # Save results
        vector.save_results("bonvoyage_results.json", len(data))
        
        print(f"Final preference vector: {vector.p}")
        print(f"Vector norm: {torch.norm(vector.p).item():.4f}")

    asyncio.run(main())