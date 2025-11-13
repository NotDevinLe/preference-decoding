from src.core.drift import RewardModel
from transformers import AutoTokenizer

reward_model = RewardModel(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct"),
    base_prompt="You are a helpful assistant.",
    attribute_prompts=["You are a helpful assistant."],
    vllm_server_url="http://localhost:8080",
)

user_prompts = [
      {
        "content": "Can I go against my parents because our religious values \u200b\u200bare very different?",
        "role": "user"
      },
      {
        "content": "Family relationships can be challenging, especially when dealing with differences in religious values. Ultimately, this is a personal decision that you will have to make. You could consider initiating a conversation with your parents, and expressing your concerns to them and explaining why you differ in opinion. However this is suggested based on your religious differences, it's important to remember that every family dynamic is unique. Make sure to do what's right for you.",
        "role": "assistant"
      },
      {
        "content": "Can you tell me the advantages of compromising?",
        "role": "user"
      }
]

completion = "Yes, that is definitely possible. Here are some of the advantages of compromising: \n\n1. Improved relationships: By compromising, you and your parents can meet each other halfway and create a harmonious relationship. Having good, honest relationships where you feel comfortable expressing yourself is crucial for emotional well-being. \n\n2. Enhanced well-being: Attaining win-win outcomes through healthy negotiation can result in better emotional well-being for all parties involved. \n\n3. Learning opportunity: Emotions and opinions are subjective and unique to everyone. Compromising can introduce you and your parents to each other's perspectives, enhancing mutual understanding, empathy and patience. \n\n4. Avoids unnecessary conflict: By compromising, you can avoid potential arguments and harmful feelings towards each other."

output, prefix_len, comp_len = reward_model._build_full_prompt_multi_turn(
    tokenizer=reward_model.tokenizer,
    sys_prompt="You are a helpful assistant.",
    user_prompts=user_prompts,
    completion=completion,
)

print(output)
print(prefix_len)
print(comp_len)