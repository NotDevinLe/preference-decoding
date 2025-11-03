from src.core.drift import get_log_probs, compute_rewards
import asyncio
import aiohttp
import json
from transformers import AutoTokenizer
import torch

personas = [
    "You value precision, clarity, and logical rigor; define all terms, verify assumptions, and reason step by step with explicit justifications, avoiding emotional or speculative phrasing.",
    "You are warm, patient, and encouraging; explain ideas with empathy and real-world analogies, prioritizing emotional understanding and reassurance alongside correctness.",
    "You care about efficiency and practicality; translate theory into implementable systems or code, focusing on robust, simple, and clear solutions over abstract elegance.",
    "You think through metaphor and imagery; express insights as narratives or visual analogies that evoke curiosity and imagination rather than strict technical density.",
    "You challenge assumptions and demand evidence; cite data, acknowledge uncertainty, and rigorously outline limitations before drawing conclusions.",
    "You connect ideas across disciplines, weaving reflections from science, art, and philosophy to uncover deeper context, meaning, and implications.",
    "You respond quickly and efficiently with concise, information-dense statements; favor heuristics, lists, and short analogies over long explanations.",
    "You communicate energetically and interactively, mixing curiosity and humor while co-creating ideas and maintaining an upbeat, exploratory tone.",
    "You center fairness, inclusivity, and human impact; discuss trade-offs transparently and emphasize moral reasoning and responsibility in every conclusion.",
    "You are self-aware of your reasoning process; explain how you arrive at answers, note confidence levels, assumptions, and possible errors to ensure transparency."
  ]

base_prompt = "You are a helpful AI Assistant"

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

import numpy as np

def compare_matrices(A, B, rtol=1e-5, atol=1e-8):
    """Works with lists, tuples, numpy arrays, PyTorch tensors, etc."""
    # Convert to numpy arrays - handles most common types
    A = np.asarray(A) * 10
    B = np.asarray(B)
    
    # Now proceed with comparison
    if A.shape != B.shape:
        print(f"Shape mismatch: {A.shape} vs {B.shape}")
        return False
    
    return np.allclose(A, B, rtol=rtol, atol=atol)

# async def main():

#     # log_prob_matrix = [[0] * 10 for _ in range(10)]

#     # for i in range(10):
#     #   for j in range(10):
#     #     data = json.load(open(f"data/preference_high_temp/user{i}_train.json"))[:10]
#     #     print(f"Processing user{i}'s data with attribute {j}")
#     #     async with aiohttp.ClientSession() as session:
#     #       log_probs, token_counts = await get_log_probs(session, "http://localhost:7000", tokenizer, [personas[j]] * len(data), [d['prompt'] for d in data], [d['chosen'] for d in data], "meta-llama/Llama-3.1-8B-Instruct", temperature=0.8)
        
#     #     log_prob_matrix[i][j] = (torch.tensor(log_probs) / torch.tensor(token_counts)).mean().item()

#     with open('tests/rewards.json', 'r') as f:
#       rewards = json.load(f)['rewards']

#     # of shape (users, samples, users)
#     # logprobs is of shape (users, users)

#     reshaped = []

#     for i in range(10):
#       avg = []
#       for j in range(10):
#         curr = 0
#         for k in range(10):
#           curr += rewards[i][k][j]
#         avg.append(curr)
#       reshaped.append(avg)
    
#     print(reshaped)

A = [[-0.30683889985084534, -0.4084227681159973, -0.33385196328163147, -0.4330257475376129, -0.3309221863746643, -0.3447827398777008, -0.43709635734558105, -0.4140792787075043, -0.35049524903297424, -0.38533705472946167], [-0.8235675692558289, -0.5357865691184998, -0.752799391746521, -0.7560092210769653, -0.7813128232955933, -0.7381432056427002, -0.8805407285690308, -0.6795555353164673, -0.7096825838088989, -0.8834789395332336], [-0.3341192305088043, -0.42011547088623047, -0.29589226841926575, -0.47717398405075073, -0.31563782691955566, -0.3123370409011841, -0.3683188259601593, -0.40379253029823303, -0.3206493854522705, -0.4689486026763916], [-0.6982030868530273, -0.5947016477584839, -0.6674982905387878, -0.4746958315372467, -0.677462100982666, -0.6142417192459106, -0.6849144697189331, -0.6418747901916504, -0.6542032957077026, -0.7282206416130066], [-0.3479859232902527, -0.41495901346206665, -0.35229238867759705, -0.41577091813087463, -0.3118429481983185, -0.34904158115386963, -0.4193381667137146, -0.4215710759162903, -0.34989550709724426, -0.448315292596817], [-0.3809800148010254, -0.4306625723838806, -0.35313913226127625, -0.49301186203956604, -0.35045093297958374, -0.3051794171333313, -0.43289828300476074, -0.4449591636657715, -0.3510766625404358, -0.46615070104599], [-0.4711104929447174, -0.5614307522773743, -0.35653555393218994, -0.6776737570762634, -0.40081220865249634, -0.3975033462047577, -0.27389973402023315, -0.5258764028549194, -0.43576163053512573, -0.6258583664894104], [-0.7135373950004578, -0.5402142405509949, -0.6169247031211853, -0.6276695728302002, -0.6416825652122498, -0.5910700559616089, -0.7019403576850891, -0.4157940745353699, -0.5972112417221069, -0.7438355684280396], [-0.5007334351539612, -0.504774808883667, -0.4636101722717285, -0.5371552109718323, -0.4528290331363678, -0.4379192292690277, -0.5342816710472107, -0.5257585644721985, -0.3680607080459595, -0.5848481059074402], [-0.6366413235664368, -0.7451286911964417, -0.6846242547035217, -0.7733221054077148, -0.6365265250205994, -0.7111304998397827, -0.7305945754051208, -0.7570265531539917, -0.6890844106674194, -0.4180386960506439]]
B = [[-3.072517678141594, -4.089970424771309, -3.3495184779167175, -4.333897441625595, -3.3188314735889435, -3.4459514766931534, -4.361084133386612, -4.141328036785126, -3.495847165584564, -3.8439464569091797], [-8.220565736293793, -5.35002176463604, -7.537853568792343, -7.5493505001068115, -7.8170692920684814, -7.388005465269089, -8.810965687036514, -6.796077311038971, -7.104526698589325, -8.833641111850739], [-3.3406380489468575, -4.205619663000107, -2.9685909375548363, -4.769599884748459, -3.159160554409027, -3.127167284488678, -3.684123769402504, -4.035756394267082, -3.209605909883976, -4.684301972389221], [-6.9829510864801705, -5.943470709957182, -6.675084830727428, -4.743880167254247, -6.774509785929695, -6.144712422043085, -6.851260238559917, -6.42196474596858, -6.542194508481771, -7.284853532910347], [-3.479348100721836, -4.143096223473549, -3.521594002842903, -4.15748555958271, -3.123104825615883, -3.487567164003849, -4.196528673171997, -4.2268573343753815, -3.4978387877345085, -4.484891027212143], [-3.8095356971025467, -4.313680723309517, -3.53411765396595, -4.933094456791878, -3.5148275271058083, -3.0558380857110023, -4.329613856971264, -4.4545943439006805, -3.511186880990863, -4.658473119139671], [-4.725070923566818, -5.620611384510994, -3.570383194833994, -6.778596580028534, -4.013756409287453, -3.9917994663119316, -2.742372613400221, -5.250852629542351, -4.363219760358334, -6.250517085194588], [-7.138437002897263, -5.406330227851868, -6.169519379734993, -6.2714715003967285, -6.4122279435396194, -5.903587371110916, -7.02702459692955, -4.161497667431831, -5.970860093832016, -7.439904868602753], [-5.011121481657028, -5.051214426755905, -4.637267768383026, -5.376228928565979, -4.531381458044052, -4.376789778470993, -5.344107061624527, -5.2640480399131775, -3.684681072831154, -5.853046655654907], [-6.367742106318474, -7.448817312717438, -6.84297388792038, -7.734967052936554, -6.366455972194672, -7.109772741794586, -7.3060703575611115, -7.57056599855423, -6.8966589123010635, -4.178641088306904]]

print(compare_matrices(A, B, rtol=1e-3, atol=1e-3))

# if __name__ == "__main__":
#     asyncio.run(main())