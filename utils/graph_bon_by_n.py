import json
import matplotlib.pyplot as plt

# Load results
with open('../results/rm_bon_by_n.jsonl', 'r') as f:
    rm_results = [json.loads(line) for line in f if line.strip()]
with open('../results/approx_bon_by_n.jsonl', 'r') as f:
    approx_results = [json.loads(line) for line in f if line.strip()]

# Group by k
rm_by_k = {entry['k']: entry for entry in rm_results}
approx_by_k = {entry['k']: entry for entry in approx_results}

ks = sorted(set(rm_by_k.keys()) & set(approx_by_k.keys()))

# Prepare data for plotting
rm_selected_minus_max = [rm_by_k[k].get('avg_selected_minus_max', 0) for k in ks]
approx_selected_minus_max = [approx_by_k[k].get('avg_selected_minus_max', 0) for k in ks]

plt.figure(figsize=(8, 6))
plt.plot(ks, rm_selected_minus_max, 'o-', label='RM')
plt.plot(ks, approx_selected_minus_max, 'o-', label='Approx')
plt.title('Avg (Selected - Max Gold)')
plt.xlabel('k')
plt.ylabel('Selected - Max Gold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('bon_by_n_comparison.png')
# plt.show()
