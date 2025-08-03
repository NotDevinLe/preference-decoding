import json
import math

with open("../results/preference/user1_p.json", "r") as f:
    p_vector_list = json.load(f)

for entry in p_vector_list:
    for lambda_ in [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 100, 1000, 10000]:
        if entry['lambda'] == lambda_ and entry['sample_size'] == 200:
            nonzero = 0
            for i in range(len(entry['p'])):
                if abs(entry['p'][i]) >= 1e-6:
                    nonzero += 1
            print(entry['lambda'], nonzero)
