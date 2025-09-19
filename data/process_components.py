import json
import numpy as np

with open('data/parameter_search_gumbel.json', 'r') as f:
    data = json.load(f)

cleaned = [data[0], data[3], data[-1]]

processed = []

for row in cleaned:
    features = np.where(np.array(row[3]) == 1.0)[0].tolist()
    print(len(features))