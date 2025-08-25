import json

trimmed_outputs = [8, 12, 8, 4, 8, 4, 8, 1, 5, 8, 8, 2, 8, 11, 3, 8, 11, 2, 8]
base_outputs = [5, 3, 9, 7, 12, 9, 7, 5, 5, 5, 8, 2, 5, 11, 3, 5, 5, 1, 5]
    
with open("../data/bon_all.json", "r") as f:
    bon_data = json.load(f)

outputs = []

for i, entry in enumerate(bon_data):
    outputs.append((entry['outputs'][base_outputs[i]], entry['outputs'][trimmed_outputs[i]]))

for i, output in enumerate(outputs):
    print("Output 1:")
    print(output[0])
    print("Output 2:")
    print(output[1])
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print(f"Index: {i}")
    print()
    print()
    print()
    print()
    print()
    print()
