import pickle

with open("user1data.pkl", 'rb') as f:
    data = pickle.load(f)

for x, yw, yl in data[10:20]:
    print("Prompt: ")
    print(x)
    print()
    print("Preferred Answer:")
    print(yw)
    print()
    print("Unpreferred:")
    print(yl)

