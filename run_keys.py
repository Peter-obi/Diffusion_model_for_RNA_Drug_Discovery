import pickle
with open('processed_data/index.pkl', 'rb') as f:
    index = pickle.load(f)

print(index)

