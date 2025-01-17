import markovify
import sys
import os
import pickle

# Read text from file
if len(sys.argv) != 2:
    sys.exit("Usage: python generator.py sample.txt")
with open(sys.argv[1]) as f:
    text = f.read()

# Train model or load model
if input("Train model? (y/n): ").lower() == "y" or not os.path.exists("model.pkl"):
    text_model = markovify.Text(text)
    with open("model.pkl", "wb") as f:
        pickle.dump(text_model, f)
else:
    with open("model.pkl", "rb") as f:
        text_model = pickle.load(f)

# Generate sentences
print()
while input("Generate sentence? (y/n): ").lower() == "y":
    for i in range(5):
        print(text_model.make_sentence())
        print()
