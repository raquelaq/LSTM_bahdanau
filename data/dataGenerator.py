import random

subjects = ["i", "you", "we", "they", "he", "she"]
verbs = ["like", "love", "use", "study", "learn"]
objects = ["nlp", "pytorch", "attention", "models", "data"]

def generate_sentence():
    return " ".join([
        random.choice(subjects),
        random.choice(verbs),
        random.choice(objects),
        random.choice(objects)
    ])

N = 1000  # con 500–1000 ya va sobrado

with open("reverse_dataset.txt", "w", encoding="utf-8") as f:
    for _ in range(N):
        src = generate_sentence()
        trg = " ".join(src.split()[::-1])
        f.write(f"{src} ||| {trg}\n")
