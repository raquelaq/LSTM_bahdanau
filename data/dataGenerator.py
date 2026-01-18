import random

subjects = ["i", "you", "we", "they", "he", "she"]
verbs = ["like", "love", "use", "study", "learn", "build", "train", "debug"]
objects = ["nlp", "pytorch", "attention", "models", "data", "systems", "agents", "embeddings"]

def generate_sentence():
    length = random.choice([3, 4, 5, 6])
    words = []
    words.append(random.choice(subjects))
    words.append(random.choice(verbs))
    while len(words) < length:
        words.append(random.choice(objects))
    return " ".join(words)

N = 5000

with open("reverse_dataset.txt", "w", encoding="utf-8") as f:
    for _ in range(N):
        src = generate_sentence()
        trg = " ".join(src.split()[::-1])
        f.write(f"{src} ||| {trg}\n")

print(f"Dataset generado con {N} ejemplos en reverse_dataset.txt")
