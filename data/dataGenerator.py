import random
from pathlib import Path

random.seed(42)

NOUNS = [
    ("el gato", "the cat"), ("el perro", "the dog"), ("la casa", "the house"),
    ("el coche", "the car"), ("la universidad", "the university"),
    ("el libro", "the book"), ("la ciudad", "the city"),
    ("mi amigo", "my friend"), ("mi hermana", "my sister"),
    ("el profesor", "the teacher"), ("el estudiante", "the student"),
]
PLACES = [
    ("en casa", "at home"), ("en la universidad", "at the university"),
    ("en el trabajo", "at work"), ("en el parque", "in the park"),
    ("en la ciudad", "in the city"), ("en el restaurante", "in the restaurant"),
]
ADJS = [
    ("feliz", "happy"), ("triste", "sad"), ("cansado", "tired"),
    ("ocupado", "busy"), ("listo", "smart"), ("nuevo", "new"),
    ("viejo", "old"), ("grande", "big"), ("pequeño", "small"),
]
VERBS = [
    ("como", "eat"), ("leo", "read"), ("estudio", "study"), ("trabajo", "work"),
    ("aprendo", "learn"), ("escucho música", "listen to music"),
    ("veo una película", "watch a movie"),
]
TIMES = [
    ("hoy", "today"), ("mañana", "tomorrow"), ("ahora", "now"),
    ("cada día", "every day"), ("a veces", "sometimes"),
]

TEMPLATES = [
    # (es_template, en_template, required_fields)
    ("{time} {subj} {verb}", "{time_en} {subj_en} {verb_en}", ("time","subj","verb")),
    ("{subj} está {adj} {place}", "{subj_en} is {adj_en} {place_en}", ("subj","adj","place")),
    ("{time} {subj} está {adj}", "{time_en} {subj_en} is {adj_en}", ("time","subj","adj")),
    ("{subj} {verb} {place}", "{subj_en} {verb_en} {place_en}", ("subj","verb","place")),
]

SUBJECTS = [
    ("yo", "i"), ("tú", "you"), ("él", "he"), ("ella", "she"),
    ("nosotros", "we"), ("ellos", "they"),
]

def sample_pair():
    es_t, en_t, fields = random.choice(TEMPLATES)

    subj_es, subj_en = random.choice(SUBJECTS)
    noun_es, noun_en = random.choice(NOUNS)
    # Mezcla: a veces el sujeto es pronombre, a veces SN
    if random.random() < 0.5:
        subj = (subj_es, subj_en)
    else:
        subj = (noun_es, noun_en)

    verb_es, verb_en = random.choice(VERBS)
    time_es, time_en = random.choice(TIMES)
    adj_es, adj_en = random.choice(ADJS)
    place_es, place_en = random.choice(PLACES)

    mapping = {
        "subj": subj[0], "subj_en": subj[1],
        "verb": verb_es, "verb_en": verb_en,
        "time": time_es, "time_en": time_en,
        "adj": adj_es, "adj_en": adj_en,
        "place": place_es, "place_en": place_en,
    }

    es = es_t.format(**mapping).strip()
    en = en_t.format(**mapping).strip()
    return es, en

def generate_data(n=2000, out_path="data/es-en.txt"):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    pairs = []
    while len(pairs) < n:
        es, en = sample_pair()
        key = (es, en)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(f"{es} ||| {en}")

    out.write_text("\n".join(pairs), encoding="utf-8")
    print(f"Wrote {len(pairs)} pairs to {out}")

