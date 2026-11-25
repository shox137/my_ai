"""Creates SMALL SAMPLE data so you can test the whole pipeline in minutes.
This is only for checking that the code runs. A real specialist needs thousands of good examples.
Writes: data/router_data.csv, data/<field>.txt for each field, data/human_style.txt, data/corpus_smoke.txt"""
import csv
import os
import random

random.seed(0)
os.makedirs("data", exist_ok=True)

BIO = [
    ("a ribosome", "A ribosome builds proteins by reading messenger RNA and joining amino acids in order."),
    ("mitochondria", "Mitochondria produce most of the cell's ATP, which is the energy the cell uses."),
    ("DNA", "DNA stores the genetic instructions that cells use to build and run an organism."),
    ("a chloroplast", "Chloroplasts carry out photosynthesis, turning light, water and carbon dioxide into sugar and oxygen."),
    ("an enzyme", "Enzymes are proteins that speed up chemical reactions in living things."),
    ("a red blood cell", "Red blood cells carry oxygen around the body using a protein called hemoglobin."),
    ("a neuron", "Neurons send signals through the body as electrical and chemical messages."),
    ("the nucleus", "The nucleus holds the cell's DNA and controls many of the cell's activities."),
    ("the cell membrane", "The cell membrane controls which substances can enter and leave the cell."),
    ("photosynthesis", "Photosynthesis is how plants turn sunlight into chemical energy stored in sugar."),
]
BIO_Q = ["What is {x}?", "Explain {x} in simple words.", "Tell me about {x}.", "What does biology say about {x}?"]

CODE = [
    ("reverse a string in Python", "Use slicing: s[::-1] returns the string reversed."),
    ("loop over a list in Python", "Use a for loop: for item in items: print(item)."),
    ("read a file in Python", "Use with open('file.txt') as f: text = f.read() so the file closes automatically."),
    ("define a function in Python", "Use def: def add(a, b): return a + b."),
    ("make a dictionary in Python", "Use curly braces with key-value pairs: d = {'a': 1, 'b': 2}."),
    ("sort a list in Python", "Call sorted(items) for a new sorted list or items.sort() to sort in place."),
    ("check if a number is even in Python", "Use the remainder operator: n % 2 == 0 is True for even numbers."),
    ("count the items in a list in Python", "Use len(items) to get how many items a list has."),
]
CODE_Q = ["How do I {x}?", "Show me how to {x}.", "Write code to {x}.", "What is the easiest way to {x}?"]

GEN = [
    ("get better sleep", "Keep a regular bedtime, avoid screens before bed, and keep your room dark and cool."),
    ("stay focused while studying", "Remove distractions, work in short blocks, and take regular breaks."),
    ("boil an egg", "Put the egg in boiling water for about 8 to 10 minutes, then cool it in cold water."),
    ("learn a new language", "Practice a little every day, listen a lot, and speak early even if you make mistakes."),
    ("save money each month", "Track your spending, set a fixed amount aside first, and cut costs you do not notice."),
    ("stay healthy", "Sleep enough, move every day, eat plenty of plants, and keep in touch with people."),
]
GEN_Q = ["Give me tips to {x}.", "How can I {x}?", "Any advice on how to {x}?", "What is a good way to {x}?"]

FACTS_GEN = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Who was the first person on the moon?", "Neil Armstrong was the first person to walk on the moon, in 1969."),
    ("At what temperature does water freeze?", "Water freezes at 0 degrees Celsius."),
    ("How many days are in a week?", "There are seven days in a week."),
]


def math_pairs(n):
    out = []
    for _ in range(n):
        a, b = random.randint(2, 99), random.randint(2, 99)
        kind = random.choice(["plus", "minus", "times", "eq"])
        if kind == "plus":
            out.append((f"What is {a} plus {b}?", f"{a} plus {b} is {a + b}."))
        elif kind == "minus":
            out.append((f"What is {a} minus {b}?", f"{a} minus {b} is {a - b}."))
        elif kind == "times":
            out.append((f"What is {a} times {b}?", f"{a} times {b} is {a * b}."))
        else:
            x, m = random.randint(1, 20), random.randint(2, 9)
            c = m * x + b
            out.append((f"Solve {m}x + {b} = {c}.", f"Subtract {b} from both sides to get {m}x = {c - b}, so x = {x}."))
    return out


def expand(items, templates):
    out = []
    for x, ans in items:
        for t in templates:
            out.append((t.format(x=x), ans))
    return out


fields = {
    "biology": expand(BIO, BIO_Q),
    "coding": expand(CODE, CODE_Q),
    "math": math_pairs(80),
    "general": expand(GEN, GEN_Q) + FACTS_GEN * 3,
}

# 1) router training data (question -> field)
rows = [(q, f) for f, pairs in fields.items() for q, _ in pairs]
random.shuffle(rows)
with open("data/router_data.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["text", "label"])
    w.writerows(rows)

# 2) one chat-format file per field (what specialist.py trains on)
def chat(q, a):
    return f"<user> {q} <assistant> {a} <eos>"

for f, pairs in fields.items():
    lines = [chat(q, a) for q, a in pairs]
    random.shuffle(lines)
    with open(f"data/{f}.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines * 3) + "\n")

# 3) human-style data: stiff answer -> friendlier answer
OPENERS = ["Sure! ", "Good question. ", "Happy to help. ", "Here is the short version: "]
style = []
for pairs in fields.values():
    for _, a in pairs[:40]:
        natural = random.choice(OPENERS) + a
        style.append(f"<user> Rewrite naturally: {a} <assistant> {natural} <eos>")
random.shuffle(style)
with open("data/human_style.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(style * 3) + "\n")

# 4) a tiny pretraining corpus for the smoke test only
docs = [a for pairs in fields.values() for _, a in pairs]
with open("data/corpus_smoke.txt", "w", encoding="utf-8") as fh:
    for _ in range(60):
        random.shuffle(docs)
        fh.write("\n".join(d + " <eos>" for d in docs) + "\n")

print("wrote data/router_data.csv, data/{biology,coding,math,general,human_style}.txt, data/corpus_smoke.txt")
