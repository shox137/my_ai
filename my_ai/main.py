"""Step 5: connect everything. router -> specialist -> human-style -> output.txt
  python main.py "What does a ribosome do?"      one question
  python main.py                                 chat loop (type 'quit' to stop)
Only the base model stays in memory; adapters are swapped by loading a small file."""
import argparse
import json
import os
import time

import joblib
import torch
from tokenizers import Tokenizer

import config
from model import GPT, add_lora, generate, load_adapter, reset_adapter

p = argparse.ArgumentParser()
p.add_argument("question", nargs="?", default=None)
p.add_argument("--max-new", type=int, default=120)
a = p.parse_args()

name, cfg = config.get()
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = Tokenizer.from_file("tokenizer.json")
router = joblib.load("router.joblib")
model = GPT(**cfg)
model.load_state_dict(torch.load("base.pt", map_location="cpu"))
add_lora(model, r=8)
model.to(dev).eval()


def use_adapter(field):
    path = f"adapters/{field}.pt"
    if os.path.exists(path):
        load_adapter(model, path)
        return True
    reset_adapter(model)             # no adapter yet: plain base model
    return False


def respond(question):
    field = router.predict([question])[0]
    found = use_adapter(field)
    answer = generate(model, tok, f"<user> {question} <assistant>", max_new=a.max_new)
    final = answer
    if os.path.exists("adapters/human_style.pt"):
        use_adapter("human_style")
        final = generate(model, tok, f"<user> Rewrite naturally: {answer} <assistant>", max_new=a.max_new) or answer
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(f"## {field}\n{final}\n")
    os.makedirs("data", exist_ok=True)
    with open("data/chat_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.time(), "user": question, "field": field,
                            "adapter_found": found, "reply": final, "rating": None}) + "\n")
    return field, final


if a.question:
    field, final = respond(a.question)
    print(f"[{field}] {final}")
else:
    print("Chat started. Type 'quit' to stop.")
    while True:
        q = input("you> ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        field, final = respond(q)
        print(f"[{field}] {final}\n")
