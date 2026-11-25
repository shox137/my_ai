"""Step 3: train ONE adapter (a specialist) on top of the frozen foundation model.
  python specialist.py --field biology
Reads data/<field>.txt (one chat-format example per line), writes adapters/<field>.pt"""
import argparse
import os

import numpy as np
import torch
from tokenizers import Tokenizer

import config
from model import GPT, add_lora, freeze_all_but_adapters, save_adapter, count_params, generate

p = argparse.ArgumentParser()
p.add_argument("--field", required=True)
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--bs", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--rank", type=int, default=8)
a = p.parse_args()

name, cfg = config.get()
dev = "cuda" if torch.cuda.is_available() else "cpu"
CTX = cfg["ctx"]
tok = Tokenizer.from_file("tokenizer.json")

path = f"data/{a.field}.txt"
if not os.path.exists(path):
    raise SystemExit(f"Missing {path}. Create it first (see make_sample_data.py for the format).")
ids = np.array(tok.encode(open(path, encoding="utf-8").read()).ids, dtype=np.int64)
if len(ids) < CTX + 2:                       # tiny files: repeat them so a full window fits
    ids = np.tile(ids, (CTX + 2) // max(len(ids), 1) + 1)
data = torch.from_numpy(ids)
print(f"{a.field}: {len(data):,} tokens")

model = GPT(**cfg)
model.load_state_dict(torch.load("base.pt", map_location="cpu"))   # 1) load base  2) add adapters
add_lora(model, r=a.rank)
freeze_all_but_adapters(model)
model.to(dev)
print(f"trainable adapter params: {count_params(model, trainable_only=True):,}")

opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr)
model.train()
for step in range(a.steps):
    ix = torch.randint(len(data) - CTX - 1, (a.bs,))
    x = torch.stack([data[i:i + CTX] for i in ix]).to(dev)
    y = torch.stack([data[i + 1:i + CTX + 1] for i in ix]).to(dev)
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % 100 == 0 or step == a.steps - 1:
        print(f"{a.field} step {step:5d} loss {loss.item():.3f}")

os.makedirs("adapters", exist_ok=True)
save_adapter(model, f"adapters/{a.field}.pt")
print(f"saved adapters/{a.field}.pt")
if a.field != "human_style":
    print("test answer:", generate(model, tok, "<user> What is photosynthesis? <assistant>", max_new=60))
