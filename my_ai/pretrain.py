"""Step 2: train the foundation model from random weights.
  python pretrain.py --steps 10000
Saves base.pt every 1000 steps (free GPU sessions disconnect: re-run with --resume to continue)."""
import argparse
import contextlib
import math
import os

import numpy as np
import torch
from tokenizers import Tokenizer

import config
from model import GPT, count_params, generate

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=10000)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--warmup", type=int, default=200)
p.add_argument("--eval-every", type=int, default=200)
p.add_argument("--resume", action="store_true")
a = p.parse_args()

name, cfg = config.get()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)

data = np.load("data/corpus_ids.npy", mmap_mode="r")
n_val = max(len(data) // 20, cfg["ctx"] + 2)
train, val = data[:-n_val], data[-n_val:]
CTX = cfg["ctx"]
if len(train) < CTX + 2:
    raise SystemExit(f"Corpus too small ({len(data)} tokens) for context {CTX}. Add more text.")


def get_batch(arr):
    ix = np.random.randint(0, len(arr) - CTX - 1, a.bs)
    x = np.stack([arr[i:i + CTX] for i in ix]).astype(np.int64)
    y = np.stack([arr[i + 1:i + CTX + 1] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)


model = GPT(**cfg).to(dev)
if a.resume and os.path.exists("base.pt"):
    model.load_state_dict(torch.load("base.pt", map_location=dev))
    print("resumed from base.pt")
print(f"size={name}  params={count_params(model) / 1e6:.1f}M  device={dev}  train tokens={len(train):,}")

decay = [p_ for p_ in model.parameters() if p_.dim() >= 2]
no_decay = [p_ for p_ in model.parameters() if p_.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=a.lr, betas=(0.9, 0.95))

use_cuda = dev == "cuda"
amp_dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16
scaler = torch.cuda.amp.GradScaler(enabled=use_cuda and amp_dtype == torch.float16)


def autocast():
    return torch.autocast(device_type="cuda", dtype=amp_dtype) if use_cuda else contextlib.nullcontext()


def lr_at(step):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    t = (step - a.warmup) / max(1, a.steps - a.warmup)
    return 0.1 * a.lr + 0.5 * (a.lr - 0.1 * a.lr) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate(batches=20):
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = get_batch(val)
        with autocast():
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / batches


for step in range(a.steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    x, y = get_batch(train)
    with autocast():
        _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()

    if step % a.eval_every == 0 or step == a.steps - 1:
        print(f"step {step:6d}  train {loss.item():.3f}  val {evaluate():.3f}")
    if step > 0 and step % 1000 == 0:
        torch.save(model.state_dict(), "base.pt")

torch.save(model.state_dict(), "base.pt")
print("saved base.pt")

tok = Tokenizer.from_file("tokenizer.json")
print("--- sample ---")
print(generate(model, tok, "The", max_new=80))
