"""Shared settings. Every script reads the model size from here so they always agree."""
import json
import os

# vocab = tokenizer size, d = width, h = attention heads, layers = depth, ctx = context length
SIZES = {
    "smoke": dict(vocab=512,   d=64,  h=4,  layers=2,  ctx=64),    # seconds on CPU: tests the pipeline
    "tiny":  dict(vocab=16000, d=384, h=6,  layers=6,  ctx=256),   # ~23M params: free Colab/Kaggle GPU
    "small": dict(vocab=32000, d=768, h=12, layers=12, ctx=512),   # ~160M params: rented GPU
}
RUN_FILE = "run_config.json"


def set_size(name):
    if name not in SIZES:
        raise SystemExit(f"Unknown size '{name}'. Choose from: {', '.join(SIZES)}")
    with open(RUN_FILE, "w") as f:
        json.dump({"size": name}, f)


def get():
    """Return (size_name, model_kwargs). Defaults to 'tiny' until train_tokenizer.py sets it."""
    name = "tiny"
    if os.path.exists(RUN_FILE):
        with open(RUN_FILE) as f:
            name = json.load(f)["size"]
    return name, dict(SIZES[name])
