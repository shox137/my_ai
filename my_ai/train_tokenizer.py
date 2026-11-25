"""Step 1: train YOUR tokenizer on your corpus, then turn the corpus into numbers.
  python train_tokenizer.py --size tiny
Writes: tokenizer.json, data/corpus_ids.npy, run_config.json"""
import argparse
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import config

p = argparse.ArgumentParser()
p.add_argument("--size", default="tiny", choices=list(config.SIZES))
p.add_argument("--corpus", default="data/corpus.txt")
a = p.parse_args()

config.set_size(a.size)
_, cfg = config.get()

tok = Tokenizer(models.BPE(unk_token="<unk>"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=cfg["vocab"],
    special_tokens=["<unk>", "<pad>", "<eos>", "<user>", "<assistant>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
tok.train([a.corpus], trainer)
tok.save("tokenizer.json")
print("vocab size:", tok.get_vocab_size())

# Encode the corpus in chunks so big files do not fill memory.
chunks, buf, size = [], [], 0


def flush():
    global buf, size
    for enc in tok.encode_batch(buf):
        chunks.append(np.array(enc.ids, dtype=np.uint16))
    buf, size = [], 0


with open(a.corpus, encoding="utf-8") as f:
    for line in f:
        buf.append(line)
        size += len(line)
        if size > 5_000_000:
            flush()
if buf:
    flush()

ids = np.concatenate(chunks)
np.save("data/corpus_ids.npy", ids)
print(f"saved {len(ids):,} tokens to data/corpus_ids.npy")
