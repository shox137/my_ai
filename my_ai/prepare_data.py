"""Builds data/corpus.txt, the plain text your foundation model learns language from.
Options:
  python prepare_data.py --tinystories 300000     download that many TinyStories stories (needs internet + `datasets`)
  python prepare_data.py --from-dir my_texts      use every .txt file in a folder you provide
Each document ends with <eos> so the model learns where texts stop."""
import argparse
import glob
import os

p = argparse.ArgumentParser()
p.add_argument("--tinystories", type=int, default=0, help="number of stories to download")
p.add_argument("--from-dir", default="", help="folder of your own .txt files")
p.add_argument("--out", default="data/corpus.txt")
a = p.parse_args()
os.makedirs("data", exist_ok=True)

n = 0
with open(a.out, "w", encoding="utf-8") as out:
    if a.tinystories:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        for row in ds:
            text = " ".join(row["text"].split())
            if len(text) < 50:
                continue
            out.write(text + " <eos>\n")
            n += 1
            if n >= a.tinystories:
                break
    if a.from_dir:
        for path in sorted(glob.glob(os.path.join(a.from_dir, "**", "*.txt"), recursive=True)):
            text = " ".join(open(path, encoding="utf-8", errors="ignore").read().split())
            if len(text) < 50:
                continue
            out.write(text + " <eos>\n")
            n += 1

if n == 0:
    raise SystemExit("Nothing written. Use --tinystories N or --from-dir FOLDER.")
print(f"wrote {n} documents to {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")
