#!/usr/bin/env bash
# Runs the WHOLE pipeline on a tiny model with sample data. Takes a few minutes on CPU.
# If this finishes and main.py prints an answer, your setup works. The answers will be poor: that is expected.
set -e
python check_env.py
python make_sample_data.py
python train_tokenizer.py --size smoke --corpus data/corpus_smoke.txt
python pretrain.py --steps 400 --bs 16 --eval-every 100
for f in biology coding math general human_style; do
  python specialist.py --field $f --steps 300
done
python router.py
python main.py "What does a ribosome do?"
echo "--- output.txt ---"
cat output.txt
