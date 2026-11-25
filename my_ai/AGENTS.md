# Instructions for AI coding agents

You are helping improve a small language-model system that is **built entirely from zero**:
own tokenizer, own GPT, own adapters, own router. Read this file fully before changing anything.

## The one hard rule
**Never load pretrained weights or pretrained tokenizers** (no `from_pretrained`, no downloaded checkpoints,
no fine-tuning of Llama/Qwen/GPT-2 etc.). Every weight must come from `pretrain.py` or `specialist.py`.
Libraries (PyTorch, tokenizers, scikit-learn) and public *text datasets* are fine.
If the owner wants to change this rule they will say so explicitly.

## What the system does
question -> **router** (`router.py`, TF-IDF + logistic regression) picks a field
-> **specialist adapter** (LoRA, `adapters/<field>.pt`) on top of one shared **foundation GPT** (`base.pt`)
-> optional **human_style adapter** rewrites the answer -> `output.txt` (+ log in `data/chat_log.jsonl`).
Only one adapter is active at a time, so memory stays low.

## Files
- `config.py`  model sizes (`smoke`, `tiny`, `small`); `run_config.json` stores the chosen size
- `model.py`   GPT, `generate`, LoRA (`LoRALinear`, `add_lora`, `save/load/reset_adapter`)
- `prepare_data.py` -> `train_tokenizer.py` -> `pretrain.py` -> `specialist.py` (per field) -> `router.py` -> `main.py`
- `make_sample_data.py` creates SAMPLE data only to test the pipeline
- `run_smoke_test.sh` runs the whole pipeline on a tiny model

## How to use it (run in this order)
```
pip install -r requirements.txt
python check_env.py
bash run_smoke_test.sh            # must finish and print an answer; answers will be poor, that is expected
```
Real run (GPU): `prepare_data.py --tinystories 300000` -> `train_tokenizer.py --size tiny`
-> `pretrain.py --steps 10000` -> `specialist.py --field <f>` -> `router.py` -> `main.py "question"`.
Adapter order matters: build GPT, load `base.pt`, THEN `add_lora`, then load an adapter.
The size is chosen once in `train_tokenizer.py --size ...`; changing it means retraining everything.

## Honest status
- **The torch code has never been executed by its author** (written in a sandbox without PyTorch).
  First job: run `bash run_smoke_test.sh`, fix whatever breaks, and report exactly what you changed.
- The router was tested (97% on sample data). Sample data is templated and tiny; accuracy there means little.
- Expect small models (23M params) to be fluent-ish but factually weak. Do not claim otherwise.

## Rules for any change
1. After every change, run `bash run_smoke_test.sh`. Do not say "works" unless it ran to the end.
2. Keep it readable: this repo is also a learning project. Prefer clear code over clever code; comment the why.
3. One improvement per commit, with a message saying what and why. Do not commit weights, `.npy`, or big data.
4. Add a short note to the README when you add a command or flag.
5. If you cannot run something (no GPU, no internet), say so and mark the change "untested".

## Improvement backlog (highest value first)
1. **Make it run**: smoke test end to end; fix bugs; add `tests/` (shape checks, adapter swap changes output,
   `reset_adapter` restores base output, tokenizer round-trip).
2. **Specialist loss masking**: in `specialist.py`, compute loss only on the answer tokens (after `<assistant>`),
   not on the question. Add a validation split per field and print validation loss.
3. **Resumable training**: save optimizer state + step in the checkpoint; `--resume` should continue the LR schedule.
4. **Gradient accumulation** (`--accum`) so bigger effective batches fit a free GPU; `torch.compile` option.
5. **Model upgrades (each one separately, compare val loss)**: tie input/output embeddings, RMSNorm,
   rotary position embeddings, weight-init scaling on residual projections.
6. **Faster generation**: KV cache in `generate`; repetition penalty; stop rule on max length per answer.
7. **Instruction data**: pack multiple short examples per training window; add a general Q&A pretraining mix so
   specialists do not forget how to chat.
8. **Better router**: multi-label routing, a confidence threshold with a "general" fallback, then a small neural
   classifier that reads the foundation model's hidden states. Keep TF-IDF as the baseline to beat.
9. **Multi-topic questions**: split a question into sub-questions before routing.
10. **Evaluation harness**: `eval.py` with held-out questions per field, router accuracy, and perplexity;
    store results in `results/` so changes are comparable.
11. **Scheduled human-style retraining**: script that filters `data/chat_log.jsonl` (rating >= good), mixes in
    replay data, trains `human_style_vN.pt`, keeps old versions for rollback.
12. **Senses (later, one at a time)**: tiny CNN image classifier, then one-language-pair translation; each outputs
    text so the rest never changes.

## Things not to do
- Do not make the model learn live during chats (forgetting, drift, no rollback). Retrain on a schedule.
- Do not log other people's chats without telling them.
- Do not treat model output as expert advice (medical, legal, safety).
