# My AI, built from zero

Every model here is trained by you from random weights. No pretrained model is downloaded.

## What is in the folder

| File | What it does |
| --- | --- |
| `check_env.py` | Confirms libraries work and whether a GPU is available |
| `config.py` | Model sizes (`smoke`, `tiny`, `small`). All scripts read it |
| `model.py` | The GPT, text generation, and the adapter (LoRA) layer |
| `make_sample_data.py` | Makes SAMPLE data to test the pipeline (not real training data) |
| `prepare_data.py` | Builds `data/corpus.txt` from TinyStories or your own .txt files |
| `train_tokenizer.py` | Trains your tokenizer and converts the corpus to numbers |
| `pretrain.py` | Trains the foundation model -> `base.pt` |
| `specialist.py` | Trains one adapter (a specialist) -> `adapters/<field>.pt` |
| `router.py` | Trains the router -> `router.joblib` |
| `main.py` | router -> specialist -> human-style -> `output.txt` |
| `run_smoke_test.sh` | Runs everything on a tiny model in a few minutes |

## Order of work

1. Install: `pip install -r requirements.txt`, then `python check_env.py`
2. Smoke test (do this first): `bash run_smoke_test.sh`
3. Real run (use a GPU, e.g. free Colab or Kaggle):
   ```
   python prepare_data.py --tinystories 300000
   python train_tokenizer.py --size tiny
   python pretrain.py --steps 10000          # re-run with --resume if the session disconnects
   ```
4. Put your own field data in `data/<field>.txt`, one chat-format example per line:
   `<user> question <assistant> answer <eos>`
   then `python specialist.py --field biology`
5. Fill `data/router_data.csv` (`text,label`, labels = field names), then `python router.py`
6. `python main.py "your question"` or `python main.py` for a chat loop

## Rules that avoid common mistakes

- Pick the size once, in `train_tokenizer.py --size ...`. Changing it later means retraining everything.
- Save `base.pt`, `tokenizer.json` and `adapters/` to Google Drive often. Free sessions delete files.
- The sample data only tests that the code runs. Real specialists need thousands of good examples.
- Watch the validation loss in `pretrain.py`. If train loss falls but val loss rises, you are overfitting.
- Free GPUs realistically handle `tiny` (about 23M parameters).

## Working with an AI coding agent

Point it at `AGENTS.md`. It explains the rules, the file map, the honest status and a prioritized improvement list.
