"""Run this first: confirms your libraries work and whether a GPU is available."""
import importlib

for lib in ["torch", "tokenizers", "numpy", "pandas", "sklearn", "joblib"]:
    try:
        mod = importlib.import_module(lib)
        print(f"ok       {lib} {getattr(mod, '__version__', '')}")
    except ImportError:
        print(f"MISSING  {lib}  -> pip install -r requirements.txt")

try:
    import torch
    if torch.cuda.is_available():
        print("GPU      ", torch.cuda.get_device_name(0))
    else:
        print("GPU       none found (CPU only: use --size smoke, or switch Colab to a GPU runtime)")
except ImportError:
    pass
