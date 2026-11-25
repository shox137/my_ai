"""The whole model in one file: a GPT, the generation loop, and the adapter (LoRA) layer.
Everything here is written from zero; the only library used is PyTorch."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(nn.Module):
    """Causal self-attention: every token looks at the tokens before it."""

    def __init__(self, d, h):
        super().__init__()
        assert d % h == 0, "width d must be divisible by number of heads h"
        self.h = h
        self.qkv = nn.Linear(d, 3 * d)   # plain nn.Linear on purpose: adapters attach to these
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    """One transformer block: attention, then a small feed-forward network."""

    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attn(d, h)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, vocab, d=384, h=6, layers=6, ctx=256):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.Sequential(*[Block(d, h) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.ctx, f"sequence length {T} is longer than ctx {self.ctx}"
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        logits = self.head(self.ln(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


@torch.no_grad()
def generate(model, tok, prompt, max_new=200, temp=0.8, top_k=40):
    """Continue `prompt`. Stops at <eos>. Returns only the new text."""
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    ids = torch.tensor([tok.encode(prompt).ids], device=dev)
    start = ids.size(1)
    eos = tok.token_to_id("<eos>")
    for _ in range(max_new):
        logits, _ = model(ids[:, -model.ctx:])
        logits = logits[:, -1] / max(temp, 1e-5)
        if top_k:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        if nxt.item() == eos:
            break
        ids = torch.cat([ids, nxt], dim=1)
    model.train(was_training)
    return tok.decode(ids[0, start:].tolist()).strip()


# ---------------------------------------------------------------- adapters (LoRA)
class LoRALinear(nn.Module):
    """Freezes a Linear layer and adds a tiny trainable correction: x @ A.T @ B.T.
    B starts at zero, so a fresh adapter changes nothing until it is trained."""

    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def add_lora(model, r=8):
    """Wrap every Linear layer (except the output head) with an adapter. Call AFTER loading base weights."""
    for m in list(model.modules()):
        for name, child in list(m.named_children()):
            if isinstance(child, nn.Linear) and name != "head":
                setattr(m, name, LoRALinear(child, r))


def freeze_all_but_adapters(model):
    for n, p in model.named_parameters():
        p.requires_grad = n.endswith((".A", ".B"))


def save_adapter(model, path):
    sd = {k: v for k, v in model.state_dict().items() if k.endswith((".A", ".B"))}
    torch.save(sd, path)


def load_adapter(model, path):
    """Swap in a specialist: overwrites only the A and B matrices."""
    sd = torch.load(path, map_location=next(model.parameters()).device)
    result = model.load_state_dict(sd, strict=False)
    assert not result.unexpected_keys, f"adapter does not match this model: {result.unexpected_keys[:3]}"


def reset_adapter(model):
    """Back to the plain base model (B = 0 means 'no change')."""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            nn.init.zeros_(m.B)
