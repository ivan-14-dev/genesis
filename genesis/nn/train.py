"""Training and evaluation loops for :mod:`genesis.nn` models.

All randomness (window sampling, shuffling) comes from ``random.Random(seed)``
and every reduction runs in a fixed order, so a (model config, data, seed)
triple always yields bit-identical weights on a given backend.
"""

from __future__ import annotations

import math
import random

from .optim import AdamW, clip_grad_norm, cosine_lr
from .tensor import no_grad, scale


def evaluate_lm(model, ids) -> dict:
    """Mean NLL (nats/token) and perplexity over consecutive non-overlapping windows."""
    ids = list(ids)
    if len(ids) < 2:
        raise ValueError("evaluate_lm needs at least 2 tokens")
    B, total, n = model.config["block_size"], 0.0, 0
    with no_grad():
        for s in range(0, len(ids) - 1, B):
            y = ids[s + 1:s + B + 1]
            x = ids[s:s + len(y)]
            total += model.loss(x, y).item() * len(y)
            n += len(y)
    nll = total / n
    return {"nll": nll, "ppl": math.exp(min(nll, 700.0)), "tokens": n}


def train_lm(model, ids, steps: int, batch_size: int = 4, lr: float = 3e-3, seed: int = 0, valid_ids=None,
             eval_every=None, clip: float | None = 1.0, warmup: int = 0, log=None) -> dict:
    """Train a causal LM on random windows of a token stream with AdamW + cosine LR.

    Each step averages the gradients of ``batch_size`` windows of length
    ``block_size + 1`` (shorter if the stream is short). ``log`` (optional) is
    called with a dict per step. Returns ``{"train_loss", "lr", "grad_norm", "valid"}``.
    """
    ids = list(ids)
    if len(ids) < 2:
        raise ValueError("train_lm needs at least 2 tokens")
    if steps < 0 or batch_size < 1:
        raise ValueError("steps must be >= 0 and batch_size >= 1")
    win = min(model.config["block_size"], len(ids) - 1)
    rng = random.Random(seed)
    params = model.parameters()
    opt = AdamW(params, lr=lr)
    hist = {"train_loss": [], "lr": [], "grad_norm": [], "valid": []}
    for step in range(steps):
        opt.lr = cosine_lr(step, steps, lr, warmup)
        opt.zero_grad()
        total = 0.0
        for _ in range(batch_size):
            s = rng.randrange(len(ids) - win)
            loss = model.loss(ids[s:s + win], ids[s + 1:s + win + 1])
            scale(loss, 1.0 / batch_size).backward()
            total += loss.item()
        norm = clip_grad_norm(params, clip if clip else 0.0)
        opt.step()
        rec = {"step": step + 1, "loss": total / batch_size, "lr": opt.lr, "grad_norm": norm}
        hist["train_loss"].append(rec["loss"])
        hist["lr"].append(opt.lr)
        hist["grad_norm"].append(norm)
        if valid_ids is not None and eval_every and ((step + 1) % eval_every == 0 or step + 1 == steps):
            ev = evaluate_lm(model, valid_ids)
            hist["valid"].append({"step": step + 1, **ev})
            rec.update(valid_nll=ev["nll"], valid_ppl=ev["ppl"])
        if log is not None:
            log(rec)
    return hist


def evaluate_classifier(model, examples) -> float:
    """Accuracy of ``model.predict`` on ``[(ids, label), ...]`` (0.0 if empty)."""
    examples = list(examples)
    if not examples:
        return 0.0
    return sum(model.predict(ids) == label for ids, label in examples) / len(examples)


def train_classifier(model, examples, epochs: int, lr: float = 1e-2, seed: int = 0, batch_size: int = 8) -> dict:
    """Mini-batch AdamW on ``[(ids, label), ...]``, reshuffled each epoch with a seeded rng."""
    examples = list(examples)
    if not examples or batch_size < 1:
        raise ValueError("need examples and batch_size >= 1")
    rng = random.Random(seed)
    opt = AdamW(model.parameters(), lr=lr)
    order = list(range(len(examples)))
    hist = {"loss": [], "accuracy": []}
    for _ in range(epochs):
        rng.shuffle(order)
        total = 0.0
        for b in range(0, len(order), batch_size):
            chunk = order[b:b + batch_size]
            opt.zero_grad()
            for i in chunk:
                ids, label = examples[i]
                loss = model.loss(ids, label)
                scale(loss, 1.0 / len(chunk)).backward()
                total += loss.item()
            opt.step()
        hist["loss"].append(total / len(examples))
        hist["accuracy"].append(evaluate_classifier(model, examples))
    return hist
