"""Small trainable models built on :mod:`genesis.nn.tensor` (no pretrained weights).

Every model declares its parameters through ``param_specs(**config)`` — an
ordered list of ``(name, shape, init)`` computed without allocating anything —
which is the single source of truth for construction, ``named_parameters()``
order and safe deserialisation. Initial weights come only from
``random.Random(seed)``, so a config fully determines the untrained model.
"""

from __future__ import annotations

import math
import random
from array import array

from . import backend
from .tensor import (Tensor, add, causal_self_attention, cross_entropy, embedding, gelu, layer_norm, matmul,
                     mean_rows, no_grad, softmax_list, tanh)


def _check_int(name: str, value, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")


class Module:
    """Ordered parameter container shared by the models."""

    config: dict

    def _build(self, specs, seed: int) -> None:
        rng = random.Random(seed)
        self._params: list[tuple[str, Tensor]] = []
        for name, shape, init in specs:
            n = math.prod(shape)
            if init == "ones":
                data = array("d", [1.0]) * n
            elif init == "zeros":
                data = array("d", bytes(8 * n))
            else:
                data = array("d", [rng.gauss(0.0, init) for _ in range(n)])
            t = Tensor(data, shape, requires_grad=True)
            self._params.append((name, t))
            setattr(self, name.replace(".", "_"), t)

    def named_parameters(self) -> list[tuple[str, Tensor]]:
        return list(self._params)

    def parameters(self) -> list[Tensor]:
        return [p for _, p in self._params]

    def n_params(self) -> int:
        return sum(len(p.data) for _, p in self._params)

    def zero_grad(self) -> None:
        for _, p in self._params:
            p.grad = None


class MLPClassifier(Module):
    """Bag-of-tokens classifier: embedding -> mean over tokens -> Linear -> tanh -> Linear."""

    def __init__(self, vocab_size: int, n_classes: int, dim: int = 32, hidden: int = 64, seed: int = 0):
        specs = self.param_specs(vocab_size, n_classes, dim, hidden, seed)
        self.config = {"vocab_size": vocab_size, "n_classes": n_classes, "dim": dim, "hidden": hidden,
                       "seed": seed}
        self._build(specs, seed)

    @staticmethod
    def param_specs(vocab_size: int, n_classes: int, dim: int = 32, hidden: int = 64, seed: int = 0):
        for k, v in (("vocab_size", vocab_size), ("n_classes", n_classes), ("dim", dim), ("hidden", hidden)):
            _check_int(k, v)
        _check_int("seed", seed, minimum=-(1 << 63))
        return [("emb.weight", (vocab_size, dim), 1.0),
                ("fc1.weight", (dim, hidden), dim ** -0.5), ("fc1.bias", (hidden,), "zeros"),
                ("fc2.weight", (hidden, n_classes), hidden ** -0.5), ("fc2.bias", (n_classes,), "zeros")]

    def forward(self, ids) -> Tensor:
        ids = list(ids)
        if not ids:
            raise ValueError("MLPClassifier needs at least one token id")
        h = mean_rows(embedding(self.emb_weight, ids))
        h = tanh(add(matmul(h, self.fc1_weight), self.fc1_bias))
        return add(matmul(h, self.fc2_weight), self.fc2_bias)

    def loss(self, ids, label: int) -> Tensor:
        return cross_entropy(self.forward(ids), [label])

    def predict_proba(self, ids) -> list[float]:
        with no_grad():
            return softmax_list(self.forward(ids).data)

    def predict(self, ids) -> int:
        with no_grad():
            logits = list(self.forward(ids).data)
        return logits.index(max(logits))


class TinyGPT(Module):
    """Decoder-only pre-LN Transformer with learned positions and a tied output head."""

    def __init__(self, vocab_size: int, block_size: int = 64, dim: int = 32, n_layers: int = 2,
                 n_heads: int = 2, mlp_mult: int = 4, seed: int = 0):
        specs = self.param_specs(vocab_size, block_size, dim, n_layers, n_heads, mlp_mult, seed)
        self.config = {"vocab_size": vocab_size, "block_size": block_size, "dim": dim, "n_layers": n_layers,
                       "n_heads": n_heads, "mlp_mult": mlp_mult, "seed": seed}
        self._build(specs, seed)
        self.blocks = [{k: getattr(self, f"blocks_{i}_{k}") for k in
                        ("ln1_gamma", "ln1_beta", "attn_w_qkv", "attn_b_qkv", "attn_w_o", "attn_b_o",
                         "ln2_gamma", "ln2_beta", "mlp_w_fc", "mlp_b_fc", "mlp_w_proj", "mlp_b_proj")}
                       for i in range(n_layers)]

    @staticmethod
    def param_specs(vocab_size: int, block_size: int = 64, dim: int = 32, n_layers: int = 2, n_heads: int = 2,
                    mlp_mult: int = 4, seed: int = 0):
        for k, v in (("vocab_size", vocab_size), ("block_size", block_size), ("dim", dim),
                     ("n_layers", n_layers), ("n_heads", n_heads), ("mlp_mult", mlp_mult)):
            _check_int(k, v)
        _check_int("seed", seed, minimum=-(1 << 63))
        if dim % n_heads:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        std, proj_std, D, F = 0.02, 0.02 / math.sqrt(2 * n_layers), dim, mlp_mult * dim
        specs = [("tok_emb", (vocab_size, D), std), ("pos_emb", (block_size, D), std)]
        for i in range(n_layers):
            p = f"blocks.{i}."
            specs += [(p + "ln1.gamma", (D,), "ones"), (p + "ln1.beta", (D,), "zeros"),
                      (p + "attn.w_qkv", (D, 3 * D), std), (p + "attn.b_qkv", (3 * D,), "zeros"),
                      (p + "attn.w_o", (D, D), proj_std), (p + "attn.b_o", (D,), "zeros"),
                      (p + "ln2.gamma", (D,), "ones"), (p + "ln2.beta", (D,), "zeros"),
                      (p + "mlp.w_fc", (D, F), std), (p + "mlp.b_fc", (F,), "zeros"),
                      (p + "mlp.w_proj", (F, D), proj_std), (p + "mlp.b_proj", (D,), "zeros")]
        return specs + [("ln_f.gamma", (D,), "ones"), ("ln_f.beta", (D,), "zeros")]

    def hidden(self, ids) -> Tensor:
        """Final-LayerNorm hidden states [T,D] for 1 <= len(ids) <= block_size."""
        ids = list(ids)
        if not 1 <= len(ids) <= self.config["block_size"]:
            raise ValueError(f"sequence length must be in [1, {self.config['block_size']}], got {len(ids)}")
        H = self.config["n_heads"]
        x = add(embedding(self.tok_emb, ids), embedding(self.pos_emb, list(range(len(ids)))))
        for b in self.blocks:
            h = layer_norm(x, b["ln1_gamma"], b["ln1_beta"])
            x = add(x, causal_self_attention(h, b["attn_w_qkv"], b["attn_b_qkv"], b["attn_w_o"], b["attn_b_o"], H))
            h = layer_norm(x, b["ln2_gamma"], b["ln2_beta"])
            h = gelu(add(matmul(h, b["mlp_w_fc"]), b["mlp_b_fc"]))
            x = add(x, add(matmul(h, b["mlp_w_proj"]), b["mlp_b_proj"]))
        return layer_norm(x, self.ln_f_gamma, self.ln_f_beta)

    def forward(self, ids) -> Tensor:
        """Logits [T,V]; the head reuses the token embedding (h @ W_tok^T)."""
        return matmul(self.hidden(ids), self.tok_emb, trans_b=True)

    def loss(self, ids, targets) -> Tensor:
        targets = list(targets)
        if len(targets) != len(ids):
            raise ValueError("ids and targets must have the same length")
        return cross_entropy(self.forward(ids), targets)

    def next_logits(self, ids) -> list[float]:
        """Logits for the token following ``ids`` (cropped to the last block_size tokens)."""
        ids = list(ids)[-self.config["block_size"]:]
        V, D = self.tok_emb.shape
        with no_grad():
            h = self.hidden(ids).data[(len(ids) - 1) * D:]
            return list(backend.matmul(h, self.tok_emb.data, 1, D, V, trans_b=True))

    def generate(self, ids, max_new: int, rng: random.Random, temperature: float = 1.0, top_k=None,
                 stop_ids=()) -> list[int]:
        """Sample up to ``max_new`` tokens after ``ids`` and return only the new ones.

        ``temperature <= 0`` is greedy argmax. ``top_k`` restricts sampling to the k
        best logits (ties broken by lower id). If a token in ``stop_ids`` is drawn,
        generation ends and that token is not included in the result.
        """
        seq, new, stop = list(ids), [], set(stop_ids)
        if not seq:
            raise ValueError("generate needs a non-empty prompt")
        for _ in range(max_new):
            logits = self.next_logits(seq)
            if temperature <= 0:
                nxt = logits.index(max(logits))
            else:
                cand = list(range(len(logits)))
                if top_k is not None and 0 < top_k < len(cand):
                    cand = sorted(cand, key=lambda i: (-logits[i], i))[:top_k]
                    cand.sort()
                probs = softmax_list([logits[i] for i in cand], temperature)
                r, acc, nxt = rng.random(), 0.0, cand[-1]
                for i, p in zip(cand, probs):
                    acc += p
                    if r < acc:
                        nxt = i
                        break
            if nxt in stop:
                break
            seq.append(nxt)
            new.append(nxt)
        return new
