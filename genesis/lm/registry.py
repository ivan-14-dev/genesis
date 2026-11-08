"""In-house language-model registry: train on approved documents, record provenance, activate on admin signature.

Life cycle of a model (table ``lm_models``)::

    train (lm.train) -> "trained" -> admin-signed approval -> "active" -> superseded -> "retired"

* Training reads **only** admin-approved ``train`` documents whose signature and hash verify
  (``corpus.approved_texts``); the tokenizer is learned from the same documents. Nothing is downloaded,
  nothing is pretrained: the provenance records the corpus digests, tokenizer digest, code digest, kernel
  digest and seed, and states ``pretrained: null``.
* Validation perplexity is always reported next to a unigram baseline on the same tokens, plus bits per byte
  (comparable across tokenizers). A model that does not beat the baseline cannot be approved.
* ``test`` documents are sealed: only :func:`evaluate` reads them, and each read is audited.
* The active model is loaded only if its weights hash and the admin signature over (model id, weights hash)
  still verify. Generated text is *never* stored as knowledge.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import struct
import time
import zlib
from typing import Optional

from .. import corpus, governance
from ..memory import LongTermMemory
from .bpe import BPETokenizer
from .ngram import NGramLM, UnigramLM

MAGIC = b"GLM1\n"
MAX_BLOB = 256 << 20
LIMITS = {"vocab_size": (260, 16000), "order": (1, 6), "dim": (8, 256), "n_layers": (1, 6), "n_heads": (1, 8),
          "block_size": (8, 256), "steps": (1, 20000), "batch_size": (1, 32), "max_params": 3_000_000}
DEFAULTS = {"ngram": {"order": 3, "vocab_size": 2000, "seed": 0},
            "gpt": {"vocab_size": 1000, "dim": 64, "n_layers": 2, "n_heads": 4, "block_size": 64, "steps": 300,
                    "batch_size": 4, "lr": 3e-3, "seed": 0}}
_CACHE: dict = {}


class LMIntegrityError(RuntimeError):
    pass


def code_digest() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    nn_dir = os.path.join(os.path.dirname(here), "nn")
    files = sorted(os.path.join(here, f) for f in os.listdir(here) if f.endswith(".py"))
    files += sorted(os.path.join(nn_dir, f) for f in os.listdir(nn_dir) if f.endswith((".py", ".c")))
    h = hashlib.sha256()
    for path in files:
        with open(path, "rb") as fh:
            h.update(os.path.basename(path).encode() + b"\0" + fh.read())
    return {"sha256": h.hexdigest(), "files": [os.path.basename(f) for f in files]}


def _check_config(kind: str, cfg: dict) -> dict:
    if kind not in DEFAULTS:
        raise ValueError(f"unknown model kind {kind!r} (choose from {sorted(DEFAULTS)})")
    unknown = set(cfg) - set(DEFAULTS[kind])
    if unknown:
        raise ValueError(f"unknown options for {kind}: {sorted(unknown)}")
    out = {**DEFAULTS[kind], **{k: v for k, v in cfg.items() if v is not None}}
    for k, v in out.items():
        if k == "lr":
            if not isinstance(v, (int, float)) or not 0 < v <= 1:
                raise ValueError("lr must be in (0, 1]")
        elif k in LIMITS:
            lo, hi = LIMITS[k]
            if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
                raise ValueError(f"{k} must be an integer in [{lo}, {hi}]")
        elif k == "seed" and (not isinstance(v, int) or isinstance(v, bool)):
            raise ValueError("seed must be an integer")
    if kind == "gpt" and out["dim"] % out["n_heads"]:
        raise ValueError("dim must be divisible by n_heads")
    return out


# ------------------------------------------------------------------ weights
def _pack(kind: str, tok: BPETokenizer, model) -> bytes:
    from ..nn import serialize
    header = {"format": 1, "kind": kind, "tokenizer": tok.to_dict(),
              "model": model.to_dict() if kind == "ngram" else None}
    hb = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    payload = serialize.dumps(model) if kind == "gpt" else b""
    return zlib.compress(MAGIC + struct.pack("<I", len(hb)) + hb + payload, 6)


def _unpack(blob: bytes):
    d = zlib.decompressobj()
    raw = d.decompress(blob, MAX_BLOB)
    if d.unconsumed_tail or not d.eof:
        raise LMIntegrityError("model blob too large or truncated")
    if not raw.startswith(MAGIC):
        raise LMIntegrityError("not a GENESIS language-model blob")
    (n,) = struct.unpack("<I", raw[len(MAGIC):len(MAGIC) + 4])
    start = len(MAGIC) + 4
    header = json.loads(raw[start:start + n])
    tok = BPETokenizer.from_dict(header["tokenizer"])
    if header["kind"] == "ngram":
        return header["kind"], tok, NGramLM.from_dict(header["model"])
    if header["kind"] == "gpt":
        from ..nn import serialize
        return header["kind"], tok, serialize.loads(raw[start + n:])
    raise LMIntegrityError(f"unknown model kind {header['kind']!r}")


# ----------------------------------------------------------------- training
def _bits_per_byte(nll_total: float, texts) -> float:
    return nll_total / math.log(2) / max(1, sum(len(t.encode("utf-8")) for t in texts))


def _ngram_eval(model, uni, seqs, texts) -> dict:
    n_tok = sum(len(s) + 1 for s in seqs)  # EOS predictions included
    ppl, base = model.perplexity(seqs), uni.perplexity(seqs)
    return {"perplexity": round(ppl, 4), "unigram_perplexity": round(base, 4), "tokens": n_tok, "docs": len(seqs),
            "bits_per_byte": round(_bits_per_byte(n_tok * math.log(ppl), texts), 4),
            "unigram_bits_per_byte": round(_bits_per_byte(n_tok * math.log(base), texts), 4)}


def _stream(tok: BPETokenizer, texts) -> list:
    ids = []
    for t in texts:
        ids += tok.encode(t, add_bos=True, add_eos=True)
    return ids


def _gpt_eval(model, uni, tok, texts) -> dict:
    """Scored per document on exactly the tokens the unigram baseline predicts (document + EOS)."""
    from ..nn import train as nn_train
    seqs = [tok.encode(t) for t in texts]
    total, n_tok = 0.0, 0
    for s in seqs:
        res = nn_train.evaluate_lm(model, [tok.bos_id] + s + [tok.eos_id])
        total += res["nll"] * res["tokens"]
        n_tok += res["tokens"]
    base = uni.perplexity(seqs)
    return {"perplexity": round(math.exp(total / n_tok), 4), "unigram_perplexity": round(base, 4), "tokens": n_tok,
            "docs": len(texts), "bits_per_byte": round(_bits_per_byte(total, texts), 4),
            "unigram_bits_per_byte": round(_bits_per_byte(n_tok * math.log(base), texts), 4)}


def train(ltm: LongTermMemory, guard: governance.Guard, kind: str = "ngram", log=None, **cfg) -> dict:
    cfg = _check_config(kind, cfg)
    guard.require("lm.train", kind, cfg)
    docs = corpus.approved_texts(ltm, "train", guard.actor)
    if not docs:
        raise ValueError("no approved training document (submit with `corpus add`, approve with `corpus approve`)")
    texts = [t for _, _, t in docs]
    valid = [t for _, _, t in corpus.approved_texts(ltm, "valid", guard.actor)]
    t0 = time.perf_counter()
    tok = BPETokenizer.train(texts, cfg["vocab_size"])
    seqs = [tok.encode(t) for t in texts]
    uni = UnigramLM().fit(seqs, tok.vocab_size, tok.bos_id, tok.eos_id)
    from ..nn import backend
    if kind == "ngram":
        model = NGramLM(cfg["order"]).fit(seqs, tok.vocab_size, tok.bos_id, tok.eos_id)
        train_metrics = {"ngrams": model.num_ngrams}
        evaluate = lambda ts: _ngram_eval(model, uni, [tok.encode(t) for t in ts], ts)  # noqa: E731
        kernel = None
    else:
        from ..nn import TinyGPT, train_lm
        model = TinyGPT(tok.vocab_size, cfg["block_size"], cfg["dim"], cfg["n_layers"], cfg["n_heads"],
                        seed=cfg["seed"])
        if model.n_params() > LIMITS["max_params"]:
            raise ValueError(f"model has {model.n_params()} parameters (limit {LIMITS['max_params']})")
        res = train_lm(model, _stream(tok, texts), cfg["steps"], cfg["batch_size"], cfg["lr"], cfg["seed"],
                       warmup=min(50, cfg["steps"] // 10), log=log)
        train_metrics = {"final_train_loss": round(res["train_loss"][-1], 4), "params": model.n_params(),
                         "backend": backend.backend_name()}
        evaluate = lambda ts: _gpt_eval(model, uni, tok, ts)  # noqa: E731
        kernel = backend.kernel_source_digest()
    metrics = {"train": {**train_metrics, "docs": len(texts), "seconds": round(time.perf_counter() - t0, 2),
                         "vocab_size": tok.vocab_size},
               "valid": evaluate(valid) if valid else None}
    blob = _pack(kind, tok, model)
    digest = hashlib.sha256(blob).hexdigest()
    corpus_list = [[i, h] for i, h, _ in docs]
    provenance = {"pretrained": None, "external_data": False, "trained_by": guard.actor,
                  "corpus": corpus_list, "corpus_digest": hashlib.sha256(json.dumps(corpus_list).encode()).hexdigest(),
                  "tokenizer_digest": tok.digest(), "code": code_digest()["sha256"], "kernel_source": kernel,
                  "seed": cfg["seed"]}
    mid = ltm.db.execute(
        "INSERT INTO lm_models(ts, kind, config, corpus, weights, weights_sha256, metrics, status, approval, "
        "provenance) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (time.time(), kind, json.dumps(cfg), provenance["corpus_digest"], blob, digest, json.dumps(metrics),
         "trained", None, json.dumps(provenance))).lastrowid
    ltm.append_audit("lm.trained", "allow", guard.actor, guard.role, mid, {"weights_sha256": digest, "kind": kind})
    ltm.commit()
    return {"id": mid, "kind": kind, "config": cfg, "weights_sha256": digest, "bytes": len(blob), "metrics": metrics,
            "provenance": provenance}


# --------------------------------------------------------------- registry
def models(ltm: LongTermMemory) -> list:
    rows = ltm.db.execute("SELECT id, ts, kind, config, weights_sha256, metrics, status, provenance, "
                          "LENGTH(weights) AS bytes FROM lm_models ORDER BY id")
    return [{**dict(r), "config": json.loads(r["config"]), "metrics": json.loads(r["metrics"]),
             "provenance": json.loads(r["provenance"])} for r in rows]


def _row(ltm: LongTermMemory, model_id: int):
    row = ltm.db.execute("SELECT * FROM lm_models WHERE id=?", (model_id,)).fetchone()
    if row is None:
        raise KeyError(model_id)
    return row


def approve(ltm: LongTermMemory, seed: bytes, model_id: int, by: str = "", force: bool = False) -> dict:
    """Signs (model id, weights hash) with the admin key and makes it the active model."""
    row = _row(ltm, model_id)
    if hashlib.sha256(row["weights"]).hexdigest() != row["weights_sha256"]:
        raise LMIntegrityError("weights do not match their recorded hash")
    valid = json.loads(row["metrics"]).get("valid")
    if not force:
        if not valid:
            raise PermissionError("no validation measurement (approve validation documents first, or --force)")
        if valid["perplexity"] >= valid["unigram_perplexity"]:
            raise PermissionError("the model does not beat the unigram baseline on validation documents")
    signed = governance.admin_approval(ltm, seed, "lm-approval",
                                       {"model_id": model_id, "weights_sha256": row["weights_sha256"]}, by)
    ltm.db.execute("UPDATE lm_models SET status='retired' WHERE status='active'")
    ltm.db.execute("UPDATE lm_models SET status='active', approval=? WHERE id=?", (json.dumps(signed), model_id))
    ltm.set_meta("active_lm", model_id)
    ltm.commit()
    return {"id": model_id, "status": "active", "weights_sha256": row["weights_sha256"]}


def deactivate(ltm: LongTermMemory, seed: bytes, by: str = "") -> Optional[int]:
    mid = ltm.get_meta("active_lm")
    governance.admin_approval(ltm, seed, "lm-deactivation", {"model_id": mid}, by)
    ltm.db.execute("UPDATE lm_models SET status='retired' WHERE status='active'")
    ltm.set_meta("active_lm", None)
    ltm.commit()
    return mid


def _verify_row(ltm: LongTermMemory, row) -> Optional[str]:
    if hashlib.sha256(row["weights"]).hexdigest() != row["weights_sha256"]:
        return "weights do not match their recorded hash"
    if row["status"] == "active" and not (row["approval"] and governance.check_approval(
            ltm, json.loads(row["approval"]), "lm-approval",
            {"model_id": row["id"], "weights_sha256": row["weights_sha256"]})):
        return "active without a valid admin signature"
    return None


def verify(ltm: LongTermMemory) -> dict:
    problems = {r["id"]: p for r in ltm.db.execute("SELECT * FROM lm_models").fetchall()
                if (p := _verify_row(ltm, r))}
    return {"ok": not problems, "problems": problems}


class LoadedModel:
    def __init__(self, model_id: int, kind: str, tok: BPETokenizer, model, row):
        self.id, self.kind, self.tok, self.model = model_id, kind, tok, model
        self.config = json.loads(row["config"])
        self.provenance = json.loads(row["provenance"])

    def describe(self, lang: str = "en") -> str:
        name = f"n-gram ({self.config['order']})" if self.kind == "ngram" else \
            f"TinyGPT ({self.config['n_layers']}×{self.config['dim']})"
        n = len(self.provenance["corpus"])
        return (f"#{self.id} {name}, " + (f"entraîné ici sur {n} document(s) approuvé(s)" if lang == "fr"
                                          else f"trained here on {n} approved document(s)"))

    def generate(self, prompt: str, max_tokens: int, seed: int = 0, temperature: float = 0.8,
                 top_k: Optional[int] = 40, max_seconds: Optional[float] = None, chunk: int = 16) -> dict:
        """Continues ``prompt``; deterministic for a given seed. Stops at end-of-document, after ``max_tokens``
        or once ``max_seconds`` have elapsed (checked between chunks of ``chunk`` tokens)."""
        rng = random.Random(seed)
        ids = self.tok.encode(prompt)
        context = ids if self.kind == "ngram" else [self.tok.bos_id] + ids
        eos, new, stopped = self.tok.eos_id, [], "max_tokens"
        deadline = None if max_seconds is None else time.monotonic() + max_seconds
        while len(new) < max_tokens:
            n = min(chunk, max_tokens - len(new))
            out = self.model.generate(context + new, n, rng, temperature, top_k, stop_ids=[eos])
            ended = eos in out or len(out) < n
            new += [i for i in out if i != eos]
            if ended:
                stopped = "end"
                break
            if deadline is not None and time.monotonic() > deadline:
                stopped = "time budget"
                break
        return {"text": self.tok.decode(new), "tokens": len(new), "prompt_tokens": len(ids), "stopped": stopped}


def load_active(ltm: LongTermMemory) -> Optional[LoadedModel]:
    mid = ltm.get_meta("active_lm")
    if mid is None:
        return None
    row = _row(ltm, mid)
    problem = _verify_row(ltm, row)
    if problem or row["status"] != "active":
        raise LMIntegrityError(f"model #{mid}: {problem or 'not active'}")
    key = row["weights_sha256"]
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = LoadedModel(mid, *_unpack(row["weights"]), row)
    return _CACHE[key]


def evaluate(ltm: LongTermMemory, model_id: int, reader: str = "admin") -> dict:
    """Final measurement on the sealed test split (audited)."""
    row = _row(ltm, model_id)
    if _verify_row(ltm, row):
        raise LMIntegrityError(_verify_row(ltm, row))
    kind, tok, model = _unpack(row["weights"])
    texts = [t for _, _, t in corpus.approved_texts(ltm, "test", reader)]
    if not texts:
        raise ValueError("no approved test document")
    train_texts = [t for _, _, t in corpus.approved_texts(ltm, "train", reader)]
    uni = UnigramLM().fit([tok.encode(t) for t in train_texts], tok.vocab_size, tok.bos_id, tok.eos_id)
    res = _ngram_eval(model, uni, [tok.encode(t) for t in texts], texts) if kind == "ngram" else \
        _gpt_eval(model, uni, tok, texts)
    metrics = json.loads(row["metrics"])
    metrics["test"] = res
    ltm.db.execute("UPDATE lm_models SET metrics=? WHERE id=?", (json.dumps(metrics), model_id))
    ltm.commit()
    return res
