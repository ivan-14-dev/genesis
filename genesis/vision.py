"""Phase-5 neural perception: read quantities from pixels without being told how.

The body only says which rectangle of the retina shows which object property
(sensor layout). How appearance maps to a number is learned in two steps:

1. **Self-supervised visual code.** A small vector-quantised autoencoder is
   trained on unlabelled 3x3 pixel patches (``genesis.nn`` autograd, straight-
   through estimator, codebook refreshed by Lloyd steps). Every patch of a panel
   is mapped to one of :data:`CODES` discrete codes; the encoder is shared across
   positions, so a panel becomes a translation-invariant histogram of codes.
2. **Readout calibrated by efference copies.** The agent sets an object's
   properties with its own motors (``look``) and pairs each panel with the value
   it commanded: no external labels. A ridge-regularised linear map from code
   histogram to value is fitted from sufficient statistics, so later motor
   commands (the "before" image of every experiment) keep refining it.

Reading abstains (returns ``None``) when a patch falls in a code never seen
during calibration, when it is much farther from its code's prototype than any
calibrated patch ("unfamiliar appearance"), or when the linear reading is not
close to an integer. A wrong reading would be learned as a fact; an abstention
only costs an observation.

``kind="kmeans"`` replaces the neural encoder by k-means on raw pixels (the
ablation reported in the benchmark). Nothing here is pretrained: every weight is
fitted from the agent's own observations of one environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from array import array

from .nn import backend
from .nn.optim import AdamW
from .nn.tensor import Tensor, add, matmul, mean, mul, scale, tanh

K = 3  # patch side
CODES = 32
LATENT = 8
TRAIN_STEPS = 300
BATCH = 256
LLOYD_EVERY = 25
SAMPLE = 6000
COMMITMENT = 0.25
LEARNING_RATE = 1e-2
NOVELTY_SCALE = 2.0
NOVELTY_MARGIN = 0.25
AMBIGUITY = 0.25
RIDGE = 1e-3
KINDS = ("neural", "kmeans")


def manifest() -> dict:
    body = {
        "version": 1,
        "mechanisms": ["self-supervised vector-quantised patch autoencoder (straight-through, Lloyd codebook)",
                       "position-shared discrete codes pooled into per-panel histograms",
                       "ridge linear readout calibrated by efference copies of motor commands",
                       "abstention on uncalibrated codes, unfamiliar appearance and ambiguous readings"],
        "limits": [K, CODES, LATENT, TRAIN_STEPS, BATCH, NOVELTY_SCALE, NOVELTY_MARGIN, AMBIGUITY, RIDGE],
        "domain_knowledge": [], "pretrained_models": [],
    }
    return {**body, "digest": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()}


def register(ltm, session) -> None:
    """Versioned like the grounding faculty: an upgrade is allowed, never silent."""
    current, old = manifest(), ltm.get_meta("innate_vision")
    if old is None or old["digest"] != current["digest"]:
        history = ltm.get_meta("innate_vision_history") or []
        history.append({"session": session, "ts": time.time(), "from": old and old["digest"],
                        "to": current["digest"]})
        ltm.set_meta("innate_vision_history", history)
        ltm.set_meta("innate_vision", current)


def image_digest(image: dict) -> str:
    return hashlib.sha256(image["pixels"]).hexdigest()


def panel_patches(image: dict, rect) -> tuple:
    """All stride-1 KxK patches of one panel, intensities in [0, 1], row-major."""
    x0, y0, w, h = rect
    width, px = image["width"], image["pixels"]
    rows = [[b / 255.0 for b in px[(y0 + y) * width + x0:(y0 + y) * width + x0 + w]] for y in range(h)]
    out = array("d")
    for y in range(h - K + 1):
        window = rows[y:y + K]
        for x in range(w - K + 1):
            for r in window:
                out.extend(r[x:x + K])
    return out, (h - K + 1) * (w - K + 1)


def _solve(a: list, b: list) -> list:
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        piv = m[c][c]
        if abs(piv) < 1e-12:
            continue
        for r in range(n):
            if r != c and m[r][c]:
                f = m[r][c] / piv
                row_c, row_r = m[c], m[r]
                for j in range(c, n + 1):
                    row_r[j] -= f * row_c[j]
    return [m[i][n] / m[i][i] if abs(m[i][i]) >= 1e-12 else 0.0 for i in range(n)]


class VisualCode:
    """Patch encoder plus codebook (the discrete visual vocabulary)."""

    def __init__(self, kind: str = "neural"):
        if kind not in KINDS:
            raise ValueError(f"unknown visual code {kind!r}")
        self.kind = kind
        self.dim = LATENT if kind == "neural" else K * K
        self.w_enc = self.b_enc = self.w_dec = self.b_dec = None
        self.codebook = array("d")
        self.losses: list = []

    # ------------------------------------------------------------- inference
    def encode(self, x: array, n: int) -> array:
        if self.kind != "neural":
            return x
        z = backend.matmul(x, self.w_enc, n, K * K, LATENT)
        z = backend.ops().add_rows(z, self.b_enc, n, LATENT)
        return array("d", map(math.tanh, z))

    def assign(self, z: array, n: int) -> list:
        d, cb = self.dim, self.codebook
        norms = [sum(cb[c * d + j] ** 2 for j in range(d)) for c in range(CODES)]
        dots = backend.matmul(z, cb, n, d, CODES, trans_b=True)
        out = []
        for i in range(n):
            row = [norms[c] - 2.0 * dots[i * CODES + c] for c in range(CODES)]
            out.append(row.index(min(row)))
        return out

    def codes(self, x: array, n: int) -> list:
        return self.assign(self.encode(x, n), n)

    # -------------------------------------------------------------- training
    def _seed_codebook(self, z: array, n: int, rng: random.Random) -> None:
        """k-means++ initialisation."""
        d = self.dim
        first = rng.randrange(n)
        cb = array("d", z[first * d:(first + 1) * d])
        dist = [sum((z[t * d + j] - cb[j]) ** 2 for j in range(d)) for t in range(n)]
        for _ in range(CODES - 1):
            total = sum(dist)
            pick = n - 1
            if total > 0:
                r, acc = rng.random() * total, 0.0
                for i, v in enumerate(dist):
                    acc += v
                    if acc >= r:
                        pick = i
                        break
            row = z[pick * d:(pick + 1) * d]
            cb.extend(row)
            dist = [min(dist[t], sum((z[t * d + j] - row[j]) ** 2 for j in range(d))) for t in range(n)]
        self.codebook = cb

    def _lloyd(self, z: array, n: int, iters: int) -> None:
        d = self.dim
        for _ in range(iters):
            sums, counts = [[0.0] * d for _ in range(CODES)], [0] * CODES
            for i, c in enumerate(self.assign(z, n)):
                counts[c] += 1
                s = sums[c]
                for j in range(d):
                    s[j] += z[i * d + j]
            for c in range(CODES):
                if counts[c]:
                    for j in range(d):
                        self.codebook[c * d + j] = sums[c][j] / counts[c]

    def fit(self, patches: array, n: int, rng: random.Random) -> None:
        """Self-supervised: only pixel patches, no values."""
        p = K * K

        def sample(m: int) -> array:
            out = array("d")
            for i in (rng.randrange(n) for _ in range(m)):
                out.extend(patches[i * p:(i + 1) * p])
            return out

        if self.kind == "kmeans":
            x = sample(SAMPLE)
            self._seed_codebook(x, SAMPLE, rng)
            self._lloyd(x, SAMPLE, 15)
            return
        w_enc = Tensor(array("d", [rng.gauss(0.0, 1.0 / 3.0) for _ in range(p * LATENT)]), (p, LATENT), True)
        b_enc = Tensor(array("d", [0.0] * LATENT), (LATENT,), True)
        w_dec = Tensor(array("d", [rng.gauss(0.0, 1.0 / math.sqrt(LATENT)) for _ in range(LATENT * p)]),
                       (LATENT, p), True)
        b_dec = Tensor(array("d", [0.0] * p), (p,), True)
        self.w_enc, self.b_enc = w_enc.data, b_enc.data
        opt = AdamW([w_enc, b_enc, w_dec, b_dec], lr=LEARNING_RATE, weight_decay=0.0)
        x = sample(SAMPLE // 2)
        self._seed_codebook(self.encode(x, SAMPLE // 2), SAMPLE // 2, rng)
        for step in range(TRAIN_STEPS):
            xb = sample(BATCH)
            z = tanh(add(matmul(Tensor(xb, (BATCH, p)), w_enc), b_enc))
            q = array("d")
            d = LATENT
            for c in self.assign(z.data, BATCH):
                q.extend(self.codebook[c * d:(c + 1) * d])
            # straight-through: the decoder sees the code, the gradient flows to the encoder output
            st = add(z, Tensor(array("d", [a - b for a, b in zip(q, z.data)]), (BATCH, d)))
            diff = add(add(matmul(st, w_dec), b_dec), Tensor(array("d", [-v for v in xb]), (BATCH, p)))
            commit = add(z, Tensor(array("d", [-v for v in q]), (BATCH, d)))
            loss = add(mean(mul(diff, diff)), scale(mean(mul(commit, commit)), COMMITMENT))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (step + 1) % LLOYD_EVERY == 0:
                self.losses.append(round(loss.item(), 5))
                x = sample(SAMPLE // 2)
                self._lloyd(self.encode(x, SAMPLE // 2), SAMPLE // 2, 2)
        self.w_dec, self.b_dec = w_dec.data, b_dec.data
        x = sample(SAMPLE)
        self._lloyd(self.encode(x, SAMPLE), SAMPLE, 5)

    def to_json(self) -> dict:
        return {"kind": self.kind, "codebook": list(self.codebook), "losses": self.losses,
                **({"w_enc": list(self.w_enc), "b_enc": list(self.b_enc), "w_dec": list(self.w_dec),
                    "b_dec": list(self.b_dec)} if self.kind == "neural" else {})}

    @classmethod
    def from_json(cls, d: dict) -> "VisualCode":
        code = cls(d["kind"])
        code.codebook = array("d", d["codebook"])
        code.losses = list(d.get("losses", []))
        if code.kind == "neural":
            code.w_enc, code.b_enc = array("d", d["w_enc"]), array("d", d["b_enc"])
            code.w_dec, code.b_dec = array("d", d["w_dec"]), array("d", d["b_dec"])
        return code


class Glance:
    """What one panel looks like to the agent: codes, their histogram and familiarity."""

    __slots__ = ("hist", "codes", "errors")

    def __init__(self, hist, codes, errors):
        self.hist, self.codes, self.errors = hist, codes, errors


class VisualReader:
    """Per-environment reader: a visual code plus a readout calibrated by efference copies."""

    def __init__(self, code: VisualCode):
        self.code = code
        self.prototypes = [[0.0] * (K * K) for _ in range(CODES)]
        self.max_error = [0.0] * CODES
        self.calibrated = [0] * CODES
        n = CODES + 1
        self.gram = [[0.0] * n for _ in range(n)]
        self.rhs = [0.0] * n
        self.n_panels = 0
        self.weights: list | None = None

    @classmethod
    def fit(cls, panels: list, kind: str, rng: random.Random) -> "VisualReader":
        """``panels``: unlabelled ``(patches, n)`` pairs used to learn the visual code."""
        allp = array("d")
        for x, _ in panels:
            allp.extend(x)
        code = VisualCode(kind)
        code.fit(allp, len(allp) // (K * K), rng)
        reader = cls(code)
        sums, counts = [[0.0] * (K * K) for _ in range(CODES)], [0] * CODES
        for x, n in panels:
            for i, c in enumerate(code.codes(x, n)):
                counts[c] += 1
                s = sums[c]
                for j in range(K * K):
                    s[j] += x[i * K * K + j]
        reader.prototypes = [[v / counts[c] for v in sums[c]] if counts[c] else [0.0] * (K * K)
                             for c in range(CODES)]
        return reader

    def glance(self, patches: array, n: int) -> Glance:
        codes = self.code.codes(patches, n)
        hist = [0] * CODES
        errors = []
        p = K * K
        for i, c in enumerate(codes):
            hist[c] += 1
            proto = self.prototypes[c]
            errors.append(sum((patches[i * p + j] - proto[j]) ** 2 for j in range(p)))
        return Glance(hist, codes, errors)

    def calibrate(self, g: Glance, value: int) -> None:
        """Pair a glance with the value the agent's own motors set (efference copy)."""
        for c, e in zip(g.codes, g.errors):
            self.calibrated[c] += 1
            if e > self.max_error[c]:
                self.max_error[c] = e
        f = g.hist + [1]
        for i, fi in enumerate(f):
            if fi:
                row = self.gram[i]
                for j, fj in enumerate(f):
                    row[j] += fi * fj
                self.rhs[i] += fi * value
        self.n_panels += 1
        self.weights = None

    def _readout(self) -> list:
        if self.weights is None:
            a = [[v + (RIDGE if i == j and i < CODES else 0.0) for j, v in enumerate(row)]
                 for i, row in enumerate(self.gram)]
            self.weights = _solve(a, self.rhs)
        return self.weights

    def read(self, g: Glance) -> tuple:
        """``(value or None, detail)``; never guesses on what it has not been calibrated for."""
        if not self.n_panels:
            return None, {"abstain": "uncalibrated"}
        for c, e in zip(g.codes, g.errors):
            if not self.calibrated[c]:
                return None, {"abstain": "unfamiliar-code"}
            if e > NOVELTY_SCALE * self.max_error[c] + NOVELTY_MARGIN:
                return None, {"abstain": "unfamiliar-appearance"}
        w = self._readout()
        phi = sum(wi * hi for wi, hi in zip(w, g.hist + [1]))
        v = round(phi)
        if abs(phi - v) > AMBIGUITY:
            return None, {"abstain": "ambiguous", "reading": round(phi, 3)}
        return int(v), {"reading": round(phi, 3)}

    def to_json(self) -> dict:
        return {"code": self.code.to_json(), "prototypes": self.prototypes, "max_error": self.max_error,
                "calibrated": self.calibrated, "gram": self.gram, "rhs": self.rhs, "n_panels": self.n_panels}

    @classmethod
    def from_json(cls, d: dict) -> "VisualReader":
        r = cls(VisualCode.from_json(d["code"]))
        r.prototypes, r.max_error, r.calibrated = d["prototypes"], d["max_error"], d["calibrated"]
        r.gram, r.rhs, r.n_panels = d["gram"], d["rhs"], d["n_panels"]
        return r
