"""Byte-level BPE tokenizer, trained here from local text only (no pretrained vocabulary).

Byte-pair encoding (Sennrich, Haddow & Birch 2016, "Neural Machine Translation of Rare Words with Subword
Units") starts from an alphabet of symbols and repeatedly merges the most frequent adjacent pair into a new
symbol. Working on UTF-8 **bytes** (as in GPT-2) makes the base alphabet exactly 256 symbols, so any string
can be encoded with no unknown token and ``decode(encode(s)) == s``.

* **Pre-tokenization** (self-written, GPT-2-like): text is split by ``PRETOKEN_PATTERN`` into runs of letters
  with an optional leading space, digit groups of at most 3, punctuation/underscore runs and whitespace.
  Pieces longer than ``MAX_PIECE_CHARS`` are chunked, which bounds encoding cost on adversarial input. Merges
  never cross piece boundaries.
* **Vocabulary**: ids ``0..255`` are bytes, then the special tokens, then one id per learned merge.
* **Training** counts piece frequencies, then merges the most frequent pair until ``vocab_size`` is reached or
  the best pair occurs fewer than ``min_frequency`` times. Pair counts are updated incrementally using an index
  pair -> words containing it, and a lazy max-heap. Ties are broken by the pair's byte strings, then ids, so
  training never depends on hash ordering (``PYTHONHASHSEED``).
* **Encoding** applies merges inside each piece, lowest rank first. Special-token strings found in the input
  are encoded as ordinary bytes, never as special ids (prompt-injection safety); special ids only come from
  ``add_bos`` / ``add_eos``.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections import Counter
from typing import Iterable, Sequence

PRETOKEN_PATTERN = r" ?[^\W\d_]+| ?\d{1,3}| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"
MAX_PIECE_CHARS = 64
DEFAULT_SPECIAL_TOKENS = ("<|bos|>", "<|eos|>", "<|pad|>")
FORMAT = "genesis-bpe"
VERSION = 1
_PRETOKEN_RE = re.compile(PRETOKEN_PATTERN)
_CACHE_SIZE = 65536


def pretokenize(text: str) -> list[str]:
    """Split ``text`` into pieces whose concatenation is exactly ``text``."""
    out: list[str] = []
    for piece in _PRETOKEN_RE.findall(text):
        if len(piece) <= MAX_PIECE_CHARS:
            out.append(piece)
        else:
            out.extend(piece[i:i + MAX_PIECE_CHARS] for i in range(0, len(piece), MAX_PIECE_CHARS))
    return out


def _utf8(s: str) -> bytes:
    # surrogatepass: lone surrogates (not valid Unicode text) still encode instead of raising.
    return s.encode("utf-8", "surrogatepass")


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _merge_pair(sym: Sequence[int], a: int, b: int, new_id: int) -> list[int]:
    out: list[int] = []
    i, n = 0, len(sym)
    while i < n:
        if i + 1 < n and sym[i] == a and sym[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(sym[i])
            i += 1
    return out


def _check_specials(special_tokens) -> tuple[str, ...]:
    if not isinstance(special_tokens, (list, tuple)):
        raise ValueError("special_tokens must be a list or tuple of strings")
    specials = tuple(special_tokens)
    for s in specials:
        if not isinstance(s, str) or not s:
            raise ValueError("special tokens must be non-empty strings")
    if len(set(specials)) != len(specials):
        raise ValueError("special tokens must be unique")
    return specials


class BPETokenizer:
    """Byte-level BPE tokenizer. Build it with :meth:`train` or :meth:`from_dict`."""

    def __init__(self, merges: Iterable[Sequence[int]] = (),
                 special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS):
        self._specials = _check_specials(special_tokens)
        self._first_merge = 256 + len(self._specials)
        tok_bytes = [bytes([i]) for i in range(256)] + [_utf8(s) for s in self._specials]
        ranks: dict[tuple[int, int], int] = {}
        if isinstance(merges, (str, bytes, dict)):
            raise ValueError("merges must be a list of [a, b] id pairs")
        for rank, m in enumerate(merges):
            if not isinstance(m, (list, tuple)) or len(m) != 2:
                raise ValueError(f"merge #{rank} is not an [a, b] pair")
            a, b = m
            limit = self._first_merge + rank
            for x in (a, b):
                if not _is_int(x) or x < 0 or x >= limit or 256 <= x < self._first_merge:
                    raise ValueError(f"merge #{rank} has an invalid id: {x!r}")
            if (a, b) in ranks:
                raise ValueError(f"merge #{rank} duplicates an earlier merge: {[a, b]}")
            ranks[(a, b)] = rank
            tok_bytes.append(tok_bytes[a] + tok_bytes[b])
        self._ranks = ranks
        self._merges = list(ranks)
        self._tok_bytes = tok_bytes
        self._special_ids = {s: 256 + i for i, s in enumerate(self._specials)}
        self._cache: dict[str, tuple[int, ...]] = {}  # piece -> ids; cleared when full

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int, min_frequency: int = 2,
              special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS) -> "BPETokenizer":
        specials = _check_specials(special_tokens)
        first_merge = 256 + len(specials)
        if not _is_int(vocab_size) or vocab_size < first_merge:
            raise ValueError(f"vocab_size must be an int >= {first_merge} (256 bytes + {len(specials)} specials)")
        if not _is_int(min_frequency) or min_frequency < 1:
            raise ValueError("min_frequency must be an int >= 1")
        if isinstance(texts, str):
            texts = (texts,)

        piece_counts: Counter[str] = Counter()
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("texts must be an iterable of str")
            piece_counts.update(pretokenize(text))
        words: list[list[int]] = []
        freqs: list[int] = []
        for raw, f in sorted((_utf8(p), f) for p, f in piece_counts.items()):
            if len(raw) > 1:
                words.append(list(raw))
                freqs.append(f)
        del piece_counts

        pair_counts: dict[tuple[int, int], int] = {}
        where: dict[tuple[int, int], set[int]] = {}
        for wi, sym in enumerate(words):
            f = freqs[wi]
            for p in zip(sym, sym[1:]):
                pair_counts[p] = pair_counts.get(p, 0) + f
                s = where.get(p)
                if s is None:
                    where[p] = s = set()
                s.add(wi)

        tb = [bytes([i]) for i in range(256)] + [_utf8(s) for s in specials]
        # Max-heap by count; ties -> smallest (bytes(a), bytes(b), a, b). Entries go stale when counts change.
        heap = [(-c, tb[a], tb[b], a, b) for (a, b), c in pair_counts.items()]
        heapq.heapify(heap)
        merges: list[tuple[int, int]] = []
        while len(tb) < vocab_size and heap:
            negc, _, _, a, b = heapq.heappop(heap)
            pair = (a, b)
            count = pair_counts.get(pair, 0)
            if count != -negc:
                continue
            if count < min_frequency:
                break
            new_id = len(tb)
            tb.append(tb[a] + tb[b])
            merges.append(pair)
            delta: dict[tuple[int, int], int] = {}
            for wi in where.pop(pair):
                sym = words[wi]
                new = _merge_pair(sym, a, b, new_id)
                f = freqs[wi]
                old_pairs = set(zip(sym, sym[1:]))
                new_pairs = set(zip(new, new[1:]))
                for p in zip(sym, sym[1:]):
                    delta[p] = delta.get(p, 0) - f
                for p in zip(new, new[1:]):
                    delta[p] = delta.get(p, 0) + f
                for p in old_pairs - new_pairs:
                    s = where.get(p)
                    if s is not None:
                        s.discard(wi)
                for p in new_pairs - old_pairs:
                    s = where.get(p)
                    if s is None:
                        where[p] = s = set()
                    s.add(wi)
                words[wi] = new
            for p, d in delta.items():
                if d == 0:
                    continue
                c = pair_counts.get(p, 0) + d
                if c > 0:
                    pair_counts[p] = c
                    heapq.heappush(heap, (-c, tb[p[0]], tb[p[1]], p[0], p[1]))
                else:
                    pair_counts.pop(p, None)
                    where.pop(p, None)
        return cls(merges, specials)

    # ------------------------------------------------------------------ encoding
    def _bpe(self, piece: str) -> tuple[int, ...]:
        sym = list(_utf8(piece))
        ranks = self._ranks
        while len(sym) > 1:
            best = None
            best_rank = -1
            for p in zip(sym, sym[1:]):
                r = ranks.get(p)
                if r is not None and (best is None or r < best_rank):
                    best, best_rank = p, r
            if best is None:
                break
            sym = _merge_pair(sym, best[0], best[1], self._first_merge + best_rank)
        return tuple(sym)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a str")
        ids: list[int] = []
        if add_bos:
            ids.append(self._require_special("<|bos|>"))
        cache = self._cache
        for piece in pretokenize(text):
            piece_ids = cache.get(piece)
            if piece_ids is None:
                piece_ids = self._bpe(piece)
                if len(cache) >= _CACHE_SIZE:
                    cache.clear()
                cache[piece] = piece_ids
            ids.extend(piece_ids)
        if add_eos:
            ids.append(self._require_special("<|eos|>"))
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """Concatenate token bytes and decode as UTF-8 (invalid sequences become U+FFFD)."""
        parts: list[bytes] = []
        n = len(self._tok_bytes)
        for i in ids:
            if not _is_int(i) or i < 0 or i >= n:
                raise ValueError(f"invalid token id: {i!r}")
            if skip_special and 256 <= i < self._first_merge:
                continue
            parts.append(self._tok_bytes[i])
        return b"".join(parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ vocabulary
    def _check_id(self, token_id: int) -> int:
        if not _is_int(token_id) or token_id < 0 or token_id >= len(self._tok_bytes):
            raise ValueError(f"invalid token id: {token_id!r}")
        return token_id

    def token_bytes(self, token_id: int) -> bytes:
        return self._tok_bytes[self._check_id(token_id)]

    def token_str(self, token_id: int) -> str:
        """Readable form of one token; bytes that are not complete UTF-8 are shown as ``\\xNN``."""
        return self._tok_bytes[self._check_id(token_id)].decode("utf-8", errors="backslashreplace")

    def is_special(self, token_id: int) -> bool:
        return 256 <= self._check_id(token_id) < self._first_merge

    def special_id(self, token: str) -> int | None:
        return self._special_ids.get(token)

    def _require_special(self, token: str) -> int:
        i = self._special_ids.get(token)
        if i is None:
            raise ValueError(f"this tokenizer has no {token} special token")
        return i

    @property
    def vocab_size(self) -> int:
        return len(self._tok_bytes)

    @property
    def special_tokens(self) -> tuple[str, ...]:
        return self._specials

    @property
    def merges(self) -> list[tuple[int, int]]:
        return list(self._merges)

    @property
    def bos_id(self) -> int | None:
        return self._special_ids.get("<|bos|>")

    @property
    def eos_id(self) -> int | None:
        return self._special_ids.get("<|eos|>")

    @property
    def pad_id(self) -> int | None:
        return self._special_ids.get("<|pad|>")

    # ------------------------------------------------------------------ serialization
    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "version": VERSION,
            "pattern": PRETOKEN_PATTERN,
            "max_piece_chars": MAX_PIECE_CHARS,
            "specials": list(self._specials),
            "merges": [[a, b] for a, b in self._merges],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BPETokenizer":
        if not isinstance(d, dict):
            raise ValueError("tokenizer data must be a dict")
        if d.get("format") != FORMAT or d.get("version") != VERSION:
            raise ValueError(f"unsupported tokenizer format/version (expected {FORMAT} v{VERSION})")
        if d.get("pattern") != PRETOKEN_PATTERN or d.get("max_piece_chars") != MAX_PIECE_CHARS:
            raise ValueError("tokenizer was saved with a different pre-tokenization")
        specials, merges = d.get("specials"), d.get("merges")
        if not isinstance(specials, list) or not isinstance(merges, list):
            raise ValueError("'specials' and 'merges' must be lists")
        return cls(merges, specials)

    def digest(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(blob.encode("ascii")).hexdigest()

    def __repr__(self) -> str:
        return (f"BPETokenizer(vocab_size={self.vocab_size}, merges={len(self._merges)}, "
                f"specials={list(self._specials)})")
