"""Phase-5 world: the CausalWorld physics, seen only through rendered pixels.

The hidden rules are those of :class:`CausalWorld` (same seed, same sampling), but
the agent never receives a property value. Each property is drawn as a
grey-scale panel in which the value is the number of "token" glyphs: a plus
sign per unit for positive values, a hollow ring per unit for negative ones.
Every panel also holds 0-3 distractor glyphs (diagonal crosses of the same ink),
random positions, jittered placement, random intensity and Gaussian pixel noise,
so the value is not readable from the total ink.

What the agent's "body" knows (``spec``): the image size, which rectangle of the
retina shows which property (the sensor layout), the action tokens and the
range of values its motors can set. Nothing about glyphs, counting or the code
from appearance to number is exposed.

Agent methods:

* ``look(controls)`` configures the object with the motor command ``controls``
  and returns its image (motor babbling: one interaction each).
* ``passive_observation()`` returns ``{"before", "action", "after"}`` images of a
  natural (confounded) state, as in :class:`CausalWorld`.
* ``experiment(controls, action)`` returns ``{"before", "after"}`` images.

Rendering uses its own random stream, so the physics stream (passive states,
outcome noise) is exactly the symbolic world's for the same seed.
"""

from __future__ import annotations

import random

from .causal_world import CausalWorld

SLOTS = 8  # 8 x 8 glyph slots per panel
SLOT = 6  # pixels per slot side
PANEL = SLOTS * SLOT  # 48 x 48 pixels per property
PLUS = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))
CROSS = ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))
RING = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2))
MAX_DISTRACTORS = 3


def render_panel(value: int, rng: random.Random, noise: float) -> list:
    """Row-major intensities in [0, 1] (before pixel noise and 8-bit quantisation)."""
    img = [0.0] * (PANEL * PANEL)
    n_tokens = abs(value)
    n_distractors = rng.randint(0, MAX_DISTRACTORS)
    slots = rng.sample(range(SLOTS * SLOTS), n_tokens + n_distractors)
    token = PLUS if value >= 0 else RING
    for i, s in enumerate(slots):
        shape = token if i < n_tokens else CROSS
        oy = (s // SLOTS) * SLOT + 1 + rng.randint(0, 1)
        ox = (s % SLOTS) * SLOT + 1 + rng.randint(0, 1)
        ink = rng.uniform(0.7, 1.0)
        for dy, dx in shape:
            img[(oy + dy) * PANEL + ox + dx] = ink
    return [p + rng.gauss(0.0, noise) for p in img]


class PixelCausalWorld:
    AGENT_METHODS = ("spec", "look", "passive_observation", "experiment")

    def __init__(self, seed: int, pixel_noise: float = 0.08, **world_options):
        self._world = CausalWorld(seed, **world_options)
        self.env_id = f"pixel-causal-world-{seed}"
        self.pixel_noise = pixel_noise
        self._render_rng = random.Random(f"pixel-render-{seed}")
        if self._world.n_values + MAX_DISTRACTORS > SLOTS * SLOTS:
            raise ValueError("too many values to draw")

    # ----------------------------------------------------------- rendering
    def _image(self, state: dict, rng: random.Random) -> dict:
        width = PANEL * len(self.props)
        rows = [bytearray(width) for _ in range(PANEL)]
        for k, p in enumerate(self.props):
            panel = render_panel(state[p], rng, self.pixel_noise)
            for y in range(PANEL):
                rows[y][k * PANEL:(k + 1) * PANEL] = bytes(
                    max(0, min(255, int(round(v * 255)))) for v in panel[y * PANEL:(y + 1) * PANEL])
        return {"height": PANEL, "width": width, "pixels": b"".join(bytes(r) for r in rows)}

    # ----------------------------------------------------------- agent-visible
    def spec(self) -> dict:
        s = self._world.spec()
        s["image"] = {"height": PANEL, "width": PANEL * len(self.props), "format": "grey8",
                      "panels": {p: [k * PANEL, 0, PANEL, PANEL] for k, p in enumerate(self.props)}}
        return s

    def look(self, controls: dict) -> dict:
        return self._image(self._world._validate(controls), self._render_rng)

    def passive_observation(self) -> dict:
        s, a, nxt = self._world.passive_observation()
        return {"before": self._image(s, self._render_rng), "action": a, "after": self._image(nxt, self._render_rng)}

    def experiment(self, controls: dict, action: str) -> dict:
        s = self._world._validate(controls)
        nxt = self._world.experiment(s, action)
        return {"before": self._image(s, self._render_rng), "after": self._image(nxt, self._render_rng)}

    # ------------------------------------------------------------ harness-only
    @property
    def props(self) -> list:
        return self._world.props

    @property
    def actions(self) -> list:
        return self._world.actions

    @property
    def n_values(self) -> int:
        return self._world.n_values

    @property
    def _rules(self) -> dict:
        return self._world._rules

    @property
    def _confound(self) -> tuple:
        return self._world._confound

    def _true_next(self, state: dict, action: str) -> dict:
        return self._world._true_next(state, action)

    def _describe_rules(self) -> dict:
        return self._world._describe_rules()

    def _render(self, state: dict, rng: random.Random) -> dict:
        """Evaluation renders with the caller's stream, leaving the agent-visible streams untouched."""
        return self._image(self._world._validate(state), rng)
