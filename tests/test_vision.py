import random

import pytest

from genesis.adapters import PixelLabAdapter, PixelPassiveAdapter
from genesis.agent import Genesis
from genesis.environments import CausalWorld, PixelCausalWorld, sandboxed
from genesis.experiments import experienced, lab_accuracy, lab_eval_set


@pytest.fixture(scope="module")
def pixel_lab(tmp_path_factory):
    db = str(tmp_path_factory.mktemp("pixels") / "m.db")
    world = PixelCausalWorld(1)
    sb = sandboxed(world, 5000)
    agent = Genesis(db, seed=1)
    calibration = agent.calibrate_vision(sb, 10)
    watcher = PixelPassiveAdapter(agent, sb)
    agent.live(watcher, 60, stop_when_idle=False)
    lab = PixelLabAdapter(agent, sb)
    agent.live(lab, 40, stop_when_idle=False)
    yield {"agent": agent, "world": world, "db": db, "calibration": calibration, "watcher": watcher, "lab": lab}
    agent.close()


def test_pixel_world_shows_images_only_and_keeps_the_physics_stream():
    pixel, symbolic = PixelCausalWorld(3), CausalWorld(3)
    pixel.look({p: 2 for p in pixel.props})  # rendering must not consume the physics stream
    for _ in range(5):
        obs = pixel.passive_observation()
        _, action, _ = symbolic.passive_observation()
        assert obs["action"] == action
        assert set(obs["before"]) == {"height", "width", "pixels"}
        assert len(obs["before"]["pixels"]) == obs["before"]["height"] * obs["before"]["width"]
    assert PixelCausalWorld(3).look({p: 4 for p in pixel.props}) == PixelCausalWorld(3).look(
        {p: 4 for p in pixel.props})


def test_reading_extrapolates_beyond_calibration_and_abstains_on_unseen_appearance(pixel_lab):
    agent, world = pixel_lab["agent"], pixel_lab["world"]
    panels = world.spec()["image"]["panels"]
    rng = random.Random("test-perception")
    correct = wrong = 0
    for _ in range(4):
        state = {p: rng.randint(8, 30) for p in world.props}  # calibration only ever showed 0..5
        seen = agent.read_image(world.env_id, world._render(state, rng), panels)
        correct += sum(seen[p] == state[p] for p in world.props)
        wrong += sum(seen[p] is not None and seen[p] != state[p] for p in world.props)
    assert wrong == 0 and correct >= 12
    for v in (-1, -3):
        seen = agent.read_image(world.env_id, world._render({p: v for p in world.props}, rng), panels)
        assert all(x is None for x in seen.values())  # rings were never seen: no guess


def test_rules_are_discovered_from_pixels_and_abstentions_teach_nothing(pixel_lab):
    agent, world, watcher, lab = pixel_lab["agent"], pixel_lab["world"], pixel_lab["watcher"], pixel_lab["lab"]
    held = lab_eval_set(world, 300, 991, 0, world.n_values - 1)
    assert lab_accuracy(agent, world, held, experienced(agent, world)) == 1.0
    checks = lab.perception_checks
    assert checks["agreed"] + checks["abstained"] == checks["read"] == 4 * 40
    readable = 60 - watcher.abstentions["unreadable_situations"] + 40
    unread = watcher.abstentions["unreadable_outcomes"] + lab.abstentions["unreadable_outcomes"]
    data = sum(len(lr.data) for k, lr in agent.learners.items() if k.startswith(world.env_id + "/"))
    assert data == 4 * readable - unread


def test_vision_is_born_empty_and_persists(pixel_lab, tmp_path):
    newborn = Genesis(str(tmp_path / "n.db"))
    assert newborn.ltm.get_meta("stage0_audit")["checks"]["vision_has_no_pretrained_weights"]
    assert newborn.ltm.get_meta("innate_vision")["pretrained_models"] == []
    assert newborn.visual_reader(pixel_lab["world"].env_id) is None
    with pytest.raises(RuntimeError, match="calibrate_vision"):
        PixelLabAdapter(newborn, sandboxed(PixelCausalWorld(1), 10))
    newborn.close()
    agent, world = pixel_lab["agent"], pixel_lab["world"]
    assert pixel_lab["calibration"]["looks"] == 10
    agent.consolidate()
    again = Genesis(pixel_lab["db"] + ".copy", seed=1)
    again.ltm.set_meta(f"vision/{world.env_id}", agent.ltm.get_meta(f"vision/{world.env_id}"))
    image = world._render({p: 3 for p in world.props}, random.Random("persist"))
    panels = world.spec()["image"]["panels"]
    assert again.read_image(world.env_id, image, panels) == agent.read_image(world.env_id, image, panels)
    again.close()
