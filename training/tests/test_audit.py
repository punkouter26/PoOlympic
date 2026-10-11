"""2026-10-11 audit additions (all opt-in): action log + report, self-contact penalty, shove-ramp curriculum, fatigue /
torque-speed limits (numpy == torch, CPU gate path with a contract v5 brain), mirror maps of the new observation blocks,
optional observation clip in the exported graph."""

import gzip
import importlib.util
import json
import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from poolympic import action_log as AL
from poolympic import biomech as B
from poolympic import contract as C
from poolympic import evaluate as E

BRAINS = C.ROOT.parent / "parity" / "brains"
TOOLS = C.ROOT / "tools"


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- A: action log -------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not (BRAINS / "zero_brain.onnx").exists(), reason="zero_brain.onnx not exported")
def test_action_log_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(AL, "DIR", tmp_path)
    monkeypatch.setenv(AL.ENV, "unit")
    r = E.rung0_episode(BRAINS / "zero_brain.onnx", seed=3, seconds=1.0)
    assert AL.current is None
    path = tmp_path / "unit" / "rung0_seed3.jsonl.gz"
    lines = [json.loads(x) for x in gzip.open(path, "rt", encoding="utf-8")]
    head, ticks, summary = lines[0], lines[1:-1], lines[-1]
    assert head["header"] and head["gate"] == "rung0" and head["seed"] == 3 and len(head["actuators"]) == C.NUM_ACTIONS
    assert len(ticks) == 50 and [t["t"] for t in ticks[:3]] == [0, 1, 2]
    assert all(len(t[k]) == C.NUM_ACTIONS for t in ticks for k in ("act", "ctrl", "frc", "qd"))
    assert abs(ticks[0]["pos"][2] - E.MATT_ROOT_Z) < 1e-3 and ticks[0]["cmd"] == [0.0, 0.0, 0.0]
    assert summary["summary"] and summary["ticks"] == 50 and summary["result"]["seed"] == 3
    assert summary["outcome"] == ("fell" if r.fell else "ok")

    rep = _tool("report_actions")
    row = rep.aggregate(tmp_path / "unit")
    assert row["episodes"] == 1 and row["ticks"] == 50 and set(row["torque_nm"]) == set(B.JOINT_GROUPS)
    assert "unit" in rep.markdown([row])

    monkeypatch.delenv(AL.ENV)                       # off: no file, same result
    r2 = E.rung0_episode(BRAINS / "zero_brain.onnx", seed=3, seconds=1.0)
    assert r2.max_foot_excursion_m == r.max_foot_excursion_m and len(list((tmp_path / "unit").iterdir())) == 1


@pytest.mark.skipif(not (BRAINS / "rung2.onnx").exists(), reason="rung2.onnx not exported")
def test_action_log_event_lanes(tmp_path, monkeypatch):
    from poolympic.events import iron_pedestal
    monkeypatch.setattr(AL, "DIR", tmp_path)
    monkeypatch.setenv(AL.ENV, "heat")
    iron_pedestal.run_heat(BRAINS / "rung2.onnx", seed=2, max_s=1.0)
    row = _tool("report_actions").aggregate(tmp_path / "heat")
    assert row["episodes"] == 8 and row["ticks"] == 8 * 50


# ---- B: self-contact force penalty ------------------------------------------------------------------------------------
def test_self_contact_force():
    from poolympic.tasks import mdp
    f = torch.zeros(3, 4, 3)
    f[1, 0] = torch.tensor([3.0, 4.0, 0.0]) * 100          # 500 N
    f[2, :2, 2] = 400.0                                    # two contacts of 400 N
    env = SimpleNamespace(scene={mdp.SELF_SENSOR: SimpleNamespace(data=SimpleNamespace(force=f))})
    out = mdp.self_contact_force(env, body_weight_n=1000.0)
    assert out.tolist() == pytest.approx([0.0, 0.5, 0.8])
    assert mdp.self_contact_force(env, body_weight_n=1000.0, deadband_bw=0.45).tolist() == pytest.approx([0.0, 0.05, 0.0])


# ---- C: shove ramp (gust_curriculum generalised) ---------------------------------------------------------------------
def _fake_env(n=100):
    return SimpleNamespace(episode_length_buf=torch.zeros(n, dtype=torch.long), max_episode_length=1000,
                           common_step_counter=0)


def test_gust_curriculum_per_reset_is_unchanged():
    from poolympic.tasks import mdp
    env, ids = _fake_env(), torch.arange(10)
    kw = dict(start=0.3, step=0.02, max_level=0.34, up=0.6, down=0.3)
    env.episode_length_buf[:] = 999
    for want in (0.32, 0.34, 0.34):                       # every reset batch decides; capped at max_level
        assert float(mdp.gust_curriculum(env, ids, **kw)["gust_mps"]) == pytest.approx(want)
    env.episode_length_buf[:] = 5
    assert float(mdp.gust_curriculum(env, ids, **kw)["gust_mps"]) == pytest.approx(0.32)


def test_gust_curriculum_pooled_per_iteration():
    from poolympic.tasks import mdp
    env, ids = _fake_env(), torch.arange(40)
    kw = dict(start=0.7, step=0.02, max_level=1.4, up=0.8, down=0.5, steps_per_it=24, min_count=64)
    env.episode_length_buf[:] = 999                       # all survive
    env.common_step_counter = 30                          # iteration 1
    assert float(mdp.gust_curriculum(env, ids, **kw)["gust_mps"]) == pytest.approx(0.7)      # 40 < min_count
    assert float(mdp.gust_curriculum(env, ids, **kw)["gust_mps"]) == pytest.approx(0.72)     # 80 episodes: one step
    assert float(mdp.gust_curriculum(env, ids, **kw)["gust_mps"]) == pytest.approx(0.72)     # same iteration: no more
    env.common_step_counter = 60                          # iteration 2: 40 left over + 20 of 40 = 75 %, between down and up
    env.episode_length_buf[:20] = 5
    for _ in range(2):
        level = float(mdp.gust_curriculum(env, ids, **kw)["gust_mps"])
    assert level == pytest.approx(0.72)
    env.common_step_counter = 90                          # iteration 3, none survive: back down, never below start
    env.episode_length_buf[:] = 5
    for _ in range(2):
        level = float(mdp.gust_curriculum(env, ids, **kw)["gust_mps"])
    assert level == pytest.approx(0.7)


# ---- D: fatigue + torque-speed limit -----------------------------------------------------------------------------------
def test_fatigue_rises_recovers_and_keeps_the_floor():
    dt, g = 0.02, len(B.JOINT_GROUPS)
    level = np.zeros(g)
    hist = []
    for _ in range(int(5.0 / dt)):                         # 5 s at the reference effort (a quarter of the cap)
        level = B.fatigue_step(level, np.full(g, B.FATIGUE_EFFORT_REF), dt)
        hist.append(level[0])
    assert all(b > a for a, b in zip(hist, hist[1:])) and 0.3 < level[0] < 0.6
    loaded = level.copy()
    for _ in range(int(60.0 / dt)):                        # a minute at the cap
        level = B.fatigue_step(level, np.ones(g), dt)
    assert np.all(level == 1.0)
    lo, hi = B.force_scale(level, np.zeros(g), np.full(g, 20.0))
    assert np.allclose(lo, B.FATIGUE_FLOOR) and np.allclose(hi, B.FATIGUE_FLOOR)   # exhausted: the floor, never below
    level = loaded
    for _ in range(int(10.0 / dt)):                        # 10 s of rest
        nxt = B.fatigue_step(level, np.zeros(g), dt)
        assert np.all(nxt < level)
        level = nxt
    assert level[0] == pytest.approx(loaded[0] * math.exp(-10.0 / B.FATIGUE_TAU_REST), rel=0.01)
    assert np.all(B.fatigue_step(np.zeros(g), np.zeros(g), dt) == 0.0)


def test_torque_speed_scale_is_one_sided():
    w = np.full(4, 20.0)
    qd = np.array([0.0, 10.0, -10.0, 30.0])
    lo, hi = B.force_scale(np.zeros(4), qd, w)
    assert hi.tolist() == pytest.approx([1.0, 0.5, 1.0, 0.0])      # torque WITH the motion fades with speed
    assert lo.tolist() == pytest.approx([1.0, 1.0, 0.5, 1.0])      # braking torque keeps the full cap
    f = 1.0 - (1.0 - B.FATIGUE_FLOOR) * 0.5
    lo, hi = B.force_scale(np.full(4, 0.5), qd, w)
    assert hi.tolist() == pytest.approx([f, 0.5 * f, f, 0.0]) and lo.min() >= 0.5 * B.FATIGUE_FLOOR


def test_fatigue_formulas_numpy_equals_torch():
    rng = np.random.default_rng(0)
    level, effort, qd = rng.uniform(0, 1, (5, 6)), rng.uniform(0, 1.2, (5, 6)), rng.normal(0, 12, (5, 6))
    w = np.array([8.0, 20.0, 20.0, 15.0, 22.0, 18.0])
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
    assert np.allclose(B.fatigue_step(level, effort, 0.02), B.fatigue_step(t(level), t(effort), 0.02).numpy(), atol=1e-12)
    for a, b in zip(B.force_scale(level, qd, w), B.force_scale(t(level), t(qd), t(w))):
        assert np.allclose(a, b.numpy(), atol=1e-12)
    gi, mean, wmax = B.tables(["abdomen_flex", "hip_flex_l", "hip_flex_r", "knee_l", "ankle_dorsi_r"])
    assert gi.tolist() == [0, 3, 3, 4, 5] and wmax.tolist() == [8.0, 15.0, 15.0, 22.0, 18.0]
    assert (np.array([1.0, 0.2, 0.6, 0.5, 0.0]) @ mean).tolist() == pytest.approx([1.0, 0.0, 0.0, 0.4, 0.5, 0.0])


def test_v5_brain_runs_in_the_cpu_gate(tmp_path):
    """A 90-input (contract v5 proposal) brain: evaluate.Sim feeds the fatigue block and applies the force limits."""
    from poolympic.policy_export import ExportedPolicy, export, make_mlp, standard_metadata
    m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
    ath = C.Athlete.bind(m)
    torch.manual_seed(0)
    pol = ExportedPolicy(make_mlp(C.OBS_DIM_V5), np.zeros(C.OBS_DIM_V5), np.ones(C.OBS_DIM_V5), ath.default_pos,
                         ath.range_lo, ath.range_hi, clip_obs=None)
    contract = json.loads(C.CONTRACT_JSON.read_text())
    onnx_path = export(pol, tmp_path / "v5.onnx", standard_metadata(contract, "v5", C.FATIGUE_VERSION))
    assert json.loads((tmp_path / "v5.onnx.json").read_text())["contract_version"] == "5"
    sim = E.Sim(onnx_path)
    assert sim.v5 and sim.bio is not None
    base = sim.bio.base.copy()
    ids = ath.actuator_ids
    sim.control_tick(np.zeros(3))                          # first tick after a reset: fresh, nominal force range
    assert np.array_equal(sim.m.actuator_forcerange[ids], base) and np.all(sim.bio.level == 0.0)
    scales = []
    for _ in range(100):
        sim.control_tick(np.array([1.0, 0.0, 0.0]))
        scales.append(np.abs(sim.m.actuator_forcerange[ids]) / np.abs(base))
    s = np.array(scales)
    assert np.all(sim.bio.level > 0.0) and np.all(sim.bio.level <= 1.0)
    assert s.max() <= 1.0 + 1e-12 and s.min() >= 0.0 and s.min() < 1.0     # the limits bite, and only downwards
    sim.reset()
    assert np.array_equal(sim.m.actuator_forcerange[ids], base) and np.all(sim.bio.level == 0.0)
    v3 = E.Sim(BRAINS / "zero_brain.onnx") if (BRAINS / "zero_brain.onnx").exists() else None
    assert v3 is None or v3.bio is None                    # existing brains: no limits, unchanged physics


# ---- E: mirror maps of the new blocks, optional observation clip --------------------------------------------------------
def test_mirror_of_fatigue_and_privileged_blocks():
    from poolympic.tasks import mdp, symmetry as S
    assert set(S.TAILS) == {84, 95, 90, 84 + mdp.PRIV_DIM, 90 + mdp.PRIV_DIM}
    torch.manual_seed(0)
    for dim in S.TAILS:
        x = torch.randn(7, dim)
        x[:, S.CMD_SLICE] = 1.0                            # moving: the gait clock shifts by half a stride
        assert torch.allclose(S.mirror_obs(S.mirror_obs(x)), x)
    x = torch.randn(4, 90 + mdp.PRIV_DIM)
    y = S.mirror_obs(x)
    assert torch.equal(y[:, 84:90], x[:, 84:90])           # fatigue: both sides share a level
    p, q = x[:, 90:], y[:, 90:]                            # (L vert, L horiz, R vert, R horiz, friction, mass, shove x, y)
    assert torch.equal(q[:, :4], p[:, [2, 3, 0, 1]]) and torch.equal(q[:, 4:7], p[:, 4:7]) and torch.equal(q[:, 7], -p[:, 7])


def test_export_observation_clip_is_optional():
    from poolympic.policy_export import ExportedPolicy, make_mlp
    torch.manual_seed(0)
    mlp = make_mlp()
    args = (mlp, np.zeros(C.OBS_DIM), np.ones(C.OBS_DIM), np.zeros(23), -np.ones(23), np.ones(23))
    far, edge = torch.full((1, C.OBS_DIM), 40.0), torch.full((1, C.OBS_DIM), 5.0)
    clipped, plain = ExportedPolicy(*args, clip_obs=5.0), ExportedPolicy(*args, clip_obs=None)
    assert torch.equal(clipped(far)[1], clipped(edge)[1])
    assert not torch.equal(plain(far)[1], plain(edge)[1]) and torch.equal(plain(edge)[1], clipped(edge)[1])
