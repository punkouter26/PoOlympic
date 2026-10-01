"""Event rules with the shipped Rung 2 brain: one CPU heat each (events 5, 9, 10, 11, 12)."""

import math

import pytest

from poolympic import contract as C
from poolympic.events import crab, gauntlet, slalom, track, turntable

BRAIN = C.ROOT.parent / "parity" / "brains" / "rung2.onnx"
pytestmark = pytest.mark.skipif(not BRAIN.exists(), reason="rung2.onnx not exported")


def test_inverted_sprint_heat():
    res = track.run_race(BRAIN, "inverted", seed=1)
    assert sorted(l.place for l in res.lanes) == list(range(1, 9))
    finished = [l for l in res.lanes if l.status == "FINISHED"]
    assert len(finished) >= 6
    # 20 m at a -1.5 m/s command from standstill
    assert all(13.0 < l.finish_s < 20.0 for l in finished)
    by_place = sorted(finished, key=lambda l: l.place)
    assert all(a.finish_s <= b.finish_s for a, b in zip(by_place, by_place[1:]))


def test_turntable_heat():
    res = turntable.run_heat(BRAIN, seed=1)
    assert res.direction in (1, -1)
    assert sorted(l.place for l in res.lanes) == list(range(1, 9))
    done = [l for l in res.lanes if l.status == "DONE"]
    assert len(done) >= 6
    for l in done:
        # 3 turns at a 3 rad/s command: at best 2*pi*3/3 = 6.3 s
        assert 2 * math.pi * turntable.TURNS / turntable.WZ - 0.1 < l.time_s < 10.0
        assert l.max_drift_m < turntable.RING_R
        assert l.score == pytest.approx(l.time_s + turntable.DRIFT_PENALTY * l.max_drift_m)
    by_place = sorted(done, key=lambda l: l.place)
    assert all(a.score <= b.score for a, b in zip(by_place, by_place[1:]))


def test_crab_shuffle_heat():
    res = crab.run_heat(BRAIN, seed=1)
    assert sorted(l.place for l in res.lanes) == list(range(1, 9))
    done = [l for l in res.lanes if l.status == "FINISHED"]
    assert len(done) >= 6
    for l in done:
        # 20 m at a 1.2 m/s side-step command
        assert crab.DISTANCE / crab.VY < l.finish_s < 30.0
        assert l.max_x_drift_m < 0.5          # rails are 0.61 m from the lane line
        assert l.score == pytest.approx(l.finish_s + crab.CROSS_PENALTY * l.crossings + crab.RAIL_PENALTY * l.rail_touches)


def test_slalom_heat():
    res = slalom.run_heat(BRAIN, seed=2)        # run_heat also asserts the course constants against the pole geoms
    assert sorted(l.place for l in res.lanes) == list(range(1, 9))
    done = [l for l in res.lanes if l.status == "FINISHED"]
    assert len(done) >= 5
    for l in done:
        assert slalom.LINE[0] <= l.line_m <= slalom.LINE[1]
        assert slalom.DISTANCE / slalom.VX < l.finish_s < slalom.MAX_S
        assert l.score == pytest.approx(l.finish_s + slalom.MISS_PENALTY * l.misses + slalom.CLIP_PENALTY * l.clips)


def test_slalom_line_passes_each_pole_on_its_side():
    for g in range(slalom.N_POLES):
        y, _ = slalom.line_y(slalom.POLE_X0 + g * slalom.POLE_DX, 0.5)
        assert y == pytest.approx(0.5 if g % 2 == 0 else -0.5)
    assert slalom.line_y(0.0, 0.5) == (0.0, 0.0) and slalom.line_y(slalom.DISTANCE, 0.5) == (0.0, 0.0)


def test_gust_gauntlet_heat():
    res = gauntlet.run_heat(BRAIN, seed=1)
    assert sorted(l.place for l in res.lanes) == list(range(1, 9))
    for l in res.lanes:
        assert len(l.recoveries) == gauntlet.N_ROUNDS
        assert all(0.0 <= r <= gauntlet.ROUND_S for r in l.recoveries)
        assert l.total_s == pytest.approx(sum(l.recoveries))
    # survivors rank ahead of the eliminated, the eliminated by elimination time (later = better)
    by_place = sorted(res.lanes, key=lambda l: l.place)
    outs = [l.out_at_s for l in by_place if l.out_at_s is not None]
    assert all(l.out_at_s is None for l in by_place[: 8 - len(outs)])
    assert outs == sorted(outs, reverse=True)
    assert any(l.out_at_s is None or l.out_at_s > 10.0 for l in res.lanes)   # the first bursts are survivable


def test_homing_command_walks_back_to_the_spot():
    q0 = [1.0, 0.0, 0.0, 0.0]
    vx, vy, wz = gauntlet.homing_command(q0, 0.3, -0.2)          # pushed forward-right → walk back-left
    assert vx < 0 and vy > 0 and wz == 0.0
    assert gauntlet.homing_command(q0, 0.03, 0.03).tolist() == [0.0, 0.0, 0.0]   # inside the deadband
    half = math.radians(90) / 2                                    # facing +y: world +x offset is to the right
    vx, vy, _ = gauntlet.homing_command([math.cos(half), 0.0, 0.0, math.sin(half)], 0.3, 0.0)
    assert vy > 0 and abs(vx) < 1e-9


def test_crab_command_holds_the_course():
    q0 = [1.0, 0.0, 0.0, 0.0]
    assert crab.crab_command(q0, 0.0).tolist() == [0.0, -crab.VY, 0.0]
    vx, _, _ = crab.crab_command(q0, 0.2)                 # drifted forward towards the next rail → step back
    assert vx < 0
    half = math.radians(20) / 2                            # turned 20° left → turn right
    assert crab.crab_command([math.cos(half), 0.0, 0.0, math.sin(half)], 0.0)[2] < 0


FLIGHT = C.ROOT.parent / "parity" / "brains" / "r2f_v3_it100.onnx"


@pytest.mark.skipif(not FLIGHT.exists(), reason="r2f_v3_it100.onnx not exported")
def test_steeplechase_heat():
    res = track.run_race(FLIGHT, "steeple", seed=1)
    assert sorted(l.place for l in res.lanes) == list(range(1, 9))
    finished = [l for l in res.lanes if l.status == "FINISHED"]
    assert len(finished) >= 6
    for l in finished:
        # 50 m at a 3.5 m/s command from standstill; the flight brain spends ~40 % of the race airborne
        assert 14.0 < l.finish_s < 18.0
        assert l.flights >= 30 and l.longest_ms >= 80.0
        assert l.score_s == pytest.approx(l.finish_s - l.hang_s, abs=1e-3)
    by_place = sorted(finished, key=lambda l: l.place)
    assert all(a.score_s <= b.score_s for a, b in zip(by_place, by_place[1:]))


CRAWL = {"matt": C.ROOT.parent / "parity" / "brains" / "crawl_matt.onnx",
         "zombie": C.ROOT.parent / "parity" / "brains" / "crawl_zombie.onnx"}


@pytest.mark.skipif(not CRAWL["matt"].exists(), reason="crawl brains not exported")
def test_trench_crawl_heat():
    from poolympic.events import all_fours
    res = all_fours.run_race(CRAWL, seed=1, scene=C.ROOT / "assets" / "scene_trench8_mzmzmzmz.xml",
                             layout_path=C.ROOT / "assets" / "trench8_mzmzmzmz_layout.json", distance=all_fours.TRENCH_DISTANCE)
    assert sorted(l.place for l in res.lanes) == list(range(1, 9))
    finished = [l for l in res.lanes if l.status == "FINISHED"]
    assert len(finished) >= 7
    # 16 m: MATT squeezes under the 0.72 m ceiling (~15-19 s), the zombie crawls at its own pace (~17 s)
    assert all(12.0 < l.finish_s < 30.0 for l in finished)


SQUAT_BRAIN = C.ROOT.parent / "parity" / "brains" / "rs_v6_it3299.onnx"


@pytest.mark.skipif(not SQUAT_BRAIN.exists(), reason="Rung S brain not exported")
def test_deep_squat_heat():
    from poolympic.events import squat
    res = squat.run_heat(SQUAT_BRAIN, seed=1)
    assert sorted(l.place for l in res.lanes) == list(range(1, 9))
    assert res.duration_s == pytest.approx(squat.START_S + squat.total_s(), abs=0.03)
    for l in res.lanes:
        assert l.status in ("DONE", "FELL", "STEPPED")
        assert len(l.reps) >= 8                                   # nobody is out before the 9th rep (~27 s)
        assert all(0.0 <= p <= squat.REP_POINTS for p in l.reps)
        assert l.points == pytest.approx(sum(l.reps) + (squat.BALANCE_BONUS if l.status == "DONE" else 0.0), abs=1e-2)
        assert all(e < 0.10 for e in l.errs_m[:6])               # the first six reps (25-35 cm) are accurate
    by_place = sorted(res.lanes, key=lambda l: l.place)
    assert all(a.points >= b.points for a, b in zip(by_place, by_place[1:]))


FLAMINGO_BRAIN = C.ROOT.parent / "parity" / "brains" / "rs_v6_it2550.onnx"


@pytest.mark.skipif(not FLAMINGO_BRAIN.exists(), reason="Rung S flamingo brain not exported")
def test_flamingo_heat():
    from poolympic.events import flamingo
    res = flamingo.run_heat(FLAMINGO_BRAIN, seed=1)
    assert res.foot in ("l", "r")
    for l in res.lanes:
        assert l.status in ("TOUCHDOWN", "HOPPED", "FELL")           # the rising gusts bring everyone down before MAX_S
        assert l.status == "FELL" or l.out_at_s >= flamingo.LIFT_S   # touches during the lift time do not count
        assert l.max_hop_m <= flamingo.HOP_TOL or l.status == "HOPPED"
    by_place = sorted(res.lanes, key=lambda l: l.place)
    assert by_place[0].place == 1 and all(a.out_at_s >= b.out_at_s for a, b in zip(by_place, by_place[1:]))
    assert 15.0 < by_place[0].out_at_s < 60.0                        # heats last 23-38 s
    assert res.duration_s == pytest.approx(flamingo.START_S + by_place[0].out_at_s, abs=0.03)
