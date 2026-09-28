"""Event rules with the shipped Rung 2 brain: one CPU heat each (events 9 and 12)."""

import math

import pytest

from poolympic import contract as C
from poolympic.events import track, turntable

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
