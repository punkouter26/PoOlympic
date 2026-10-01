"""Crowd events: every body part of every athlete collides with every body part of the 7 others (tools/check_crowd_contacts.py);
the G6 testbed keeps full lane isolation."""

import sys

import pytest

from poolympic import contract as C

sys.path.insert(0, str(C.ROOT / "tools"))
import check_crowd_contacts as cc  # noqa: E402


@pytest.mark.parametrize("tag", cc.scene_tags())
def test_every_part_collides_with_every_other_athlete(tag):
    r = cc.check_scene(tag)
    assert r["athletes"] == 8
    assert not r["filter_fail"], r["filter_fail"][:5]
    assert not r["contact_fail"], r["contact_fail"][:5]
    assert not r["support_fail"], r["support_fail"][:5]
    assert not r["self_mismatch"], r["self_mismatch"]


def test_g6_testbed_stays_lane_isolated():
    r = cc.check_scene("meet8")
    assert len(r["filter_fail"]) == r["filter_pairs"] and len(r["contact_fail"]) == r["contact_pairs"]
