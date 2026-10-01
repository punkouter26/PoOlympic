"""Crowd contact audit (2026-09-30, user: "verify all parts of each creature can accurately collide with all parts of all
the 7 other players during an event"). For every event scene and lineup:

  filter    every geom of athlete i passes MuJoCo's contype/conaffinity filter against every geom of each of the other
            athletes, and no <exclude> hides the pair
  contact   for every ordered athlete pair (i, j) and every body-part pair (geom a of i, geom b of j): athlete j is moved
            (root translation) so that part b overlaps part a, then mj_kinematics + mj_collision must report a contact
            between exactly those two geoms — the real broad/mid/narrow phase, not just the bitmask
  self      an athlete's own part pairs are the ones of the isolated G6 scene (crowd bits add no self-collision)
  support   every geom of every athlete still collides with the ground / stage (ground, beam, floor, props)

The geoms are the capsules / spheres / boxes fitted to each body's skinned mesh (build_mjcf.fit_geoms), so a contact here
is a contact between the athletes' visible bodies. Usage (training/):
    uv run python tools/check_crowd_contacts.py            # all event scenes: all-MATT + the mixed lineups (zombie, GRANDMA)
    uv run python tools/check_crowd_contacts.py track8     # one scene
Writes parity/crowd_contacts.json; exit code 1 on any failure.
"""

from __future__ import annotations

import json
import sys
import time
from itertools import permutations
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
EVENT_SCENES = ["pedestal8", "shaker8", "crawl8", "slalom8", "turntable8", "track8", "crab8", "trench8"]
LINEUPS = ["", "_mzmzmzmz", "_mzgmzgmg"]        # all-MATT, alternating MATT / zombie, MATT / zombie / GRANDMA
NUDGE = np.array([0.011, 0.007, 0.004])         # m — never place two geom centres exactly on top of each other


def athletes(m: mujoco.MjModel) -> dict[str, list[int]]:
    """prefix -> geom ids of that athlete (every geom under its pelvis)."""
    out = {}
    for b in range(m.nbody):
        name = m.body(b).name
        if name.endswith("pelvis") and m.body_parentid[b] == 0:
            prefix = name[: -len("pelvis")]
            out[prefix] = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == b]
    return out


def excluded(m: mujoco.MjModel) -> set[tuple[int, int]]:
    return {tuple(sorted((int(sig) >> 16, int(sig) & 0xFFFF))) for sig in m.exclude_signature}   # (body1 << 16) + body2


def passes_filter(m, a: int, b: int, excl: set) -> bool:
    ok = bool((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a]))
    return ok and tuple(sorted((int(m.geom_bodyid[a]), int(m.geom_bodyid[b])))) not in excl


def own_pairs(m, geoms: list[int], excl: set) -> set[tuple[str, str]]:
    """Self-collision pairs of one athlete by part name (prefix stripped), parent/child skipped as MuJoCo does."""
    pairs = set()
    for i, a in enumerate(geoms):
        for b in geoms[i + 1:]:
            ba, bb = m.geom_bodyid[a], m.geom_bodyid[b]
            if ba == bb or m.body_parentid[ba] == bb or m.body_parentid[bb] == ba:
                continue
            if passes_filter(m, a, b, excl):
                pairs.add(tuple(sorted((m.geom(a).name.split("_", 1)[1], m.geom(b).name.split("_", 1)[1]))))
    return pairs


def check_scene(tag: str) -> dict:
    t0 = time.time()
    m = mujoco.MjModel.from_xml_path(str(ASSETS / f"scene_{tag}.xml"))
    d = mujoco.MjData(m)
    ath = athletes(m)
    excl = excluded(m)
    prefixes = sorted(ath)
    res = {"scene": tag, "athletes": len(prefixes), "geoms_per_athlete": sorted({len(g) for g in ath.values()}),
           "filter_fail": [], "contact_fail": [], "support_fail": []}
    # filter: every cross-athlete geom pair
    n_filter = 0
    for p, q in permutations(prefixes, 2):
        for a in ath[p]:
            for b in ath[q]:
                n_filter += 1
                if not passes_filter(m, a, b, excl):
                    res["filter_fail"].append((m.geom(a).name, m.geom(b).name))
    res["filter_pairs"] = n_filter
    # support: every athlete geom against every world geom it may stand on / hit
    stage = [g for g in range(m.ngeom)
             if m.geom(g).name in ("ground", "pedestal", "shaker") or m.geom(g).name.endswith(("_pedestal", "_shaker"))]
    for p in prefixes:
        for a in ath[p]:
            for w in stage:
                if not passes_filter(m, a, w, excl):
                    res["support_fail"].append((m.geom(a).name, m.geom(w).name))
    # self: the athlete's own pairs == the isolated G6 athlete's (by body)
    lineup = tag.split("_", 1)[1] if "_" in tag else ""          # same lineup in the isolated G6 scene (meet8)
    iso = mujoco.MjModel.from_xml_path(str(ASSETS / (f"scene_meet8_{lineup}.xml" if lineup else "scene_meet8.xml")))
    iso_ath, iso_excl = athletes(iso), excluded(iso)
    lanes = {l["prefix"]: l["lane"] for l in json.loads((ASSETS / f"{tag}_layout.json").read_text())["lanes"]}
    res["self_mismatch"] = [p for p in prefixes
                            if own_pairs(m, ath[p], excl) != own_pairs(iso, iso_ath[f"L{lanes[p]}_"], iso_excl)]
    # contact: put part b of athlete j onto part a of athlete i; the collision pipeline must report (a, b)
    key = m.key("default").qpos.copy()
    root_adr = {p: m.jnt_qposadr[m.body_jntadr[m.body(p + "pelvis").id]] for p in prefixes}   # free joint of the pelvis
    d.qpos[:] = key
    mujoco.mj_kinematics(m, d)
    base_xpos = d.geom_xpos.copy()
    n_contact = 0
    for p, q in permutations(prefixes, 2):
        for b in ath[q]:
            for a in ath[p]:
                n_contact += 1
                d.qpos[:] = key
                r = root_adr[q]
                d.qpos[r: r + 3] += base_xpos[a] - base_xpos[b] + NUDGE
                mujoco.mj_kinematics(m, d)
                mujoco.mj_collision(m, d)
                hit = any({int(c.geom1), int(c.geom2)} == {a, b} for c in d.contact[: d.ncon])
                if not hit:
                    res["contact_fail"].append((m.geom(a).name, m.geom(b).name))
    res["contact_pairs"] = n_contact
    res["ok"] = not (res["filter_fail"] or res["contact_fail"] or res["support_fail"] or res["self_mismatch"])
    res["seconds"] = round(time.time() - t0, 1)
    return res


def scene_tags(scenes: list[str] | None = None) -> list[str]:
    """Every composed lineup of the scenes (GRANDMA has no crawl brain: no GRANDMA lineup of the all-fours scenes)."""
    return [f"{s}{l}" for s in (scenes or EVENT_SCENES) for l in LINEUPS if (ASSETS / f"scene_{s}{l}.xml").exists()]


def main(argv: list[str]) -> int:
    tags = scene_tags(argv)
    report = []
    for tag in tags:
        r = check_scene(tag)
        report.append(r)
        print(f"{tag:22s} {'PASS' if r['ok'] else 'FAIL'}  athletes {r['athletes']}  geoms/athlete {r['geoms_per_athlete']}  "
              f"filter {r['filter_pairs'] - len(r['filter_fail'])}/{r['filter_pairs']}  "
              f"contacts {r['contact_pairs'] - len(r['contact_fail'])}/{r['contact_pairs']}  "
              f"support fails {len(r['support_fail'])}  self mismatches {len(r['self_mismatch'])}  ({r['seconds']} s)", flush=True)
        for k in ("filter_fail", "contact_fail", "support_fail"):
            if r[k]:
                print(f"   {k}: {r[k][:6]}{' …' if len(r[k]) > 6 else ''}")
    out = ROOT.parent / "parity" / "crowd_contacts.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"-> {out}")
    return 0 if all(r["ok"] for r in report) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
