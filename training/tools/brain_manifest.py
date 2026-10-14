"""Production brain manifest: parity/brains/manifest.json — one record per DEPLOYED brain, generated from the files.

Deployed = a brain in parity/brains whose name has no `_itNNN` suffix (rung2.onnx, kim_rung0.onnx, ...) or that has a
confidence critic next to it (<name>.critic.onnx: critics are exported only for brains the game runs, e.g.
r0_v2_it1000, r2f_v3_it100, rs_v8_it1199). Per brain: file, body (from the sidecar's model fingerprint), role + events
(ROLES below — the one hand-kept table), source run + iteration (sidecar checkpoint path), contract version, sha256,
fingerprint, critic, and the G1 / probe result of that checkpoint when a parity/watch*.jsonl or eval JSON has it.

  uv run python tools/brain_manifest.py            # (re)generate after deploying a brain
  uv run python tools/brain_manifest.py --check    # exit 1 when a deployed file is missing, changed, or not listed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT.parent / "parity"
BRAINS = PARITY / "brains"
MANIFEST = BRAINS / "manifest.json"
BODIES = ("matt", "mattbio", "zombie", "grandma", "kim")
LOCO = [5, 9, 10, 11, 12, 19, 22]        # Gust Gauntlet, Inverted Sprint, Crab, Slalom, Turntable, Terminal Velocity, Brake
# brain file stem -> (role, events it drives, body when the fingerprint cannot tell). Roles are what tools/fit_odds.py and
# the event scenes ask for: one brain per (role, body).
ROLES = {
    "rung2": ("rung2", LOCO, "matt"), "zombie_rung2": ("rung2", LOCO, "zombie"),
    "grandma_rung2": ("rung2", LOCO, "grandma"), "kim_rung2": ("rung2", LOCO, "kim"),
    "r0_v2_it1000": ("pedestal", [1], "matt"), "zombie_rung0": ("pedestal", [1], "zombie"),
    "grandma_rung0": ("pedestal", [1], "grandma"), "kim_rung0": ("pedestal", [1], "kim"),
    "crawl_matt": ("crawl", [8, 23], "matt"), "crawl_zombie": ("crawl", [8, 23], "zombie"),
    "crawl_grandma": ("crawl", [8, 23], "grandma"), "crawl_kim": ("crawl", [8, 23], "kim"),
    "slope_matt": ("slope", [14], "matt"), "slope_zombie": ("slope", [14], "zombie"),
    "slope_grandma": ("slope", [14], "grandma"), "slope_kim": ("slope", [14], "kim"),
    "r2f_v3_it100": ("flight", [13], "matt"),
    "rs_v8_it1199": ("stance", [7], "matt"), "rs_v6_it2550": ("stance", [6], "matt"), "rs_v6_it3299": ("stance", [], "matt"),
    "rsq_v2_it500": ("stance", [3], "matt"),               # Deep Squat specialist (2026-10-13): 80 of 80 finish the 12 reps
    "getup_kim": ("getup", [27], "kim"),                  # + fall recovery in the track races and the slalom
    "getup_matt": ("getup", [27], "matt"),                  # = mgetup_prone_v1 it 600 (2026-10-11), supine + face down 30/30
    "getup_zombie": ("getup", [27], "zombie"),            # = zgetup_prone_v1 it 600 (2026-10-11), supine + face down 30/30
    "getup_grandma": ("getup", [27], "grandma"),            # = ggetup_roll_v4_it599, back from git history (2026-10-11)
    "rung1": ("rung1", [], "matt"), "r0_v2_latest": ("rung0-old", [], "matt"),
    "random_brain": ("test", [], "matt"), "random_brain_v4": ("test", [], "matt"), "zero_brain": ("test", [], "matt"),
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def deployed_files() -> list[Path]:
    out = []
    for p in sorted(BRAINS.glob("*.onnx")):
        if p.name.endswith(".critic.onnx"):
            continue
        if not re.search(r"_it\d+$", p.stem) or (BRAINS / f"{p.stem}.critic.onnx").exists():
            out.append(p)
    return out


def _fingerprints() -> dict[str, str]:
    """model fingerprint sha -> body (parity/contract*.json)."""
    out = {}
    for b in BODIES:
        f = PARITY / ("contract.json" if b == "matt" else f"contract_{b}.json")
        if f.exists():
            out[json.loads(f.read_text()).get("fingerprint_sha256") or ""] = b
    return out


def _gate(run: str | None, it: int | None, names: list[str]) -> dict | None:
    """The gate record of a checkpoint: its line in parity/watch_<prefix>.jsonl, else an eval JSON named after it."""
    for prefix in dict.fromkeys([*(re.sub(r"_it\d+$", "", n) for n in names), run]):
        f = PARITY / f"watch_{prefix}.jsonl" if prefix else None
        if f and f.exists() and it is not None:
            for line in f.read_text().splitlines():
                rec = json.loads(line)
                if rec.get("it") == it:
                    return {"source": f.name, **{k: v for k, v in rec.items() if k.startswith(("g1_", "getup", "crawl"))
                                                 and not k.endswith("_log")}}
    for n in names:
        for f in sorted(PARITY.glob(f"eval_rung*_{n}.json")) + sorted((PARITY / "watch").glob(f"g1_rung*_{n}.json")):
            d = json.loads(f.read_text())
            if "passed_seeds" in d:
                return {"source": f.relative_to(PARITY).as_posix(), "result": f"{d['passed_seeds']}/{d['seeds']}",
                        "PASS": d.get("PASS")}
    return None


def record(p: Path, fp_body: dict[str, str]) -> dict:
    side = json.loads(Path(str(p) + ".json").read_text()) if Path(str(p) + ".json").exists() else {}
    ckpt = (side.get("checkpoint") or "").replace("\\", "/")
    m = re.search(r"([^/]+)/(?:\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d_)?([^/]+)/model_(\d+)\.pt$", ckpt)
    exp, run, it = (m.group(1), m.group(2), int(m.group(3))) if m else (None, None, None)
    role, events, body = ROLES.get(p.stem, ("unassigned", [], None))
    critic = BRAINS / f"{p.stem}.critic.onnx"
    digest = sha256(p)
    return {"file": p.name, "body": fp_body.get(side.get("fingerprint_sha256") or "-", body), "role": role, "events": events,
            "name": side.get("name"), "experiment": exp, "run": run, "iteration": it,
            "contract_version": int(side["contract_version"]) if side.get("contract_version") else None,
            "sha256": digest, "sidecar_sha256_ok": side.get("onnx_sha256") == digest if side else None,
            "fingerprint_sha256": side.get("fingerprint_sha256") or None,
            "fingerprint_current": (side.get("fingerprint_sha256") or "-") in fp_body,
            "critic": critic.name if critic.exists() else None,
            "gate": _gate(run, it, [n for n in (side.get("name"), p.stem) if n])}


def build() -> dict:
    fp = _fingerprints()
    return {"note": "generated by training/tools/brain_manifest.py — do not edit; re-run after deploying a brain",
            "brains": [record(p, fp) for p in deployed_files()]}


def check() -> int:
    if not MANIFEST.exists():
        print(f"missing {MANIFEST} — run tools/brain_manifest.py")
        return 1
    listed = {b["file"]: b for b in json.loads(MANIFEST.read_text())["brains"]}
    bad = 0
    for name, b in listed.items():
        p = BRAINS / name
        if not p.exists():
            bad += 1
            print(f"MISSING  {name}")
        elif sha256(p) != b["sha256"]:
            bad += 1
            print(f"CHANGED  {name} (manifest {b['sha256'][:12]}, file {sha256(p)[:12]})")
    for p in deployed_files():
        if p.name not in listed:
            bad += 1
            print(f"UNLISTED {p.name}")
    print(f"{len(listed)} deployed brains, {bad} problem(s)")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check()
    man = build()
    MANIFEST.write_text(json.dumps(man, indent=1) + "\n")
    for b in man["brains"]:
        gate = {k: v for k, v in (b["gate"] or {}).items() if k != "source"}
        print(f"{b['file']:26s} {str(b['body']):8s} {b['role']:10s} {str(b['run']):16s} it {str(b['iteration']):5s} "
              f"v{b['contract_version']} {'' if b['fingerprint_current'] else 'OLD-FINGERPRINT '}{gate or ''}")
    loose = [b["file"] for b in man["brains"] if b["role"] == "unassigned"]
    if loose:
        print(f"no role in ROLES (add them): {loose}")
    print(f"wrote {MANIFEST} ({len(man['brains'])} brains)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
