"""Unattended checkpoint gate for any rung (recipe v5 successor of watch_eval.py): wait for model_<it>.pt, export it,
run the G1 drills of every requested rung on CPU MuJoCo (eval_cpu.py, sequentially — parallel CPU evals slow training
to a crawl) and, optionally, the bio-realism probe. One JSON line per checkpoint in parity/watch_<prefix>.jsonl.

Usage: uv run python tools/watch_gate.py <run_dir> <prefix> --gates S,2 --its 200,400,600 [--seeds 10] [--bio]
Body = $POOLYMPIC_BODY (the gates and exports use that body's contract / scene).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT.parent / "parity"


def run(args: list[str]) -> str:
    p = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    return p.stdout + p.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("prefix")
    ap.add_argument("--gates", default="2", help="comma list of 0, 1, 2, S, getup")
    ap.add_argument("--its", required=True, help="comma list of checkpoint iterations")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--bio", action="store_true", help="also run tools/bio_probe.py (walk / run / sprint / spin)")
    a = ap.parse_args()
    out = PARITY / f"watch_{a.prefix}.jsonl"
    for it in [int(x) for x in a.its.split(",")]:
        ckpt = a.run / f"model_{it}.pt"
        while not ckpt.exists():
            time.sleep(30)
        time.sleep(20)                                   # let the checkpoint finish writing
        name = f"{a.prefix}_it{it}"
        exp = run([str(ROOT / "tools" / "export_brain.py"), str(ckpt), name])
        onnx = PARITY / "brains" / f"{name}.onnx"
        rec = {"it": it, "brain": onnx.name, "export": exp.strip().splitlines()[-1] if exp.strip() else ""}
        for g in a.gates.split(","):
            if g in ("getup", "getup_prone"):            # R3: tools/getup_probe.py (supine / face-down start, 10 s)
                txt = run([str(ROOT / "tools" / "getup_probe.py"), str(onnx), "--seeds", str(a.seeds),
                           "--out", str(PARITY / "watch" / f"{g}_{name}.json")] + (["--prone"] if g == "getup_prone" else []))
                m = re.search(r"up (\d+)/(\d+).*?up at end (\d+)/", txt)
                rec[g] = f"up {m.group(1)}/{m.group(2)}, at end {m.group(3)}" if m else "ERROR " + txt[-300:]
                continue
            if g == "crawl":                             # Event 8: tools/crawl_probe.py (prone start, 30 m at 1.2 m/s)
                txt = run([str(ROOT / "tools" / "crawl_probe.py"), str(onnx), "--seeds", "5",
                           "--out", str(PARITY / "watch" / f"crawl_{name}.json")])
                m = re.search(r"finished (\d+)/(\d+) in ([\d.]+-[\d.]+) s.*?lane ([\d.]+) m", txt)
                rec["crawl"] = f"finished {m.group(1)}/{m.group(2)} in {m.group(3)} s, lane {m.group(4)} m" if m else "ERROR " + txt[-300:]
                continue
            txt = run([str(ROOT / "tools" / "eval_cpu.py"), str(onnx), "--rung", g, "--seeds", str(a.seeds),
                       "--out", str(PARITY / "watch" / f"g1_rung{g}_{name}.json")])
            m = re.search(rf"G1 rung {g}: (\d+)/(\d+) seeds pass(.*?)->", txt)
            rec[f"g1_{g}"] = f"{m.group(1)}/{m.group(2)}{m.group(3).strip() and ' ' + m.group(3).strip()}" if m else "ERROR"
            if not m:
                rec[f"g1_{g}_log"] = txt[-800:]
            falls = len(re.findall(r"fell=(?!None|False)", txt))
            rec[f"g1_{g}_falls"] = falls
        if a.bio:
            bio_json = PARITY / "watch" / f"bio_{name}.json"
            run([str(ROOT / "tools" / "bio_probe.py"), str(onnx), "--drills", "walk_1.2,run_3.0,sprint_3.8,spin_2.5",
                 "--out", str(bio_json)])
            if bio_json.exists():
                d = json.loads(bio_json.read_text())["drills"]
                rec["bio"] = {k: {"v": v["speed"], "grf": v["grf_peak_bw_max"], "P": v["power_w"], "clip": v["clip_cm"],
                                  "fell": v["fell"]} for k, v in d.items()}
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    (PARITY / "watch").mkdir(exist_ok=True)
    raise SystemExit(main())
