"""G5 — closed-loop comparison CLI: Unity play-mode recording vs golden CPU-MuJoCo reference.
Criteria and lane mapping: poolympic/closed_loop.py.

Usage: uv run python tools/compare_closed_loop.py <reference_name> <unity_record_name> [run_json_path]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poolympic.closed_loop import PARITY, g5_compare, load  # noqa: E402


def main(ref_name: str, uni_name: str, run_path: str | None = None) -> int:
    ref = load(PARITY / f"reference_trajectory_{ref_name}.json")
    uni = load(Path(run_path) if run_path else PARITY / f"unity_run_{uni_name}.json")
    report = g5_compare(ref, uni, ref_name, uni_name)
    out = PARITY / f"g5_report_{uni_name}.json"
    out.write_text(json.dumps(report, indent=1))
    for k, v in report.items():
        if k != "checks":
            print(f"{k:28s} {v}")
    for k, v in report["checks"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("G5", "PASS" if report["G5_pass"] else "FAIL")
    return 0 if report["G5_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
