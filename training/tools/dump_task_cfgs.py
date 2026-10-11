"""Stable hash of every registered task's env cfg + PPO cfg, per body: the proof that a refactor left the tasks alone.

  uv run python tools/dump_task_cfgs.py --out before.json      # all bodies (one subprocess each: a process = one body)
  uv run python tools/dump_task_cfgs.py --diff before.json after.json
  uv run python tools/dump_task_cfgs.py --list                 # current / superseded task IDs of $POOLYMPIC_BODY

Per task: "cfg" = hash of the config tree (train + play env cfg, rl cfg, runner class; functions by NAME, so moving a
function between modules does not change it; lambdas by source text + the plain values they close over), "code" = hash
of the source of every poolympic function / class the tree refers to (changes when such a function is edited, even with
identical configs). "_assets" = sha256 of every MJCF under training/assets.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAIN = (str, int, float, bool, type(None), Path)
_SPECS: dict[str, str] = {}


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _src(f) -> str:
    try:
        return inspect.getsource(f)
    except (OSError, TypeError):
        return getattr(f, "__qualname__", repr(type(f)))


def canon(o, code: dict, seen: tuple = ()):
    """JSON-able canonical form of a config tree; fills `code` with {name: source} of poolympic callables."""
    if isinstance(o, enum.Enum):
        return f"{type(o).__name__}.{o.name}"
    if isinstance(o, Path):
        try:
            return o.resolve().relative_to(ROOT.parent).as_posix()
        except ValueError:
            return o.as_posix()
    if isinstance(o, float):
        return repr(o)
    if isinstance(o, PLAIN):
        return o
    if id(o) in seen:
        return "<cycle>"
    seen = seen + (id(o),)
    if isinstance(o, dict):
        return {str(k): canon(v, code, seen) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
    if isinstance(o, (list, tuple)):
        return [canon(v, code, seen) for v in o]
    if isinstance(o, (set, frozenset)):
        return sorted(str(canon(v, code, seen)) for v in o)
    if isinstance(o, type) or inspect.isfunction(o) or inspect.ismethod(o) or inspect.isbuiltin(o):
        name = getattr(o, "__name__", repr(o))
        if getattr(o, "__module__", "").startswith("poolympic") and name != "<lambda>":
            code[name] = _src(o).strip()
        if name != "<lambda>":
            return f"<{name}>"
        cells = [c.cell_contents for c in (o.__closure__ or ())]
        globs = {n: o.__globals__[n] for n in o.__code__.co_names
                 if n in o.__globals__ and isinstance(o.__globals__[n], PLAIN)}
        key = json.dumps({"lambda": _src(o).strip(), "closure": canon([c for c in cells if isinstance(c, PLAIN)], code, seen),
                          "globals": canon(globs, code, seen)}, sort_keys=True)
        if key not in _SPECS:
            # an entity's spec_fn: hashed by the MJCF it produces, so the lambda may move or be rewritten
            _SPECS[key] = key
            if o.__code__.co_argcount == 0:
                try:
                    spec = o()
                    spec.compile()
                    _SPECS[key] = "mjspec:" + _sha(spec.to_xml())
                except Exception:  # noqa: BLE001 — not a spec factory: keep the source form
                    pass
        return _SPECS[key]
    if dataclasses.is_dataclass(o):
        return {"__class__": type(o).__name__,
                **{f.name: canon(getattr(o, f.name), code, seen) for f in dataclasses.fields(o)}}
    if hasattr(o, "tolist"):
        return canon(o.tolist(), code, seen)
    if hasattr(o, "__dict__"):
        return {"__class__": type(o).__name__, **canon(vars(o), code, seen)}
    return repr(o)


def dump_body() -> dict:
    import poolympic.tasks as T
    from mjlab.tasks.registry import _REGISTRY

    superseded = getattr(T, "SUPERSEDED", frozenset())
    out = {}
    for tid in sorted(t for t in _REGISTRY if t.startswith("PoOlympic-")):
        t = _REGISTRY[tid]
        code: dict = {}
        tree = canon({"env": t.env_cfg, "play": t.play_env_cfg, "rl": t.rl_cfg, "runner": t.runner_cls}, code)
        out[tid] = {"cfg": _sha(json.dumps(tree, sort_keys=True)), "code": _sha(json.dumps(code, sort_keys=True)),
                    "num_envs": t.env_cfg.scene.num_envs, "max_iterations": t.rl_cfg.max_iterations,
                    "save_interval": t.rl_cfg.save_interval, "superseded": tid in superseded}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--one-body", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.diff:
        b, c = (json.loads(Path(p).read_text()) for p in a.diff)
        n = 0
        for body in sorted(set(b) | set(c)):
            tb, tc = b.get(body, {}), c.get(body, {})
            for tid in sorted(set(tb) | set(tc)):
                if tid not in tb:
                    print(f"NEW      {body:8s} {tid}")
                elif tid not in tc:
                    n += 1
                    print(f"REMOVED  {body:8s} {tid}")
                elif tb[tid] != tc[tid]:
                    keys = [k for k in ("cfg", "code") if isinstance(tb[tid], dict) and tb[tid].get(k) != tc[tid].get(k)]
                    if keys or not isinstance(tb[tid], dict):
                        n += "cfg" in keys or not isinstance(tb[tid], dict)
                        print(f"CHANGED  {body:8s} {tid}  ({', '.join(keys) or 'value'})")
        print(f"{n} existing task cfg(s) / asset set(s) changed or removed")
        sys.exit(1 if n else 0)

    if a.one_body:
        print(json.dumps(dump_body()))
        return
    if a.list:
        for tid, v in dump_body().items():
            print(f"{'superseded' if v['superseded'] else 'current   '}  {v['num_envs']:5d} envs  {tid}")
        return

    from poolympic import bodies
    res = {"_assets": {"all": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                               for p in sorted((ROOT / "assets").rglob("*.xml"))}}}
    for name in sorted(bodies.BODIES):
        r = subprocess.run([sys.executable, __file__, "--one-body"], env={**os.environ, "POOLYMPIC_BODY": name},
                           capture_output=True, text=True, cwd=ROOT)
        if r.returncode:
            sys.exit(f"{name}: {r.stderr[-2000:]}")
        res[name] = json.loads(r.stdout.strip().splitlines()[-1])
        print(f"{name}: {len(res[name])} tasks", file=sys.stderr)
    text = json.dumps(res, indent=1, sort_keys=True)
    Path(a.out).write_text(text) if a.out else print(text)


if __name__ == "__main__":
    main()
