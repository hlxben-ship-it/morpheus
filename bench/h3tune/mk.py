"""Write a variant file: python mk.py <name> <model> <clip> '<json of extra keys>' ["note"]   (ref defaults to base_<model>_<clip>)"""
import json, sys
from pathlib import Path
name, model, clip = sys.argv[1:4]
extra = json.loads(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else {}
v = {"name": name, "model": model, "clip": clip, "ref": f"base_{model}_{clip}", "note": sys.argv[5] if len(sys.argv) > 5 else "", **extra}
if name == v["ref"]: v.pop("ref")
d = Path(__file__).parent / "variants"; d.mkdir(exist_ok=True)
(d / f"{name}.json").write_text(json.dumps(v, indent=1), encoding="utf-8")
print(name)
