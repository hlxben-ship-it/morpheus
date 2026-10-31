"""Run case number i (1-based) of tools/e2e_cases.json through e2e_soul: tools/e2e_case.py 7 [--no-idle]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e2e_soul  # noqa: E402

cases = json.loads((Path(__file__).parent / "e2e_cases.json").read_text(encoding="utf-8"))
i = int(sys.argv[1])
r = e2e_soul.run(*cases[i - 1], idle="--no-idle" not in sys.argv)
r["case"] = i
print(json.dumps(r, ensure_ascii=False, indent=1))
