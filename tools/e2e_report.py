"""One line per E2E run from tools/e2e_runs.jsonl: time, grid fit, each edit's scores and how feedback went."""
import json
from pathlib import Path

runs = [json.loads(l) for l in (Path(__file__).parent / "e2e_runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"{'#':>2} {'name':<14}{'min':>5} {'fit':>3}  edits (followed/identity/style/artifacts, try)          hello")
for k, r in enumerate(runs, 1):
    edits = " ".join(f"{e['score']['followed']}{e['score']['identity']}{e['score']['style']}{e['score']['artifacts']}"
                     f"{'+' if e.get('attempt') else ''}" for e in r["edits"] if e.get("score"))
    print(f"{k:>2} {r['name'][:13]:<14}{r['total'] / 60:>5.1f} {r['grid2']['fit']:>3}  {edits:<52} {r['hello']['mood']}")
