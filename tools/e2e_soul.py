"""End-to-end Soulforge runs through the live server: one agent per description, the way a user makes one.

A judge (Claude, looking at the pictures) plays the user: it picks from the grid, asks for one refinement, locks a
candidate in, asks for a clothing change and a detail, then the agent is brought to life and greeted. Each step is
timed and the edits are scored, one JSON line per agent in tools/e2e_runs.jsonl.

    .venv/Scripts/python.exe tools/e2e_soul.py "a grumpy dwarf blacksmith who reviews code" [--no-idle]
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import morpheus  # noqa: E402
import soulforge  # noqa: E402

API = "http://127.0.0.1:5196"
SEEN = ROOT / "tools" / "e2e_seen"
LOG = ROOT / "tools" / "e2e_runs.jsonl"


def post(path: str, data: dict) -> dict:
    req = urllib.request.Request(API + path, data=json.dumps({**data, "background": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        job = json.loads(r.read())["job"]
    while True:
        time.sleep(2)
        with urllib.request.urlopen(f"{API}/api/job/{job}") as r:
            state = json.loads(r.read())
        if state["state"] != "running":
            if state["state"] == "error":
                raise RuntimeError(state["error"])
            return state["result"]


def claude(prompt: str, schema: dict, system: str) -> dict:
    cmd = [shutil.which("claude") or "claude", "-p", "--output-format", "json", "--json-schema", json.dumps(schema),
           "--tools", "Read", "--model", "sonnet", "--no-session-persistence", "--system-prompt", system]
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", cwd=SEEN, timeout=240,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    out = json.loads(p.stdout)
    if not out.get("structured_output"):
        raise RuntimeError(str(out.get("result"))[:300])
    return out["structured_output"]


USER = """You play a demanding but fair user of Soulforge, an app that draws pixel-art avatars for AI agents. You look at
pictures with the Read tool and answer as that user would, in short chat messages."""

PICK = {"type": "object", "properties": {
    "best": {"type": "integer"}, "why": {"type": "string"}, "fit": {"type": "integer"},
    "problems": {"type": "array", "items": {"type": "string"}}, "message": {"type": "string"}},
    "required": ["best", "why", "fit", "problems", "message"]}

SCORE = {"type": "object", "properties": {
    "followed": {"type": "integer"}, "identity": {"type": "integer"}, "style": {"type": "integer"},
    "artifacts": {"type": "integer"}, "notes": {"type": "string"}, "feedback": {"type": "string"}},
    "required": ["followed", "identity", "style", "artifacts", "notes", "feedback"]}

ALIVE = {"type": "object", "properties": {
    "in_character": {"type": "integer"}, "mood_fits": {"type": "integer"}, "notes": {"type": "string"}},
    "required": ["in_character", "mood_fits", "notes"]}


def sheet(agent: dict) -> str:
    f = SEEN / f"{agent['slug']}_g{len(agent['grids'])}.png"
    soulforge.sheet(agent["grids"][-1], f)
    return f.name


def picture(ref: str, name: str) -> str:
    shutil.copy(morpheus.OUTPUT / ref, SEEN / name)
    return name


def judge_grid(agent: dict, description: str, ask: str) -> dict:
    return claude(f"You asked for: {description}\nThe app made {agent['name']}: {agent['soul']}\n"
                  f"Grid of 9 numbered candidates: {sheet(agent)}. Look at it.\n"
                  f"best: the candidate number that best fits the request and is the cleanest picture (clearly an adult, whole head, "
                  f"face clear, no extra people, no cut-off hair). fit: 1-5 how well the best one fits. problems: what is "
                  f"wrong across the grid (short). message: {ask}", PICK, USER)


def judge_edit(before: str, after: str, request: str) -> dict:
    return claude(f"You asked for this change: \"{request}\". Before: {before}. After: {after}. Look at both.\n"
                  "Score 1-5 each: followed (the asked change is clearly there), identity (same person: face, hair, "
                  "pose), style (still the same crisp pixel art, no smooth or glossy patch), artifacts (5 = no seams, "
                  "smears, doubled parts or broken anatomy). notes: one or two sentences, concrete. feedback: if any score is 3 or "
                  "lower, the one short chat message you would send next to get it fixed (as a user would write it, "
                  "e.g. 'no, the coat should be longer and darker'); otherwise empty.", SCORE, USER)


def run(description: str, clothing: str, detail: str, idle: bool = True) -> dict:
    SEEN.mkdir(parents=True, exist_ok=True)
    rec = {"description": description, "t0": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}

    def step(name, fn):
        t = time.time()
        out = fn()
        rec["steps"][name] = round(time.time() - t, 1)
        return out

    agent = step("new", lambda: post("/api/soul/new", {"description": description}))
    rec["slug"], rec["name"], rec["look"] = agent["slug"], agent["name"], agent["look"]
    j1 = judge_grid(agent, description, "one chat message: name the number you like and one concrete change to it "
                                        "(e.g. 'like 4 but with a longer coat'), or ask for a fresh grid if none fits.")
    rec["grid1"] = j1
    agent = step("refine", lambda: post("/api/soul/chat", {"agent": agent["slug"], "message": j1["message"]}))
    rec["refine_reply"] = agent["messages"][-1]["text"]
    j2 = judge_grid(agent, description, "'take N' with the number you lock in.")
    rec["grid2"] = j2
    agent = step("lock", lambda: post("/api/soul/chat", {"agent": agent["slug"], "message": j2["message"]}))
    if not agent.get("base"):  # the designer asked back instead of locking in: the click does it
        agent = post("/api/soul/chat", {"agent": agent["slug"], "message": "Lock it in", "base": j2["best"]})
    rec["locked"] = bool(agent.get("base"))
    rec["edits"] = []
    for request in (clothing, detail):
        before = agent["base"]["file"]
        say, asked = request, request
        for attempt in range(2):  # the request, then at most one round of feedback on it
            agent = step(f"edit:{request[:24]}:{attempt}", lambda: post("/api/soul/chat", {"agent": agent["slug"], "message": say}))
            last = agent["messages"][-1]
            e = {"request": say, "reply": last["text"], "instruction": (last.get("edit") or {}).get("instruction"),
                 "attempt": attempt}
            if not last.get("edit"):
                rec["edits"].append(e)
                break
            n = len(rec["edits"])
            # feedback is judged against the picture before the first try: a redo starts from there
            e["score"] = judge_edit(picture(before, f"{agent['slug']}_e{n}a.png"),
                                    picture(last["edit"]["after"], f"{agent['slug']}_e{n}b.png"), asked)
            e["after"] = last["edit"]["after"]
            e["edits_now"] = len(agent.get("edits", []))
            rec["edits"].append(e)
            if not e["score"]["feedback"].strip():
                break
            say = e["score"]["feedback"]
    if idle:
        agent = step("idle", lambda: post("/api/soul/chat", {"agent": agent["slug"], "message": "Bring her to life", "idle": True}))
        rec["frames"] = sorted((agent.get("idle") or {}).get("frames", {}))
    agent = step("hello", lambda: post("/api/soul/chat", {"agent": agent["slug"], "message": "Hi! How are you today?"}))
    last = agent["messages"][-1]
    rec["hello"] = {"text": last["text"], "mood": last.get("mood"), "edited": bool(last.get("edit"))}
    rec["total"] = round(sum(rec["steps"].values()), 1)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("description")
    ap.add_argument("--clothing", default="Change the outfit to a long dark green travelling coat.")
    ap.add_argument("--detail", default="Add small round glasses.")
    ap.add_argument("--no-idle", action="store_true")
    a = ap.parse_args()
    print(json.dumps(run(a.description, a.clothing, a.detail, not a.no_idle), ensure_ascii=False, indent=1))
