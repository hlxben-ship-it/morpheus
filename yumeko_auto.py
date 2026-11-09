"""Yumeko flow v2: Phase 3 on autopilot, the review, and the exam after training.

The user picks the look (Phase 1 and 2) and the final set; Claude does the long work in between:
1. Checklist. Claude looks at the bases and writes pass/fail checks plus her tags; the user corrects them in chat.
2. Hardening. Test rounds of 8 pictures, judged against the checklist, the tags revised until about 3 in 4 pass.
3. Building. A coverage plan (close-ups, full body, action, back views, other outfits) rendered and judged picture by picture.
   Passing pictures are captioned with what is actually in them, near-duplicates are dropped.
4. Review. The user hearts or dismisses each picture; a dismissal's comment joins the checklist and the slot is rendered again.
Training then uses short captions with two trigger words and trained text encoders (the Nova2 v5 recipe), and the exam
renders the same scenes with every stage on the trigger alone, so stages are judged against each other fairly.

All state lives in job["auto"]; a server restart leaves the build paused, and Resume carries on where it stopped.
"""
import random
import shutil
import threading
import time

from PIL import Image

import morpheus
import yumeko as y

RUNNING = set()  # jobs whose build thread is alive
PASS_RATE = 0.75  # hardening stops once this share of a test round passes
HARDEN_ROUNDS = 4
JUDGE_BATCH = 8  # candidates per Claude look. Each call re-reads the system prompt and her references, so fuller calls cost less;
# Kiri's build (4 per call) took ~38 judge calls. A contact-sheet pre-sort was weighed and dropped: with ~30% rejects it adds a call.
TRIES = 3  # renders per slot before it is given up
DUPE = 6  # dHash distance at or below which two pictures count as the same picture

CARD_SCHEMA = {"type": "object", "properties": {
    "reply": {"type": "string"}, "checklist": {"type": "array", "items": {"type": "string"}},
    "permanent": {"type": "string"}, "signature": {"type": "string"}, "negative": {"type": "string"}, "keep": {"type": "string"}},
    "required": ["reply", "checklist", "permanent", "signature", "negative", "keep"]}
CARD_SYSTEM = """You are Yumeko, lazy, playful and a little smug, about to build a character's LoRA training set on your own.
Open every listed picture with Read first. Then write:
- checklist: 5 to 9 visual pass/fail checks a reviewer can verify on one picture, most defining first, each naming colour, shape
  and position, e.g. "hair: blonde, very long, high ponytail with thin braids". Cover face, hair, eyes, marks, jewellery and every
  defining piece of the signature outfit.
- permanent: Danbooru tags for what is on her with any clothes: hair, eyes, skin, marks, and the accessories she keeps in every
  outfit (goggles, glasses, a hair ornament, jewellery) once the user has said so. No quality, count or age tags.
- signature: everything else she wears in her signature look. Her other outfits replace all of it, so anything here is
  missing from those pictures.
- signature: Danbooru tags for her signature outfit, head to toe, with colours.
- negative: up to 15 Danbooru tags for concrete wrong things to avoid with her, plus a few general quality negatives.
- reply: two or three sentences. Say what you think defines her, then ask which features matter most. On the first turn, also
  name each accessory in her signature look (goggles, glasses, hat, jewellery, a weapon or tool she carries) and ask which
  of them stay with her in other outfits; move those the user keeps into permanent.
- keep: the tags for what the user called most important, and any tags the user tested or asked for by name. The build can
  never remove these. Empty until the user has said what matters.
Tags are real Danbooru tags only, never invented phrases ("thin red line under left eye", "chrome plate on right temple"): the
model doesn't know them. Use the nearest real tag (facial mark, single mechanical arm, prosthesis) and leave the exact detail to
the checklist. Never a plural that summons two of a thing she has one of (mechanical arms).
She is one adult woman, always SFW. On later turns the user answers or corrects; update every field to match and reply briefly."""
JUDGE_SCHEMA = {"type": "object", "properties": {"results": {"type": "array", "items": {"type": "object", "properties": {
    "index": {"type": "integer"}, "pass": {"type": "boolean"}, "problems": {"type": "string"},
    "framing": {"type": "string", "enum": ["closeup", "upper", "cowboy", "full"]}, "tags": {"type": "string"}},
    "required": ["index", "pass", "problems", "framing", "tags"]}}}, "required": ["results"]}
JUDGE_SYSTEM = """You check candidate training pictures for one character's LoRA. Open the reference pictures and every candidate
with Read before answering, and answer for every candidate.
Judge like a fan who would recognise her at a glance, not like a pixel counter. A candidate passes when every defining item
that is visible is right: hair colour and style, eye colour, the main pieces of the outfit and their colours. Small inaccuracies
in a detail (the colours of beads, trim or piping, a fringe, a buckle, the exact boot style) go into problems but do not fail the
picture, unless the user's own checklist line says that detail is a must. Lighting shifts shades: a lighter or darker shade of
the right hue (navy for blue, auburn for red) is a note, not a fail; a different hue (black for blue, brown for amber) fails. Group "signature": the whole checklist applies.
Protected features (the user's most important, listed in the message) are musts in every group: when the part of her they belong
to is in frame (her face for a face implant, her arm for a mechanical arm) and the feature is missing or plainly different, the
picture fails and problems starts with "missing: <tag>". A feature that is out of frame is not missing.
Group "other": only the items about her body, face, hair, eyes and jewellery; her clothes are meant to differ and must match the
outfit named. Group "any": identity items only.
Always fail: see-through or lingerie-like clothes, a pose or framing centred on the bottom or chest; a different face, hair colour or eye colour; a missing or wrong defining piece; more than one person; broken anatomy
that is clearly visible (wrong finger count, extra or merged limbs); text or watermark; nudity; and, when framing_required is
true, a framing other than the one asked. Small rendering noise is fine.
For each candidate give:
- index, pass, problems (short, empty when it passes)
- framing: what the picture really shows. closeup = head and shoulders, upper = to the waist, cowboy = to mid thigh,
  full = the whole figure including the feet.
- tags: 6 to 14 Danbooru tags for what is really visible and changes between pictures: framing, view angle, pose, action,
  expression, background, lighting, and for group "other" the clothes she wears. Leave out her fixed features and her signature
  outfit; the trigger words learn those."""
FIX_SCHEMA = {"type": "object", "properties": {"reply": {"type": "string"}, "permanent": {"type": "string"},
                                               "signature": {"type": "string"}, "negative": {"type": "string"}},
              "required": ["reply", "permanent", "signature", "negative"]}
FIX_SYSTEM = """You harden a Danbooru tag prompt so an SDXL anime model draws the same character every time. You get her checklist,
her current tags (permanent: body, face, hair, eyes, jewellery; signature: outfit; negative) and the failures of the last test
round or the user's complaints, and the history of earlier rounds with what each change did. Return revised tags.
Rules, learned the hard way:
- Change little per round, two or three tags. A round that got worse means the last change was wrong; you are handed the best
  tags so far, start from those and try something else.
- Use real Danbooru tags. Never invent descriptive phrases, counts or positions ("three beads at ponytail base").
- No framing, view, pose or background tags (cowboy shot, front view, from above, from behind), in the positive or the
  negative: every picture gets its own scene, and a view in the negative makes those slots impossible.
- Avoid words that summon other things: "tail", "ears" (summons animal ears), "horn", "wing", "forehead". Fix ears with a
  negative such as "pointy ears, animal ears" only.
- No duplicates or near-duplicates; permanent stays under 22 tags, signature under 18, negative under 20.
- A detail the model can't hold (bead colours, a logo) is not worth more tags; leave it.
- Never put a garment of her planned other outfits into the negative (you get the list): a complaint about one of them means
  "make it right", not "ban it". Complaints about other outfits are fixed by the outfit's own tags, not hers.
One adult woman, SFW. reply: one sentence naming what you changed."""

SETTINGS = ["simple background, white background", "simple background, grey background", "outdoors, forest, daytime", "city street, daytime",
            "beach, daytime", "indoors, living room", "night, city lights", "mountains, sunset", "castle interior", "rooftop, overcast",
            "field, flowers, blue sky", "library interior"]
FACES = ["smile", "light smile", "serious", "closed mouth", "determined", "smirk", "surprised"]
SIGNATURE_SLOTS = [  # kind, count, framing, scenes; framing_required only where the model tends to cheat
    ("closeup", 8, "closeup", ["portrait, close-up, looking at viewer", "close-up, face focus, looking at viewer", "portrait, three quarter view", "portrait, from side"]),
    ("upper", 8, "upper", ["upper body, looking at viewer", "upper body, three quarter view", "upper body, from side, looking to the side", "upper body, from above, looking up"]),
    ("cowboy", 8, "cowboy", ["cowboy shot, standing, looking at viewer", "cowboy shot, three quarter view, hand on hip", "cowboy shot, from below", "cowboy shot, arms behind back"]),
    ("full", 10, "full", ["full body, standing, wide shot, feet visible, looking at viewer", "full body, walking, wide shot, feet visible", "full body, standing, three quarter view, wide shot, feet visible", "full body, wide shot, from side, feet visible", "full body, contrapposto, wide shot, feet visible"]),
    ("action", 8, "", ["running, dynamic pose, motion lines", "jumping, mid-air, dynamic pose", "fighting stance, dynamic pose", "kneeling, one knee", "sitting, crossed legs", "leaning forward, reaching out", "looking back, turning around, hair flowing", "stretching, arms up"]),
    ("back", 6, "", ["from behind, looking back, upper body", "from behind, full body, wide shot, feet visible", "from side, upper body, wind, hair flowing"]),
]
# Each a complete outfit, underlayer and shoes included: a coat alone over "no usual top" came out nude.
OTHER_OUTFITS = ["grey hoodie, blue jeans, sneakers", "light blue sundress, opaque fabric, sandals", "beige winter coat, closed coat, white turtleneck sweater, scarf, blue jeans",
                 "grey t-shirt, denim shorts, sneakers", "black business suit, white collared shirt, black trousers"]
MODEST = "see-through, lingerie, cleavage, topless, ass focus"  # always in the negative; the judge catches what still slips through
NO_TEXT = "text, english text, sign"  # Akane: the cyberpunk style filled backgrounds with lettered neon signs
# Akane: her signature tags named boots, heels and a katana, and every close-up came out full body. Tight framings get weight
# and a negative against the wide shot.
TIGHT = {"closeup": ("(close-up:1.3), (portrait:1.2), ", "full body, wide shot, feet, shoes, boots, legs, thighs"),
         "upper": ("(upper body:1.2), ", "full body, wide shot, feet, shoes, boots")}
OTHER_SCENES = [("upper", "upper body, looking at viewer"), ("cowboy", "cowboy shot, standing"), ("full", "full body, standing, wide shot, feet visible"), ("", "sitting, relaxed")]
EXAM = [  # (group, scene, seed); signature scenes get the outfit trigger, the rest the plain trigger alone
    ("signature", "upper body, looking at viewer, smile, garden, daytime", 1001),
    ("signature", "full body, standing, simple background, grey background", 1002),
    ("signature", "cowboy shot, running, city street, sunset", 1003),
    ("signature", "from side, fighting stance, forest", 1004),
    ("any", "portrait, close-up, looking at viewer, night sky", 1005),
    ("other", "grey hoodie, blue jeans, sitting, cafe interior", 1006),
]


# ---------- helpers ----------
def _job(job_id):
    return y.load(job_id)


def _update(job_id: str, change) -> dict:
    """Load, change and save under the data lock, so the build thread and the page don't overwrite each other."""
    with y.DATA:
        j = y.load(job_id)
        change(j)
        return y.save(j)


def _log(a: dict, text: str) -> None:
    a.setdefault("log", []).append({"t": time.time(), "text": text})
    a["log"] = a["log"][-60:]


def _see(job: dict, files: list, tag: str) -> list:
    """Copy pictures into Claude's working folder, the only place its Read reaches. Returns the copies."""
    out = []
    for i, f in enumerate(files):
        c = y.CHAT_DIR / f"auto_{job['id'][-8:]}_{tag}_{i}.png"
        shutil.copy(y.ROOT / job["id"] / f, c)
        out.append(c)
    return out


def _references(job: dict) -> list:
    return [b["file"] for b in job["bases"]] + ([job["phase1"]["reference"]] if job["phase1"].get("reference") else [])


def outfit_trigger(job: dict) -> str:
    """The signature outfit's trigger word. Imported jobs may have trained their own (Nova2: nova2_armor)."""
    return (job.get("auto") or {}).get("outfit_trigger") or job["agent"]["trigger_token"] + "_outfit"


def dhash(path) -> int:
    im = Image.open(path).convert("L").resize((17, 16), Image.LANCZOS)
    px = list(im.getdata())
    return sum(1 << i for i in range(256) if px[(i // 16) * 17 + i % 16] > px[(i // 16) * 17 + i % 16 + 1])


def _dupe(job: dict, h: int, own: str) -> bool:
    return any(s.get("hash") is not None and s["id"] != own and s["state"] in ("kept", "approved") and bin(s["hash"] ^ h).count("1") <= DUPE
               for s in job["auto"]["slots"])


def _prompt(job: dict, slot: dict) -> tuple[str, str]:
    a = job["auto"]
    style = y.style_tags(job)
    if slot["group"] == "signature":
        who = a["permanent"] + ", " + a["signature"]
        negative = a["negative"]
    else:
        who = a["permanent"] + ", " + slot["outfit"]
        negative = a["negative"] + ", " + a["signature"]  # stops her usual clothes leaking in
    lead, against = TIGHT.get(slot.get("framing"), ("", ""))
    negative = ", ".join(x for x in [MODEST, NO_TEXT, against, negative] if x)
    return lead + style + y.ADULT + "1girl, solo, " + who + ", " + slot["scene"], negative


def _plan(job: dict, budget: str) -> list:
    share = 0.5 if budget == "quick" else 1.0
    slots = []
    for kind, n, framing, scenes in SIGNATURE_SLOTS:
        for i in range(max(2, round(n * share))):
            scene = f"{scenes[i % len(scenes)]}, {random.choice(FACES)}, {random.choice(SETTINGS)}"
            slots.append({"group": "signature", "kind": kind, "framing": framing, "scene": scene})
    for o in OTHER_OUTFITS[:3 if budget == "quick" else 5]:
        for kind, scene in OTHER_SCENES[:3 if budget == "quick" else 4]:
            slots.append({"group": "other", "outfit": o, "kind": kind or "action", "framing": kind, "scene": f"{scene}, {random.choice(FACES)}, {random.choice(SETTINGS)}"})
    for i, s in enumerate(slots):
        s |= {"id": f"s{i:03}", "state": "todo", "tries": 0, "file": None, "caption": "", "problems": "", "comment": ""}
    return slots


# ---------- Claude ----------
def _card_turn(job: dict, text: str) -> dict:
    a = job["auto"]
    copies = _see(job, _references(job), "ref")
    try:
        context = (f"[Character: {job['agent']['name']}. {job['agent']['identity_summary']}]\n"
                   f"[Base tags: {y.identity_tags(job)}]\n"
                   + (f"[Current card. checklist: {a['checklist']} | permanent: {a['permanent']} | signature: {a['signature']} | negative: {a['negative']}]\n" if a.get("checklist") else "")
                   + "[Conversation so far:\n" + "\n".join(f"{m['role']}: {m['text']}" for m in a.get("chat", [])[-8:]) + "]\n"
                   + "[Pictures of her, open each with Read: " + ", ".join(c.name for c in copies) + "]\n")
        answer, _ = y.ask_claude(context + text, None, ghost=True, see=True, system=CARD_SYSTEM, schema=CARD_SCHEMA)
    finally:
        for c in copies:
            c.unlink(missing_ok=True)
    return answer


def _judge(job: dict, slots: list) -> list:
    """One Claude look at up to JUDGE_BATCH candidates, with her references beside them."""
    a = job["auto"]
    refs = _see(job, job_refs(job), "jref")
    cands = _see(job, [s["file"] for s in slots], "cand")
    try:
        lines = []
        for i, (s, c) in enumerate(zip(slots, cands)):
            group = s["group"] if s["group"] in ("signature", "other", "any") else "signature"
            lines.append(f"{i}: {c.name} | group {group}" + (f" wearing {s['outfit']}" if s.get("outfit") else "")
                         + f" | framing asked: {s.get('framing') or 'any'} | framing_required: {s.get('framing') in ('full', 'closeup')}")
        msg = ("[Checklist:\n- " + "\n- ".join(a["checklist"]) + "]\n"
               + (f"[Protected features, musts: {a['keep']}]\n" if a.get("keep") else "") +
               f"[References of her, open each with Read: {', '.join(r.name for r in refs)}]\n"
               "[Candidates:\n" + "\n".join(lines) + "]\nJudge every candidate.")
        answer, _ = y.ask_claude(msg, None, ghost=True, see=True, system=JUDGE_SYSTEM, schema=JUDGE_SCHEMA)
        a["judge_calls"] = a.get("judge_calls", 0) + 1  # in-memory; record() below persists it with the verdicts
    finally:
        for c in refs + cands:
            c.unlink(missing_ok=True)
    by = {r["index"]: r for r in answer["results"]}
    return [by.get(i, {"pass": False, "problems": "not judged", "framing": "", "tags": ""}) for i in range(len(slots))]


def job_refs(job: dict) -> list:
    """The pictures the judge compares against: the bases, then up to two more views of her signature look, hearted ones first,
    else pictures that passed hardening. They stand in for the second base, which is optional."""
    sig = [s for s in job["auto"]["slots"] if s["group"] == "signature" and s.get("file")]
    more = [s["file"] for s in sig if s["state"] == "approved"] + [s["file"] for s in sig if s["state"] == "kept" and s["kind"] in ("full", "closeup", "back")]
    return [b["file"] for b in job["bases"]] + list(dict.fromkeys(more))[:max(0, 3 - len(job["bases"]))]


def _protect(a: dict) -> None:
    """Tags the user called most important stay: Akane's hardening dropped "mechanical parts on face, chrome, cable" as vague,
    and her chrome cheek implant vanished from the whole set. Missing keep tags go back where they were, never into the negative."""
    keep = [t.strip() for t in (a.get("keep") or "").split(",") if t.strip()]
    if not keep:
        return
    have = {t.strip() for t in (a["permanent"] + "," + a["signature"]).split(",")}
    missing = [t for t in keep if t not in have]
    if missing:
        a["permanent"] = a["permanent"].rstrip(", ") + ", " + ", ".join(missing)
    a["negative"] = ", ".join(t.strip() for t in a["negative"].split(",") if t.strip() and t.strip() not in keep)


def _fix(job: dict, failures: list) -> str:
    a = job["auto"]
    history = "\n".join(f"Round {r['round']}: {r['passed']} of {r['of']} passed" + (f", then changed: {r['changes']}" if r.get("changes") else "") for r in a["rounds"])
    msg = (f"[Checklist:\n- " + "\n- ".join(a["checklist"]) + "]\n" + (f"[History:\n{history}]\n" if history else "") +
           f"[permanent: {a['permanent']}]\n[signature: {a['signature']}]\n[negative: {a['negative']}]\n"
           + (f"[Protected, the user's most important tags, never remove them: {a['keep']}]\n" if a.get("keep") else "") +
           f"[Her planned other outfits: {' | '.join(sorted({x['outfit'] for x in a['slots'] if x.get('outfit')}))}]\n"
           "[Failures:\n- " + "\n- ".join(failures[:16]) + "]\nRevise the tags.")
    answer, _ = y.ask_claude(msg, None, ghost=True, see=False, system=FIX_SYSTEM, schema=FIX_SCHEMA)
    a["permanent"], a["signature"], a["negative"] = answer["permanent"], answer["signature"], answer["negative"]
    _protect(a)
    return answer["reply"]


# ---------- actions ----------
def open_card(req: dict) -> dict:
    """Start of Phase 3 on autopilot: Claude reads the bases and proposes the checklist and her tags."""
    job = y.load(req.get("id"))
    if job["phase"] != 3:
        raise ValueError("This job isn't building its set.")
    if job.get("auto"):
        return job
    job["auto"] = {"state": "card", "chat": [], "checklist": [], "permanent": "", "signature": "", "negative": "", "slots": [], "rounds": [], "log": [], "budget": "full"}

    def work(j):
        answer = _card_turn(j, "Look at her and set up the card.")
        a = j["auto"]
        a.update({k: answer[k] for k in ("checklist", "permanent", "signature", "negative", "keep")})
        a["chat"].append({"role": "yumeko", "text": answer["reply"]})
        return j
    return y._background(job, "Yumeko is studying her", work)


def card_chat(req: dict) -> dict:
    job = y.load(req.get("id"))
    a = job.get("auto")
    if not a or a["state"] != "card":
        raise ValueError("The card is settled; the build is running.")
    text = str(req.get("text", "")).strip()
    if not text:
        raise ValueError("Say something first.")
    a["chat"].append({"role": "user", "text": text})

    def work(j):
        answer = _card_turn(j, text)
        b = j["auto"]
        b.update({k: answer[k] for k in ("checklist", "permanent", "signature", "negative", "keep")})
        b["chat"].append({"role": "yumeko", "text": answer["reply"]})
        return j
    return y._background(job, "Yumeko is thinking", work)


def card_edit(req: dict) -> dict:
    """The user's own corrections to the card, field by field."""
    def change(j):
        a = j["auto"]
        if a["state"] != "card":
            raise ValueError("The card is settled; the build is running.")
        if isinstance(req.get("checklist"), list):
            a["checklist"] = [str(c).strip() for c in req["checklist"] if str(c).strip()]
        for k in ("permanent", "signature", "negative", "keep"):
            if k in req:
                a[k] = str(req[k]).strip()
        if req.get("budget") in ("quick", "full"):
            a["budget"] = req["budget"]
    return _update(req.get("id"), change)


def go(req: dict) -> dict:
    """The card is right: harden, then build, on its own."""
    def change(j):
        a = j["auto"]
        if a["state"] != "card":
            raise ValueError("Already building.")
        if len(a["checklist"]) < 3 or not a["permanent"] or not a["signature"]:
            raise ValueError("The card needs a checklist and her tags first.")
        a["state"], a["slots"] = "hardening", _plan(j, a.get("budget", "full"))
        _log(a, f"Plan: {len(a['slots'])} pictures")
    job = _update(req.get("id"), change)
    start(job["id"])
    return job


def pause(req: dict) -> dict:
    def change(j):
        j["auto"]["paused"] = True
    return _update(req.get("id"), change)


def resume(req: dict) -> dict:
    def change(j):
        j["auto"]["paused"] = False
    job = _update(req.get("id"), change)
    start(job["id"])
    return job


def verdict(req: dict) -> dict:
    """Review: heart keeps a picture for training; dismiss with a comment sends the slot back to be rendered again."""
    sid, keep, comment = req.get("slot"), req.get("keep") is True, str(req.get("comment", "")).strip()
    if not keep and not comment:
        raise ValueError("Say in a few words what's wrong, so I can fix it in the rest too.")

    def change(j):
        a = j["auto"]
        s = next((s for s in a["slots"] if s["id"] == sid), None)
        if not s or s["state"] not in ("kept", "approved"):
            raise ValueError("That picture isn't up for review.")
        if keep:
            s["state"] = "approved"
            return
        old = y.ROOT / j["id"] / s["file"]
        s.update(state="todo", tries=0, comment=comment, file=None, hash=None, caption="", replaced=s.get("replaced", 0) + 1)
        old.unlink(missing_ok=True)
        a.setdefault("comments", []).append(comment)
        item = f"Not: {comment}"
        if item not in a["checklist"]:
            a["checklist"].append(item)
        a["state"] = "building"
        _log(a, f"Dismissed {sid}: {comment}")
    job = _update(req.get("id"), change)
    if not keep:
        start(job["id"])
    return job


def second_view(req: dict) -> dict:
    """Optional: from the card, go back to add a second base, for a back or a detail the first doesn't show."""
    def change(j):
        if j["phase"] != 3 or (j.get("auto") or {}).get("state", "card") != "card" or len(j["bases"]) > 1:
            raise ValueError("A second view can only be added before the build starts.")
        j["phase"] = 2
        if j.get("auto"):
            j["auto"]["second_pending"] = True
    return _update(req.get("id"), change)


def retag(req: dict) -> dict:
    """After the build: change her tags and redraw only the other outfits, e.g. once the user decides her goggles stay on in
    every outfit. Signature pictures keep their hearts."""
    def change(j):
        a = j["auto"]
        if a["state"] not in ("review", "building"):
            raise ValueError("The set isn't built yet; change the card instead.")
        for k in ("permanent", "signature"):
            if str(req.get(k, "")).strip():
                a[k] = str(req[k]).strip()
        n = 0
        for s in a["slots"]:
            if s["group"] == "other" and s["state"] in ("kept", "approved", "skipped"):
                if s.get("file"):
                    (y.ROOT / j["id"] / s["file"]).unlink(missing_ok=True)
                s.update(state="todo", tries=0, file=None, hash=None, caption="")
                n += 1
        a["state"], a["paused"] = "building", False
        _log(a, f"Her tags changed, redrawing {n} other-outfit pictures")
    job = _update(req.get("id"), change)
    start(job["id"])
    return job


def train(req: dict) -> dict:
    """Every picture has a heart: they become the dataset, captioned with what is in them, and training starts."""
    job = y.load(req.get("id"))
    a = job.get("auto") or {}
    if a.get("state") != "review":
        raise ValueError("The set isn't ready for training.")
    approved = [s for s in a["slots"] if s["state"] == "approved"]
    if [s for s in a["slots"] if s["state"] == "kept"]:
        raise ValueError("Give every picture a heart or a dismissal first.")
    if len(approved) < y.MIN_SET:
        raise ValueError(f"Needs {y.MIN_SET} approved pictures, there are {len(approved)}.")
    trigger, ot = job["agent"]["trigger_token"], outfit_trigger(job)
    with y.DATA:
        job = y.load(job["id"])
        job["dataset"]["accepted"] = [{"file": s["file"], "group": s["group"], "planned": [], "tags": y._split(s["caption"]),
                                       "caption": ", ".join([trigger] + ([ot] if s["group"] == "signature" else []) + y._split(s["caption"]))} for s in approved]
        job["dataset"]["batch"] = []
        y.save(job)
    return y.start_training({"id": job["id"]})


def exam(req: dict) -> dict:
    """The same six scenes for every stage, on her trigger words and permanent tags, judged against the checklist."""
    job = y.load(req.get("id"))
    stopped = (job.get("train") or {}).get("state") == "stopped" and y.stages_of_saved(job)
    if job["phase"] != 5 and not stopped:
        raise ValueError("Training isn't finished.")
    if stopped:  # a stopped run goes back to review; examining its saved stages moves it on like a finished one
        job["phase"] = 5
        y.save(job)

    def work(j):
        result, shown = {}, {}
        for stage in y.stages_of(j):
            lora = y._stage_file(j, stage)
            files = []
            for k, (group, scene, seed) in enumerate(EXAM):
                j["busy"] = f"Exam, stage {stage}, picture {k + 1} of {len(EXAM)}"
                y.save(j)
                # trigger plus her permanent tags, as Use sends them: the trigger alone drifted on Kiri (navy hair, bigger chest)
                words = j["agent"]["trigger_token"] + (", " + outfit_trigger(j) + ", " + j["auto"]["signature"] if group == "signature" else "") + ", " + j["auto"]["permanent"]
                name = f"exam_{stage}_{k}.png"
                y.render(words + ", " + y.style_tags(j) + y.ADULT + "1girl, solo, " + scene, j["auto"]["negative"], seed, loras=[(lora, 1.0)]).save(y.ROOT / j["id"] / name)
                files.append(name)
            shown[stage] = files
        # Judge each scene's stage versions side by side in one look: judged per stage, near-identical pictures got opposite
        # verdicts on Kiri (6 of 6 against 0 of 6 on the same hair).
        verdicts = {stage: [None] * len(EXAM) for stage in shown}
        for k, (g, scene, _) in enumerate(EXAM):
            j["busy"] = f"Judging scene {k + 1} of {len(EXAM)}"
            y.save(j)
            outfit = scene.split(",")[0] + ", " + scene.split(",")[1] if g == "other" else ""
            slots = [{"file": shown[stage][k], "group": g, "framing": "", "outfit": outfit} for stage in shown]
            for stage, v in zip(shown, _judge(j, slots)):
                verdicts[stage][k] = v
        for stage, files in shown.items():
            vs = verdicts[stage]
            result[str(stage)] = {"files": files, "passed": sum(v["pass"] for v in vs), "problems": [v["problems"] for v in vs]}
        j["exam"] = {"stages": result, "suggested": max(result, key=lambda s: (result[s]["passed"], int(s))), "of": len(EXAM)}
        return j
    return y._background(job, "Exam", work)


def keep_stage(req: dict) -> dict:
    """Choosing a stage writes her card with the full tags as a fallback, then calibrates the fewest identity tags in the background."""
    job = y.choose_stage(req)
    if job.get("auto"):
        def change(j):
            j["look"] = {"positive": outfit_trigger(j), "negative": "", "identity": j["auto"]["permanent"], "identity_negative": j["auto"]["negative"],
                         "outfit": j["auto"]["signature"], "outfit_negative": ""}
            j["lora"]["recipe"] = "v2"
        job = _update(job["id"], change)
        job = calibrate({"id": job["id"]})
    return job


# ---------- identity calibration ----------
CALIB_ROUNDS = 8
CALIB_SCHEMA = {"type": "object", "properties": {"reply": {"type": "string"}, "identity": {"type": "string"}, "identity_negative": {"type": "string"},
                                                 "outfit": {"type": "string"}, "outfit_negative": {"type": "string"}},
                "required": ["reply", "identity", "identity_negative", "outfit", "outfit_negative"]}
CALIB_SYSTEM = """A character LoRA is trained. Her trigger word carries most of her look, but some features drift when the trigger is
used alone. Your job: find the FEWEST Danbooru tags that, added silently to every render, hold her look. Four sets:
- identity: positive tags for body, face, hair, eyes that drift (typically hair colour, eye colour, body size). Sent with every render.
- identity_negative: negative tags against the observed wrong versions (e.g. "brown hair, black hair" when blue hair drifts).
- outfit: positive tags for signature outfit pieces that go missing when her outfit trigger is on.
- outfit_negative: negatives against wrong versions of her signature outfit; only sent with the outfit trigger.
You get her checklist, her full tag card (for wording), the current four sets, the exam failures of the current sets, and the
history of earlier rounds. Rules:
- Fewer is better. Add a tag only against a drift seen in the failures. Each set stays under 8 tags.
- When the current sets passed everything, try a smaller set: drop the one to three tags most likely redundant.
- When a smaller set failed, start from the last set that passed and drop something else, or stop shrinking.
- Real Danbooru tags only. No framing, pose, scene or style tags. Identity sets must suit any outfit: never a garment in
  identity or identity_negative.
- Leave out modesty negatives (cleavage, see-through, lingerie): those are always added separately.
- Cleavage, a chest-centred framing or a bigger bust means her body drifted, not her clothes: add her body size from the card
  (e.g. small breasts) to identity and the opposite (large breasts) to identity_negative before touching necklines.
- Keep what worked: start from the best round so far and change two or three tags, never rewrite the sets.
reply: one sentence naming what you changed."""


def _calib_round(j: dict, sets: dict, lora: str, tag: str) -> list:
    """Render the exam scenes with the trigger plus the given sets and judge them in one look."""
    t = j["agent"]["trigger_token"]
    slots = []
    for k, (group, scene, seed) in enumerate(EXAM):
        sig = group == "signature"
        pos = ", ".join(x for x in [t, outfit_trigger(j) if sig else "", sets["identity"], sets["outfit"] if sig else ""] if x)
        neg = ", ".join(x for x in [MODEST, sets["identity_negative"], sets["outfit_negative"] if sig else ""] if x)
        name = f"calib_{tag}_{k}.png"
        j["busy"] = f"Calibrating identity, round {int(tag) + 1}, picture {k + 1} of {len(EXAM)}"
        y.save(j)
        y.render(pos + ", " + y.style_tags(j) + y.ADULT + "1girl, solo, " + scene, neg, seed, loras=[(lora, 1.0)]).save(y.ROOT / j["id"] / name)
        outfit = scene.split(",")[0] + ", " + scene.split(",")[1] if group == "other" else ""
        slots.append({"file": name, "group": group, "framing": "", "outfit": outfit})
    j["busy"] = f"Calibrating identity, judging round {int(tag) + 1}"
    y.save(j)
    return [dict(v, file=s["file"]) for s, v in zip(slots, _judge(j, slots))]


def _ntags(sets: dict) -> int:
    return sum(len([x for x in v.split(",") if x.strip()]) for v in sets.values())


def calibrate(req: dict) -> dict:
    """Find the fewest identity and outfit tags (positive and negative) that hold her look on her trigger words.
    Round 1 is the trigger alone; Claude adds tags against what drifted, and once a round passes every scene it tries smaller
    sets. The smallest set that passed everything wins, else the best round. Writes look.identity, identity_negative, outfit,
    outfit_negative, which Use adds silently while the Identity pill is on."""
    job = y.load(req.get("id"))
    if not job.get("lora") or not job.get("auto"):
        raise ValueError("Pick a stage first.")
    n = len(EXAM)

    def work(j):
        a, lora = j["auto"], j["lora"]["file"]
        sets = {"identity": "", "identity_negative": "", "outfit": "", "outfit_negative": ""}
        rounds = []
        for r in range(CALIB_ROUNDS):
            vs = _calib_round(j, sets, lora, str(r))
            passed = sum(v["pass"] for v in vs)
            rounds.append({"round": r + 1, "sets": dict(sets), "passed": passed, "tags": _ntags(sets), "problems": [v["problems"] for v in vs], "files": [v["file"] for v in vs]})
            j["calib"] = {"rounds": rounds, "of": n, "done": False}
            y.save(j)
            full = [x for x in rounds if x["passed"] == n]
            if r == CALIB_ROUNDS - 1:
                break
            if full and sum(x["passed"] < n for x in rounds[rounds.index(full[0]):]) >= 2:
                break  # two shrink attempts failed after the first full pass: the smallest passing set is final
            history = "\n".join(f"Round {x['round']}: {x['passed']} of {n} passed with {x['tags']} tags: {x['sets']}" + (f" -> {x['changes']}" if x.get("changes") else "") for x in rounds)
            fails = [f"scene {k + 1} ({EXAM[k][0]}: {EXAM[k][1]}): {p}" for k, p in enumerate(rounds[-1]["problems"]) if p]
            msg = ("[Checklist:\n- " + "\n- ".join(a["checklist"]) + "]\n"
                   f"[Her full card. permanent: {a['permanent']} | signature: {a['signature']} | negative: {a['negative']}]\n"
                   f"[History:\n{history}]\n[Current sets: {sets}]\n"
                   f"[Last round: {passed} of {n} passed. Notes and problems:\n- " + ("\n- ".join(fails) or "none") + "]\n"
                   + ("Every scene passed: try a smaller set." if passed == n else "Fix what drifted with the fewest tags."))
            answer, _ = y.ask_claude(msg, None, ghost=True, see=False, system=CALIB_SYSTEM, schema=CALIB_SCHEMA)
            sets = {k: answer[k].strip().strip(",").strip() for k in sets}
            rounds[-1]["changes"] = answer["reply"]
        full = [x for x in rounds if x["passed"] == n]
        best = min(full, key=lambda x: (x["tags"], -x["round"])) if full else max(rounds, key=lambda x: (x["passed"], -x["tags"]))
        j["look"] = (j.get("look") or {}) | best["sets"]
        j["calib"] = {"rounds": rounds, "of": n, "done": True, "best": best["round"]}
        return j
    return y._background(job, "Calibrating identity", work)


# ---------- the build thread ----------
def start(job_id: str) -> None:
    if job_id in RUNNING:
        return
    RUNNING.add(job_id)
    threading.Thread(target=_run, args=(job_id,), daemon=True).start()


def _render_slot(job_id: str, slot: dict) -> str:
    j = y.load(job_id)
    positive, negative = _prompt(j, slot)
    seed = random.randrange(2 ** 31)
    name = f"auto_{time.time_ns():x}.png"
    with y.GPU:
        y.ensure_comfy()
        y.render(positive, negative, seed).save(y.ROOT / job_id / name)
    slot.update(seed=seed, positive=positive, negative=negative)
    return name


def _stop(job_id: str) -> bool:
    j = y.load(job_id)
    return bool(j["auto"].get("paused")) or y.training_now() or j["phase"] != 3


def _run(job_id: str) -> None:
    try:
        while not _stop(job_id):
            j = y.load(job_id)
            state = j["auto"]["state"]
            if state == "hardening":
                _harden_round(job_id)
            elif state == "building":
                if not _build_step(job_id):
                    def done(j):
                        j["auto"]["state"], j["auto"]["reviewed"], j["auto"]["now"] = "review", True, ""
                        _log(j["auto"], "Set built, ready for review")
                    _update(job_id, done)
            else:
                break
    except Exception as e:
        def failed(j):
            j["auto"]["paused"] = True
            j["error"] = f"Autopilot stopped: {e}"
            _log(j["auto"], f"Stopped: {e}")
        _update(job_id, failed)
    finally:
        RUNNING.discard(job_id)


def _harden_round(job_id: str) -> None:
    j = y.load(job_id)
    r = len(j["auto"]["rounds"]) + 1
    tests = [{"group": "signature", "kind": kind, "framing": framing, "scene": f"{random.choice(scenes)}, {random.choice(FACES)}, {random.choice(SETTINGS)}", "id": f"h{r}_{i}"}
             for i, (kind, _, framing, scenes) in enumerate(SIGNATURE_SLOTS + SIGNATURE_SLOTS[3:5])]
    for i, t in enumerate(tests):
        if _stop(job_id):
            return
        t["file"] = _render_slot(job_id, t)
        _update(job_id, lambda j: j["auto"].__setitem__("now", f"Test round {r}, picture {i + 1} of {len(tests)}"))
    _update(job_id, lambda j: j["auto"].__setitem__("now", f"Judging test round {r}"))
    j = y.load(job_id)
    verdicts = []
    for i in range(0, len(tests), JUDGE_BATCH):
        verdicts += _judge(j, tests[i:i + JUDGE_BATCH])
    passed = sum(v["pass"] for v in verdicts)
    failures = [v["problems"] for v in verdicts if not v["pass"]]
    calls = j["auto"].get("judge_calls", 0)

    def record(j):
        a = j["auto"]
        a["judge_calls"] = max(a.get("judge_calls", 0), calls)
        a["rounds"].append({"round": r, "passed": passed, "of": len(tests), "files": [t["file"] for t in tests], "problems": [v["problems"] for v in verdicts],
                            "tags": {f: a[f] for f in ("permanent", "signature", "negative")}})
        best = max(a["rounds"], key=lambda x: x["passed"])
        if passed < best["passed"]:  # the last change made it worse: back to the best tags before trying again
            a.update(best["tags"])
            _log(a, f"Round {r} was worse than round {best['round']}, back to its tags")
        _log(a, f"Test round {r}: {passed} of {len(tests)} pass")
        # perfect test pictures go straight into the set, into an open slot of the same kind
        for t, v in zip(tests, verdicts):
            if v["pass"]:
                s = next((s for s in a["slots"] if s["state"] == "todo" and s["group"] == "signature" and s["kind"] == t["kind"]), None)
                h = dhash(y.ROOT / j["id"] / t["file"])
                if s and not _dupe(j, h, s["id"]):
                    s.update(state="kept", file=t["file"], hash=h, caption=v["tags"], scene=t["scene"], positive=t.get("positive"), seed=t.get("seed"))
        if passed / len(tests) >= PASS_RATE or r >= HARDEN_ROUNDS:
            a["state"] = "building"
            _log(a, "Prompt holds, building the set" if passed / len(tests) >= PASS_RATE else f"Still {passed} of {len(tests)} after {r} rounds; building anyway, the judge filters")
    j = _update(job_id, record)
    if j["auto"]["state"] == "hardening":
        _update(job_id, lambda j: j["auto"].__setitem__("now", "Revising her tags"))
        j = y.load(job_id)
        reply = _fix(j, failures)

        def revised(k):
            for f in ("permanent", "signature", "negative"):
                k["auto"][f] = j["auto"][f]
            k["auto"]["rounds"][-1]["changes"] = reply
            _log(k["auto"], f"Revised: {reply}")
        _update(job_id, revised)


def _build_step(job_id: str) -> bool:
    """Render and judge up to JUDGE_BATCH open slots. False when nothing is left to do."""
    j = y.load(job_id)
    a = j["auto"]
    comments = a.get("comments", [])
    if len(comments) > a.get("fixed_comments", 0) and len(comments) - a.get("fixed_comments", 0) >= 2:  # repeated complaints harden the prompt too
        reply = _fix(j, [f"The user rejected a picture: {c}" for c in comments[a.get('fixed_comments', 0):]])

        def revised(k):
            for f in ("permanent", "signature", "negative"):
                k["auto"][f] = j["auto"][f]
            k["auto"]["fixed_comments"] = len(comments)
            _log(k["auto"], f"Revised from your comments: {reply}")
        _update(job_id, revised)
        j = y.load(job_id)
        a = j["auto"]
    todo = [s for s in a["slots"] if s["state"] == "todo" and s["tries"] < TRIES][:JUDGE_BATCH]
    if not todo:
        return False
    done = sum(s["state"] in ("kept", "approved") for s in a["slots"])
    for i, s in enumerate(todo):
        if _stop(job_id):
            return True
        _update(job_id, lambda k: k["auto"].__setitem__("now", f"Rendering, {done} of {len(a['slots'])} in the set"))
        s["file"] = _render_slot(job_id, s)
    _update(job_id, lambda k: k["auto"].__setitem__("now", f"Judging, {done} of {len(a['slots'])} in the set"))
    jj = y.load(job_id)
    for s in todo:  # a near-copy of a kept picture is out anyway; don't spend a Claude look on it
        s["hash"] = dhash(y.ROOT / jj["id"] / s["file"])
        s["dupe"] = _dupe(jj, s["hash"], s["id"])
    fresh = [s for s in todo if not s["dupe"]]
    judged = iter(_judge(jj, fresh) if fresh else [])
    verdicts = [{"pass": False, "problems": "", "framing": "", "tags": ""} if s["dupe"] else next(judged) for s in todo]
    calls = jj["auto"].get("judge_calls", 0)

    def record(k):
        b = k["auto"]
        b["judge_calls"] = max(b.get("judge_calls", 0), calls)
        for s, v in zip(todo, verdicts):
            live = next(x for x in b["slots"] if x["id"] == s["id"])
            live["tries"] += 1
            h = s["hash"]
            framing_ok = s.get("framing") not in ("full", "closeup") or v["framing"] == s["framing"]
            if v["pass"] and framing_ok and not _dupe(k, h, s["id"]):
                live.update(state="kept", file=s["file"], hash=h, caption=v["tags"], seed=s["seed"], positive=s["positive"], problems="", notes=v["problems"])
            else:
                why = v["problems"] or ("framing: " + v["framing"] if not framing_ok else "too close to a picture already in the set")
                live.update(problems=why)
                b.setdefault("rejected", 0)
                b["rejected"] += 1
                (y.ROOT / k["id"] / s["file"]).unlink(missing_ok=True)
                if live["tries"] >= TRIES:
                    live["state"] = "skipped"
                    _log(b, f"Gave up on {live['id']} ({live['kind']}): {why}")
    _update(job_id, record)
    return True


def view(job: dict) -> dict:
    """What the page needs beyond job.json: whether the thread is alive."""
    a = job.get("auto")
    if a:
        a["running"] = job["id"] in RUNNING
    return job


def resume_all() -> None:
    """After a server restart builds are left paused; nothing restarts on its own."""
    if y.ROOT.exists():
        for d in y.ROOT.iterdir():
            if (d / "job.json").exists():
                j = y.load(d.name)
                if j.get("auto", {}).get("state") in ("hardening", "building") and not j["auto"].get("paused"):
                    def change(k):
                        k["auto"]["paused"] = True
                    _update(d.name, change)


ACTIONS = {"auto_open": open_card, "auto_chat": card_chat, "auto_card": card_edit, "auto_go": go, "auto_pause": pause, "auto_resume": resume,
           "auto_verdict": verdict, "auto_retag": retag, "auto_second": second_view, "auto_train": train, "auto_exam": exam, "auto_stage": keep_stage, "auto_calibrate": calibrate}
