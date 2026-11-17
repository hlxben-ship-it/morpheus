# What runs between the transformer forwards of the denoise loop? Samples the stack of the thread that calls the model every 50 ms
# while it is NOT inside MiniMaxH3Model.forward (after the first forward, before the last one ends).
import sys as _sys, threading as _th, time as _time, collections as _c, traceback as _tb
from models.minimax_h3 import transformer as _tr
_o = _tr.MiniMaxH3Model.forward; _s = {"in": False, "tid": None, "n": 0, "last_end": None, "gap": 0.0}; _hits = _c.Counter(); _pre = _c.deque(maxlen=400)
def _fw(self, *a, **k):
    _s["tid"] = _th.get_ident()
    if _s["last_end"] is not None: _s["gap"] += _time.perf_counter() - _s["last_end"]
    if not _s["n"] % 4:   # first forward of a job: keep the samples of the 15 s before it
        now = _time.perf_counter()
        for t, key in _pre:
            if now - t < 15: _hits["[before first] " + key] += 1
        _pre.clear()
    _s["in"] = True
    try: return _o(self, *a, **k)
    finally: _s["in"] = False; _s["n"] += 1; _s["last_end"] = _time.perf_counter()
_tr.MiniMaxH3Model.forward = _fw
def _sample():
    while True:
        _time.sleep(0.05)
        if _s["in"]: continue
        f = _sys._current_frames().get(_s["tid"])
        if f is None: continue
        st = _tb.extract_stack(f)[-6:]
        key = " < ".join(f"{x.name}:{x.lineno}@{x.filename.split('/')[-1]}" for x in reversed(st))
        if _s["n"] % 4 == 0 and _s["n"] < 4 * int(var.get("repeat", 1)): _pre.append((_time.perf_counter(), key))
        elif _s["n"] % 4 == 0: _hits["[after last] " + key] += 1
        else: _hits["[between] " + key] += 1
_th.Thread(target=_sample, daemon=True).start()
def _rep():
    print("## OUTSIDE gaps between forwards: %.1fs total; samples (x50 ms):" % _s["gap"])
    for k, n in _hits.most_common(22): print("## OUTSIDE %4d  %s" % (n, k[:330]), flush=True)
after = [_rep]
