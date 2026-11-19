"""status.json of a shootout run -> one self-contained HTML page (Canvas). Video links go through the Morpheus server (:5196/output/).
    python bench/h3shootout/page.py <run dir>   -> prints the HTML"""
import json, sys, time
from pathlib import Path

LABEL = {"flash_int8": "HyperFlow Flash (int8)", "gguf_q4_k_m": "HyperFlow GGUF Q4_K_M", "gguf_q6_k": "HyperFlow GGUF Q6_K", "10eros_q5_0": "10Eros-Max TURBO Q5_0", "10eros_q6_k": "10Eros-Max TURBO Q6_K"}
COLS = [("load_s", "Load", "s"), ("step_first_s", "Step 1", "s"), ("step_avg_warm_s", "Ø step 2–8", "s"), ("sampling_s", "Sampling", "s"),
        ("decode_s", "Decode", "s"), ("total_s", "Total", "s"), ("worker_ram_peak_gb", "RAM peak (worker)", "GB"), ("ram_avg_gb", "anon RAM Ø", "GB"), ("ram_peak_gb", "anon RAM peak", "GB"),
        ("vram_peak_gb", "VRAM peak", "GB")]


def fmt(v, unit):
    if v is None:
        return "–"
    if unit == "s":
        return f"{int(v // 60)}:{int(v % 60):02d}" if v >= 60 else f"{v:.1f} s"
    return f"{v:.1f} {unit}"


def page(run: Path) -> str:
    st = json.loads((run / "status.json").read_text(encoding="utf-8"))
    base = f"http://localhost:5196/output/bench/h3shootout/{run.name}/"
    done = [m for m in st["models"].values() if m.get("state") == "done"]
    best = {k: min((m[k] for m in done if m.get(k) is not None), default=None) for k, _, _ in COLS}
    rows, cards = [], []
    for n, m in st["models"].items():
        s = m.get("state")
        if s in ("loading", "sampling", "decoding"):
            el = m.get("elapsed") or (time.time() - m.get("started", time.time()))
            pct = {"loading": 5, "sampling": 10 + 80 * (m.get("step", 0) / (m.get("total") or 8)), "decoding": 92}[s]
            info = {"loading": "loading model", "sampling": f"step {m.get('step', 0) + 1}/{m.get('total') or 8}", "decoding": "VAE decode"}[s]
            badge = f'<div class="bar"><i style="width:{pct:.0f}%"></i></div><small>{info} · {fmt(el, "s")} · RAM now {fmt(m.get("ram_now_gb"), "GB")} · peak {fmt(m.get("ram_peak_gb"), "GB")}</small>'
        elif s == "done":
            badge = f'<small>▶ player below · {m["video"]}</small>'
        elif s == "error":
            badge = f'<span class="err">error: {m.get("error", "")[:160]}</span>'
        else:
            badge = '<small class="q">queued</small>'
        cells = "".join(f'<td class="{"best" if s == "done" and m.get(k) is not None and m.get(k) == best[k] and len(done) > 1 else ""}">{fmt(m.get(k), u)}</td>' for k, _, u in COLS)
        rows.append(f"<tr><th>{LABEL.get(n, n)}</th>{cells}</tr>")
        cards.append(f'<div class="card {s}"><b>{LABEL.get(n, n)}</b><span class="st">{s}</span>{badge}</div>')
    cfg = st["settings"]
    head = "".join(f"<th>{l}</th>" for _, l, _ in COLS)
    fin = "✅ finished" if st.get("finished") else "⏳ running"
    return f"""<style>
body{{font:13px 'IBM Plex Mono',monospace;background:#1E1B3A;color:#F6EBDD;margin:0;padding:16px}}
h1{{font:600 18px Montserrat,sans-serif;margin:0 0 4px;color:#8A6CD8}} .sub{{opacity:.7;margin-bottom:14px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;margin-bottom:16px}}
.card{{background:#2a2650;border-radius:8px;padding:10px;display:flex;flex-direction:column;gap:6px;border-left:4px solid #4A78B0}}
.card.done{{border-color:#5fc48a}} .card.error{{border-color:#F08AA8}} .card.queued{{opacity:.55}}
.st{{font-size:11px;text-transform:uppercase;opacity:.7}} a{{color:#F08AA8}} .err{{color:#F08AA8}}
.bar{{height:6px;background:#1E1B3A;border-radius:3px;overflow:hidden}} .bar i{{display:block;height:100%;background:#8A6CD8}}
table{{border-collapse:collapse;width:100%}} td,th{{padding:6px 8px;border-bottom:1px solid #3a3570;text-align:right;white-space:nowrap}}
tr th:first-child{{text-align:left}} td.best{{color:#5fc48a;font-weight:600}} .wrap{{overflow-x:auto}}
</style>
<h1>H3 shootout · {fin}</h1>
<div class="sub">run {st['run']} · {cfg['seconds']} s (158 frames) · {cfg['resolution']} {cfg['aspect']} · {cfg['steps']} steps · seed {cfg['seed']} · {cfg['encoder']} · no LoRAs · cold load each</div>
<div class="cards">{''.join(cards)}</div>
<div class="wrap"><table><tr><th>Model</th>{head}</tr>{''.join(rows)}</table></div>
<p class="sub">green = best · RAM peak (worker) = render process max RSS incl. memory-mapped weights (the real footprint) · anon RAM = WSL MemTotal−MemAvailable, 1 s samples — excludes the weights in page cache · videos open via the Morpheus server (:5196)</p>"""


if __name__ == "__main__":
    print(page(Path(sys.argv[1])))
