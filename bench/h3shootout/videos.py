"""<run>/videos.html: a static page with one inline player per finished clip (no reload, so playback is never interrupted).
    python bench/h3shootout/videos.py <run dir>"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from page import LABEL

run = Path(sys.argv[1])
st = json.loads((run / "status.json").read_text(encoding="utf-8"))
figs = "".join(f'<figure><video controls preload="metadata" src="{m["video"]}"></video><figcaption><button onclick="big(&quot;{m["video"]}&quot;)">⤢ big</button>{LABEL.get(n, n)} · {m.get("total_s", 0) // 60:.0f}:{m.get("total_s", 0) % 60:02.0f}</figcaption></figure>'
               for n, m in st["models"].items() if m.get("state") == "done" and m.get("video"))
(run / "videos.html").write_text(f"""<!doctype html><meta charset="utf-8"><style>
body{{margin:0;padding:0 16px 16px;background:#1E1B3A;color:#F6EBDD;font:13px 'IBM Plex Mono',monospace}}
.g{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}} figure{{margin:0;background:#2a2650;border-radius:8px;padding:8px}}
video{{width:100%;border-radius:4px;background:#000}} button{{background:#8A6CD8;color:#F6EBDD;border:0;border-radius:4px;padding:2px 8px;cursor:pointer;float:right}}
#big{{display:none;position:fixed;inset:0;background:#000;z-index:9;flex-direction:column}} #big.on{{display:flex}} #big video{{flex:1;min-height:0;height:100%;object-fit:contain;border-radius:0}}
#big button{{position:absolute;top:8px;right:8px;z-index:10;font-size:16px}} figcaption{{margin-top:4px}} h3{{color:#8A6CD8;font-family:Montserrat,sans-serif}}
</style><h3>Clips</h3><div class="g">{figs or "none finished yet"}</div>
<div id="big"><button onclick="small()">✕ close</button><video controls></video></div>
<script>const B=document.getElementById('big'),V=B.querySelector('video');
function big(src){{document.querySelectorAll('.g video').forEach(v=>v.pause());V.src=src;B.classList.add('on');V.play()}}
function small(){{V.pause();B.classList.remove('on')}} addEventListener('keydown',e=>e.key==='Escape'&&small());</script>""", encoding="utf-8")
