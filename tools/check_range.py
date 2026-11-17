"""Range support of Handler.send_file (needed to seek in a <video>): pass + block, no server import.
    .venv/Scripts/python tools/check_range.py
"""
import ast, io, re, sys, mimetypes, tempfile
from http.server import BaseHTTPRequestHandler
from pathlib import Path

src = (Path(__file__).resolve().parent.parent / "server.py").read_text(encoding="utf-8")
fn = next(n for c in ast.parse(src).body if isinstance(c, ast.ClassDef) and c.name == "Handler" for n in c.body if getattr(n, "name", "") == "send_file")
ns = {"re": re, "mimetypes": mimetypes, "Path": Path}
exec(ast.get_source_segment(src, fn).replace("    def send_file", "def send_file", 1).replace("\n    ", "\n"), ns)


class H(BaseHTTPRequestHandler):
    send_file = ns["send_file"]
    def __init__(self, rng):
        self.headers, self.hdr, self.wfile, self.code = ({"Range": rng} if rng else {}), {}, io.BytesIO(), None
    def send_response(self, c, m=None): self.code = c
    def send_header(self, k, v): self.hdr[k] = v
    def end_headers(self): pass
    def send_error(self, c, *a): self.code = c

fails = 0
def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name); fails += not ok

with tempfile.TemporaryDirectory() as t:
    base = Path(t); f = base / "a.mp4"; f.write_bytes(bytes(range(100)))
    def get(rng):
        h = H(rng); h.send_file(f, base); return h
    h = get(None); check("no Range: 200, whole file, Accept-Ranges", h.code == 200 and len(h.wfile.getvalue()) == 100 and h.hdr["Accept-Ranges"] == "bytes")
    h = get("bytes=10-19"); check("bytes=10-19: 206 + right bytes + Content-Range", h.code == 206 and h.wfile.getvalue() == bytes(range(10, 20)) and h.hdr["Content-Range"] == "bytes 10-19/100" and h.hdr["Content-Length"] == "10")
    h = get("bytes=90-"); check("open end: to EOF", h.code == 206 and h.wfile.getvalue() == bytes(range(90, 100)))
    h = get("bytes=0-999"); check("end past EOF clamps", h.code == 206 and len(h.wfile.getvalue()) == 100)
    h = get("bytes=-5"); check("suffix: last 5", h.code == 206 and h.wfile.getvalue() == bytes(range(95, 100)))
    h = get("bytes=200-300"); check("block: start past EOF → 416", h.code == 416 and h.wfile.getvalue() == b"")
    h = get("garbage"); check("block: bad header → plain 200", h.code == 200 and len(h.wfile.getvalue()) == 100)
print("all passed" if not fails else f"{fails} failed"); sys.exit(bool(fails))
