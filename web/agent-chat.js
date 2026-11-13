// <morpheus-agent-chat> — the prompt-writer chat, for any Morpheus page.
//   attributes: task="LTX 2.3 video prompt for the Playground"  (context the agent gets with every message)
//               fake                                           (no pi, no OpenRouter, no cost)
//               model="deepseek/deepseek-v4.1-flash"          (start model; the server default otherwise)
//               use                                            (a Use button next to Copy, see "prompt-use")
//   el.addImage(dataUrl)   hand over a picture from the host page (setImage(null) clears them)
//   pictures are named img1, img2 … in upload order across the chat; the agent picks the flow from how you use them
//   event "prompt-copy"    detail: {prompt} — fired when the user copies a prompt
//   event "prompt-use"     detail: {prompt} — fired by the Use button; the host puts the prompt where it belongs
//   el.sent                how many pictures went out so far (the next one is img<sent + 1>)
// Talks to /api/agentchat/{open,send,close}; the key and the agent process stay on the server.
const ICON = {  // Lucide (ISC), inlined
  image: '<path d="M16 5h6"/><path d="M19 2v6"/><path d="M21 11.5V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7.5"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/><circle cx="9" cy="9" r="2"/>',
  send: '<path d="m5 12 7-7 7 7"/><path d="M12 19V5"/>',
  copy: '<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
  spin: '<path d="M21 12a9 9 0 1 1-6.219-8.56"/>',
  down: '<path d="m6 9 6 6 6-6"/>',
  eye: '<path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"/><circle cx="12" cy="12" r="3"/>',
  eyeoff: '<path d="M10.733 5.076a10.744 10.744 0 0 1 11.205 6.575 1 1 0 0 1 0 .696 10.747 10.747 0 0 1-1.444 2.49"/><path d="M14.084 14.158a3 3 0 0 1-4.242-4.242"/><path d="M17.479 17.499a10.75 10.75 0 0 1-15.417-5.151 1 1 0 0 1 0-.696 10.75 10.75 0 0 1 4.446-5.143"/><path d="m2 2 20 20"/>',
};
const svg = (n, size = 18) => `<svg viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICON[n]}</svg>`;
const CSS = `
:host { display:flex; flex-direction:column; min-height:0; height:100%; background:#141022; border:1px solid #2e2647;
  border-radius:16px; color:#ece8f7; font:14px/1.5 Inter, system-ui, sans-serif; overflow:hidden; }
* { scrollbar-width:thin; scrollbar-color:#3a3060 transparent; }
::-webkit-scrollbar { width:6px; height:6px; } ::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:#3a3060; border-radius:6px; } ::-webkit-scrollbar-thumb:hover { background:#8A6CD8; }
header { display:flex; align-items:center; gap:8px; padding:11px 16px; border-bottom:1px solid #2e2647; }
header b { font:600 12.5px 'IBM Plex Mono', monospace; letter-spacing:.08em; }
.stats:not([hidden]) + .model { margin-left:0; }
.stats b { font-weight:600; color:#F6EBDD; } .stats i { font-style:normal; color:#F08AA8; }
header small { color:#9a90b8; font-size:11px; margin-left:auto; font-family:'IBM Plex Mono', monospace;
  background:#1c1630; border:1px solid #2e2647; border-radius:20px; padding:2px 9px; }
.log { flex:1; overflow-y:auto; padding:16px; display:flex; flex-direction:column; gap:12px; }
.msg { max-width:85%; padding:10px 13px; border-radius:14px; white-space:pre-wrap; overflow-wrap:anywhere; }
.agent { background:#1c1630; align-self:flex-start; border-bottom-left-radius:4px; }
.user { background:linear-gradient(135deg,#3d2770,#33285c); align-self:flex-end; border-bottom-right-radius:4px; }
.user img { display:block; max-width:200px; max-height:150px; border-radius:9px; margin-bottom:7px; }
.note { background:transparent; color:#9a90b8; font-size:12px; align-self:center; padding:2px 10px; }
.err { background:#3a1424; color:#ff9db3; align-self:flex-start; }
.wait { color:#9a90b8; display:flex; align-items:center; gap:8px; }
.wait svg { animation:spin 1s linear infinite; } @keyframes spin { to { transform:rotate(360deg); } }
.prompt { position:relative; margin:8px 0 2px; background:#0b0814; border:1px solid #2e2647; border-radius:10px;
  padding:12px 44px 12px 13px; font:12.5px/1.6 'IBM Plex Mono', monospace; white-space:pre-wrap; }
.copy { position:absolute; top:7px; right:7px; width:30px; height:30px; display:grid; place-items:center; padding:0;
  background:#1c1630; border:1px solid #2e2647; border-radius:8px; color:#9a90b8; cursor:pointer; transition:.15s; }
.copy:hover { color:#ece8f7; border-color:#8A6CD8; }
.copy.done { color:#6be0a8; border-color:#6be0a8; }
.prompt.usable { padding-right:44px; padding-bottom:44px; }
.use { position:absolute; bottom:7px; right:7px; height:30px; padding:0 11px; border:0; border-radius:8px; cursor:pointer;
  color:#fff; background:linear-gradient(135deg,#8A6CD8,#F08AA8); font:600 11.5px 'IBM Plex Mono', monospace; }
.use:hover { filter:brightness(1.1); }
footer { padding:12px; }
.box { background:#1c1630; border:1px solid #2e2647; border-radius:14px; padding:8px; transition:border-color .15s, box-shadow .15s; }
.box:focus-within, .box.drag { border-color:#8A6CD8; box-shadow:0 0 0 3px rgba(138,108,216,.18); }
.chips { display:flex; flex-wrap:wrap; gap:10px; padding:4px 2px 8px; } .chips:empty { display:none; }
.chip { position:relative; width:64px; height:64px; border-radius:10px; background:center/cover no-repeat; border:1px solid #2e2647; }
.tag { position:absolute; left:4px; bottom:4px; background:rgba(11,8,20,.8); color:#F6EBDD; border-radius:6px; padding:0 5px;
  font:600 10.5px/16px 'IBM Plex Mono', monospace; }
.pics { display:flex; flex-wrap:wrap; align-items:flex-start; gap:6px; margin-bottom:7px; } .pic { position:relative; }
.user .pic img { margin:0; max-width:150px; max-height:120px; }
.chip button { position:absolute; top:-6px; right:-6px; width:20px; height:20px; padding:0; display:grid; place-items:center;
  border-radius:50%; border:0; background:#F08AA8; color:#1E1B3A; cursor:pointer; }
textarea { display:block; width:100%; box-sizing:border-box; resize:none; min-height:44px; max-height:160px; background:transparent;
  color:inherit; border:0; outline:0; padding:4px 6px; font:inherit; }
textarea::placeholder { color:#6f6690; }
.row { display:flex; align-items:center; gap:6px; margin-top:4px; }
.icon { width:34px; height:34px; display:grid; place-items:center; padding:0; background:transparent; border:0; border-radius:9px;
  color:#9a90b8; cursor:pointer; transition:.15s; }
.icon:hover { background:#2a2245; color:#ece8f7; }
.hint { color:#6f6690; font-size:11.5px; margin-left:4px; }
.go { margin-left:auto; width:36px; height:36px; display:grid; place-items:center; padding:0; border:0; border-radius:50%; cursor:pointer;
  color:#fff; background:linear-gradient(135deg,#8A6CD8,#F08AA8); transition:transform .15s, opacity .15s; }
.go:hover:not(:disabled) { transform:scale(1.06); }
.go:disabled { opacity:.4; cursor:default; }
.icon:disabled { opacity:.35; cursor:not-allowed; } .icon:disabled:hover { background:transparent; color:#9a90b8; }
.picker { position:relative; }
.stats:not([hidden]) + .picker { margin-left:0; } .picker { margin-left:auto; }
.model { display:flex; align-items:center; gap:4px; color:#9a90b8; font:11px 'IBM Plex Mono', monospace; cursor:pointer;
  background:#1c1630; border:1px solid #2e2647; border-radius:20px; padding:2px 6px 2px 9px; transition:.15s; }
.model:hover, .model.open { color:#ece8f7; border-color:#8A6CD8; }
.menu { position:absolute; right:0; top:calc(100% + 6px); z-index:5; min-width:270px; background:#1c1630; border:1px solid #2e2647;
  border-radius:12px; padding:5px; box-shadow:0 12px 32px rgba(0,0,0,.5); }
.menu[hidden] { display:none; }
.opt { display:grid; grid-template-columns:18px 1fr; gap:2px 9px; align-items:center; width:100%; text-align:left; padding:7px 9px;
  border:0; border-radius:8px; background:transparent; color:#ece8f7; cursor:pointer; font:inherit; }
.opt:hover { background:#2a2245; } .opt.on { background:#2e2552; }
.opt svg { color:#6be0a8; } .opt.blind svg { color:#6f6690; }
.opt span { font-size:13px; } .opt small { grid-column:2; color:#9a90b8; font:10.5px 'IBM Plex Mono', monospace; background:none; border:0; padding:0; margin:0; }
`;

class AgentChat extends HTMLElement {
  connectedCallback() {
    if (this.shadowRoot) return;
    const root = this.attachShadow({ mode: "open" });
    root.innerHTML = `<style>${CSS}</style>
      <header><b>✦ PROMPT WRITER</b><small class="stats" hidden></small><div class="picker"><button class="model" title="Switch model">…</button><div class="menu" hidden></div></div></header>
      <div class="log"></div>
      <footer><div class="box">
        <div class="chips"></div>
        <textarea rows="1" placeholder="Look at this, write me an LTX 2.3 prompt…"></textarea>
        <div class="row">
          <button class="icon pick" title="Add pictures — img1, img2 … (or drop / paste)">${svg("image")}</button>
          <span class="hint">Enter to send · Shift+Enter new line</span>
          <button class="go" title="Send">${svg("send")}</button>
        </div>
        <input type="file" accept="image/*" multiple hidden>
      </div></footer>`;
    this.$ = (s) => root.querySelector(s);
    this.images = []; this.sent = 0;  // pending pictures; how many went out (next is img<sent + 1>)
    const box = this.$(".box"), file = this.$("input"), text = this.$("textarea");
    this.$(".pick").onclick = () => file.click();
    file.onchange = () => { [...file.files].forEach((f) => this.readFile(f)); file.value = ""; };
    box.ondragover = (e) => { e.preventDefault(); box.classList.add("drag"); };
    box.ondragleave = () => box.classList.remove("drag");
    box.ondrop = (e) => { e.preventDefault(); box.classList.remove("drag"); [...e.dataTransfer.files].filter((f) => f.type.startsWith("image/")).forEach((f) => this.readFile(f)); };
    text.oninput = () => { text.style.height = "auto"; text.style.height = text.scrollHeight + "px"; };
    text.onpaste = (e) => { [...e.clipboardData.files].filter((f) => f.type.startsWith("image/")).forEach((f) => this.readFile(f)); };
    text.onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); this.send(); } };
    this.$(".go").onclick = () => this.send();
    this.open();
    addEventListener("pagehide", () => this.session && navigator.sendBeacon("/api/agentchat/close", JSON.stringify({ session: this.session })));
  }

  async api(action, body) {
    const r = await fetch(`/api/agentchat/${action}`, { method: "POST", body: JSON.stringify(body) });
    const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
    if (!r.ok || j.error) throw new Error(j.error || `HTTP ${r.status}`);
    return j;
  }

  async open() {
    try {
      const j = await this.api("open", { fake: this.hasAttribute("fake"), model: this.getAttribute("model") || undefined });
      this.session = j.session;
      this.models = j.models; this.fake = j.fake; this.skills = j.skills;
      this.showModel(j.model);
      this.$(".model").onclick = (e) => { e.stopPropagation(); this.toggleMenu(); };
      this.shadowRoot.addEventListener("click", (e) => { if (!e.target.closest(".picker")) this.toggleMenu(false); });
      this.add("agent", j.greeting);
    } catch (e) { this.add("err", e.message); }
  }

  // running totals for this chat: tokens in / out and pi's cost estimate; the tooltip has the last message alone
  count(u) {
    const t = (this.total ||= { in: 0, out: 0, cost: 0 });
    t.in += u.in; t.out += u.out; t.cost += u.cost;
    const k = (n) => (n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n));
    const el = this.$(".stats");
    el.hidden = false;
    el.innerHTML = `↑ <b>${k(t.in)}</b> ↓ <b>${k(t.out)}</b> · <i>$${t.cost.toFixed(4)}</i>`;
    el.title = `This chat: ${t.in} tokens in, ${t.out} out, ≈ $${t.cost.toFixed(4)} (estimate from pi's price table)\n`
      + `Last message: ${u.in} in, ${u.out} out, ≈ $${u.cost.toFixed(4)}`;
  }

  showModel(id) {
    this.model = this.models.find((m) => m.id === id) || { id, label: id, vision: true };
    const b = this.$(".model");
    b.innerHTML = `${this.fake ? "fake · " : ""}${this.model.label}${svg("down", 13)}`;
    b.title = `${this.model.id} — click to switch\nskills: ${this.skills.join(", ")}`;
    const pick = this.$(".pick");
    pick.disabled = !this.model.vision;
    pick.title = this.model.vision ? "Add pictures — img1, img2 … (or drop / paste)" : `${this.model.label} can't see pictures`;
  }

  toggleMenu(open = this.$(".menu").hidden) {
    const menu = this.$(".menu");
    menu.hidden = !open; this.$(".model").classList.toggle("open", open);
    if (!open) return;
    menu.innerHTML = "";
    this.models.forEach((m) => {
      const o = document.createElement("button");
      o.className = `opt${m.id === this.model.id ? " on" : ""}${m.vision ? "" : " blind"}`;
      o.title = m.vision ? "Sees pictures" : "Text only — can't see pictures";
      o.innerHTML = `${svg(m.vision ? "eye" : "eyeoff", 15)}<span>${m.label}</span><small>${m.price} per M tokens in / out</small>`;
      o.onclick = () => this.switchModel(m.id);
      menu.append(o);
    });
  }

  async switchModel(id) {
    this.toggleMenu(false);
    if (id === this.model.id) return;
    try {
      const j = await this.api("model", { session: this.session, model: id });
      this.showModel(j.model);
      this.add("agent note", `Switched to ${this.model.label}${this.model.vision ? "" : " — text only, it can't see pictures"}.`);
    } catch (e) { this.add("err", e.message); }
  }

  readFile(f) {
    const r = new FileReader();
    r.onload = () => this.addImage(r.result);
    r.readAsDataURL(f);
  }

  addImage(dataUrl) {
    if (this.model && !this.model.vision) return this.add("err", `${this.model.label} can't see pictures — switch model first.`);
    if (this.images.length < 6) this.images.push(dataUrl);
    this.drawChips();
  }

  setImage(dataUrl) { this.images = dataUrl ? [dataUrl] : []; this.drawChips(); }

  drawChips() {
    const box = this.$(".chips");
    box.innerHTML = "";
    this.images.forEach((url, i) => {
      const chip = document.createElement("div");
      chip.className = "chip"; chip.style.backgroundImage = `url("${url}")`;
      chip.innerHTML = `<span class="tag">img${this.sent + i + 1}</span><button title="Remove">${svg("x", 12)}</button>`;
      chip.querySelector("button").onclick = () => { this.images.splice(i, 1); this.drawChips(); };
      box.append(chip);
    });
  }

  add(kind, text, images = [], first = 1) {
    const el = document.createElement("div");
    el.className = `msg ${kind}`;
    if (images.length) {
      const pics = document.createElement("div"); pics.className = "pics";
      images.forEach((url, i) => { pics.insertAdjacentHTML("beforeend", `<div class="pic"><img src="${url}"><span class="tag">img${first + i}</span></div>`); });
      el.append(pics);
    }
    // a ```prompt block becomes a box with its own Copy button; the chatter around it stays text
    const parts = String(text).split(/```(?:prompt)?\s*\n([\s\S]*?)```/);
    parts.forEach((p, i) => {
      if (i % 2 === 0) { if (p.trim()) el.append(p.trim()); return; }
      const box = document.createElement("div");
      box.className = "prompt";
      box.textContent = p.trim();
      const b = document.createElement("button");
      b.className = "copy"; b.title = "Copy prompt"; b.innerHTML = svg("copy", 16);
      b.onclick = async () => {
        await navigator.clipboard.writeText(p.trim());
        b.classList.add("done"); b.innerHTML = svg("check", 16); b.title = "Copied";
        setTimeout(() => { b.classList.remove("done"); b.innerHTML = svg("copy", 16); b.title = "Copy prompt"; }, 1500);
        this.dispatchEvent(new CustomEvent("prompt-copy", { detail: { prompt: p.trim() }, bubbles: true }));
      };
      box.append(b);
      if (this.hasAttribute("use")) {
        const u = document.createElement("button");
        u.className = "use"; u.title = "Put this prompt into the page"; u.textContent = "Use";
        u.onclick = () => this.dispatchEvent(new CustomEvent("prompt-use", { detail: { prompt: p.trim() }, bubbles: true }));
        box.classList.add("usable"); box.append(u);
      }
      el.append(box);
    });
    this.$(".log").append(el);
    el.scrollIntoView({ block: "end", behavior: "smooth" });
    return el;
  }

  async send() {
    const text = this.$("textarea"), go = this.$(".go");
    const message = text.value.trim();
    if ((!message && !this.images.length) || go.disabled || !this.session) return;
    if (this.images.length && !this.model.vision) return this.add("err", `${this.model.label} can't see pictures — remove them or switch model.`);
    const images = this.images;
    this.add("user", message, images, this.sent + 1);
    this.sent += images.length; this.images = [];
    text.value = ""; this.drawChips(); go.disabled = true;
    const wait = this.add("agent wait", "writing…");
    wait.insertAdjacentHTML("afterbegin", svg("spin", 15));
    text.style.height = "auto";
    try {
      const j = await this.api("send", { session: this.session, message, images, task: this.getAttribute("task") || "" });
      wait.remove(); this.add("agent", j.reply || "(empty answer)");
      if (j.usage) this.count(j.usage);
    } catch (e) { wait.remove(); this.add("err", e.message); }
    go.disabled = false; text.focus();
  }
}
customElements.define("morpheus-agent-chat", AgentChat);
