// Create: the new panel flow. A COPY of the panel editor in yumeko.html (openPanel .. saveMasks, 2026-11-11);
// changes here never reach the old flow. It shares the page's helpers ($, act, get, esc, file, folds ...) but keeps
// its own state (panel, tool, drawn, timer), so both editors can't step on each other.
window.createFlow = (() => {
// Create's own layout fixes (scoped to .cflow, so the old editor stays as it is)
document.head.insertAdjacentHTML('beforeend', `<style>
  .cflow.nov { grid-template-columns:minmax(0,1.9fr) minmax(400px,1fr); }
  @media (max-width:1200px) { .cflow.nov { grid-template-columns:minmax(0,1fr) minmax(380px,1fr); } }
  .cflow .pmid > :first-child, .cflow .pside > :first-child { margin-top:0; }
  .cflow .skgrid { display:grid; grid-template-columns:repeat(var(--skcols, 3), minmax(0, 1fr)); gap:12px; width:100%; max-width:calc(var(--skcols, 3) * 340px); margin:0 auto; }
  .cflow .skcell { position:relative; width:100%; padding:0; border:2px solid var(--line, #333); border-radius:12px; overflow:hidden; background:var(--raised, #111); cursor:pointer; display:block; transition:transform .18s ease, box-shadow .18s ease, border-color .18s; z-index:1; }
  .cflow .skcell img { width:100% !important; height:auto !important; max-height:none !important; aspect-ratio:auto !important; object-fit:fill; display:block; }
  .cflow .skcell:hover, .cflow .skcell:focus-visible { transform:scale(1.7); z-index:5; border-color:var(--accent); box-shadow:0 18px 50px rgba(0,0,0,.6); }
  .cflow .skcount { width:auto; }
  .cflow .skwait { display:grid; place-items:center; color:var(--muted); cursor:default; border:2px dashed var(--line, #444); aspect-ratio:var(--skar, 1.5); }
  .cflow .skbox { overflow:visible; }
  .cflow .snips { display:flex; flex-wrap:wrap; gap:6px; margin-left:auto; justify-content:flex-end; }
  .cflow .snip { padding:3px 11px; border-radius:999px; border:1px solid var(--accent); background:rgba(160,90,255,.14); color:inherit; font-size:12px; cursor:grab; user-select:none; }
  .cflow .snip:active { cursor:grabbing; } .cflow .snip[aria-pressed="true"] { background:var(--accent); color:#fff; }
  .cflow #snipedit { margin:0 0 8px; } .cflow #snipedit input { width:100%; }
  dialog.skbig { border:1px solid var(--accent); border-radius:14px; background:var(--panel); color:inherit; padding:12px; max-width:96vw; max-height:96vh; }
  dialog.skbig::backdrop { background:rgba(0,0,0,.8); }
  dialog.skbig img { display:block; max-width:calc(96vw - 26px); max-height:calc(96vh - 90px); margin:0 auto; border-radius:8px; }
  dialog.skbig .skbar { display:flex; align-items:center; gap:10px; margin-top:10px; } dialog.skbig .skbar span { flex:1; color:var(--muted); font-size:13px; }
  .cflow .skbox .boxhead { justify-content:space-between; }
  .cflow .cbusy { margin:0 0 10px; padding:8px 12px; border:1px solid var(--accent); border-radius:10px; background:var(--panel); }
</style>`);
let panel, drawn = {}, timer, mangaData;
const tool = { qwen:{ text:'', loras:['qwen-lightning'], refv:0 }, inp:{ text:'', negative:null, denoise:0.6, loras:[], control:'off', cstr:1, detail:'off', dstr:0.35, seed:'', steps:30, cfg:5, refv:0 }, active:null, size:36, erase:false, fix:{ who:null, open:new Set(), by:{} } };
async function openPanel(id) { clearTimeout(timer); const m = mangaData = await get('/api/yumeko/manga'); panel = await get('/api/yumeko/panel/' + id); panel._loras = m.loras; drawn = {}; tool.active = panel.characters[0]?.id ?? null; paintPanel(); pollPanel(); }
function pollPanel() { clearTimeout(timer); timer = setTimeout(async () => { if (!location.hash.includes(panel?.id)) return; const b = panel.busy, n = panel.versions.length, l = panel._loras;
  // only the render state comes from the server: replacing the whole object made edits land on a stale copy
  try { const fresh = await get('/api/yumeko/panel/' + panel.id); ['busy', 'error', 'versions', 'current', 'composition', 'progress', 'sketches', 'sketch_picked', 'sketch_count'].forEach(k => panel[k] = fresh[k]); panel._loras = l; if (b !== panel.busy || n !== panel.versions.length) { const keep = captureMasks(); paintPanel(); restoreMasks(keep); } else showBusy(); } catch (e) { console.error(e); } pollPanel(); }, panel.busy ? 1000 : 2000); }
/* the running render over the big picture: what runs, a wide bar, steps */
function showBusy() { const st = $('stage'); if (!st) return; let o = st.querySelector('.stagebusy'); if (!panel.busy) { o?.remove(); return; }
  if (!o) { o = document.createElement('div'); o.className = 'stagebusy'; o.innerHTML = '<b></b><div class="bar"><i></i></div><small></small>'; st.append(o); }
  const pr = panel.progress, pct = pr?.total ? Math.round(100 * pr.step / pr.total) : null;
  o.querySelector('b').textContent = panel.busy + ' …'; o.querySelector('.bar').classList.toggle('wait', pct === null);
  o.querySelector('i').style.width = (pct ?? 100) + '%'; o.querySelector('small').textContent = pct === null ? 'loading the model …' : `step ${pr.step} of ${pr.total} · ${pct}%`; }
const dir = () => '_manga/' + panel.id;
async function savePanel() { const { _loras, ...p } = panel; const r = await act('panel_edit', { id:p.id, title:p.title, tags:p.tags, prompt:p.prompt, negative:p.negative, characters:p.characters, width:p.width, height:p.height, locked:!!p.locked, pass_tags:p.pass_tags ?? '', pass_negative:p.pass_negative ?? '' }); return r; }
// Take what is typed but not yet saved (a textarea saves on blur, which can lose the race against a button click).
// The pill tags of a character's LoRA (from her finished set); older cards find it by file or name.
function fx(c) { return tool.fix.by[c.id] ||= { preset:'small', text:'', denoise:0.5, loras:['qwen-lightning'] }; }
const QWEN_LORAS = [['qwen-lightning', 'Lightning'], ['qwen-anything2real', 'Anything2Real'], ['qwen-multiple-angles', 'Multiple angles'], ['qwen-sfw', 'SFW']];
/* her Qwen reference: her own upload for this panel, else her base image */
/* reference picker: her training pictures, her generated ones, or a file from the PC; resolves a data URL or null */
function pickRef(ids) {
  return new Promise(done => {
    const L = panel?._loras || [], want = (ids || []).map(c => L.find(x => x.id === c.lora || x.file === c.file || x.name === c.name)?.id).filter(Boolean);
    const loras = want.length ? L.filter(l => want.includes(l.id)) : L;
    let who = loras[0]?.id, kind = 'trained', data = {}, fin = false;
    const d = document.createElement('dialog'); d.className = 'rp'; d.setAttribute('aria-label', 'Choose a reference picture');
    const end = v => { if (fin) return; fin = true; done(v); d.close(); };
    const toData = async src => { const b = await (await fetch(src)).blob(); return new Promise(r => { const f = new FileReader(); f.onload = () => r(f.result); f.readAsDataURL(b); }); };
    const paint = async () => {
      d.innerHTML = `<div class="row" style="margin:0;align-items:center">${loras.length > 1 ? `<select id="rpwho" aria-label="Character">${loras.map(l => `<option value="${l.id}" ${l.id === who ? 'selected' : ''}>${esc(l.name)}</option>`).join('')}</select>` : ''}
        <span class="chips row" style="margin:0">${[['trained', 'Trained'], ['generated', 'Generated']].map(([k, t]) => `<button class="chip" data-k="${k}" aria-pressed="${kind === k}">${t}</button>`).join('')}</span>
        <button id="rppc" style="margin-left:auto">Ref from PC</button><button id="rpx" aria-label="Close">✕</button><input type="file" accept="image/*" id="rpfile" hidden></div><div class="rpg"><p class="note">Loading …</p></div>`;
      d.querySelector('#rpx').onclick = () => end(null);
      d.querySelector('#rppc').onclick = () => d.querySelector('#rpfile').click();
      d.querySelector('#rpfile').onchange = e => { const f = e.target.files[0]; if (!f) return; const r = new FileReader(); r.onload = () => crop(r.result); r.readAsDataURL(f); };
      d.querySelectorAll('[data-k]').forEach(b => b.onclick = () => { kind = b.dataset.k; paint(); });
      const sel = d.querySelector('#rpwho'); if (sel) sel.onchange = () => { who = sel.value; paint(); };
      const g = d.querySelector('.rpg'); if (!who) { g.innerHTML = '<p class="note">No trained character here: use Ref from PC.</p>'; return; }
      try { data[who] ||= await get('/api/yumeko/job/' + who); } catch { g.innerHTML = '<p class="note">Couldn\'t load her pictures.</p>'; return; }
      const j = data[who], list = kind === 'trained' ? (j.dataset?.accepted || []) : (j.gallery || []);
      if (!list.length) { g.innerHTML = `<p class="note">No ${kind} pictures yet.</p>`; return; }
      g.innerHTML = ''; list.forEach(x => { const src = file(j.id, x.file), b = document.createElement('button'); b.title = (x.tags || []).join?.(', ') || x.prompt || '';
        b.innerHTML = `<img loading="lazy" src="${src}" alt="">`; b.onclick = async () => crop(await toData(src)); g.append(b); });
    };
    /* step 2: the picked picture large, with a square to cut out of it */
    const crop = src => {
      d.innerHTML = `<div class="row" style="margin:0;align-items:center"><button id="rpback">← Back</button><span class="note" style="margin:0">Drag the square, pull its corner to resize.</span>
        <button id="rpall" style="margin-left:auto">Whole picture</button><button class="primary" id="rpok">Use square</button><button id="rpx" aria-label="Close">✕</button></div>
        <div class="rpc"><div class="wrap"><img alt="Picked picture" src="${src}"><div class="sq"><i></i></div></div></div>`;
      const img = d.querySelector('img'), sq = d.querySelector('.sq'), wrap = d.querySelector('.wrap'); let b = null;
      const put = () => { Object.assign(sq.style, { left:b.x + 'px', top:b.y + 'px', width:b.s + 'px', height:b.s + 'px' }); };
      img.onload = () => { const W = img.clientWidth, H = img.clientHeight, s0 = Math.min(W, H) * .7; b = { x:(W - s0) / 2, y:(H - s0) / 2, s:s0 }; put(); };
      wrap.onpointerdown = e => { if (!b) return; e.preventDefault(); const W = img.clientWidth, H = img.clientHeight, sx = e.clientX, sy = e.clientY, o = { ...b }, size = e.target.tagName === 'I';
        if (!size && e.target !== sq) { const r = wrap.getBoundingClientRect(); o.x = Math.min(Math.max(e.clientX - r.left - o.s / 2, 0), W - o.s); o.y = Math.min(Math.max(e.clientY - r.top - o.s / 2, 0), H - o.s); b = { ...o }; put(); }
        const mv = ev => { const dx = ev.clientX - sx, dy = ev.clientY - sy;
          if (size) { const n = Math.max(32, Math.min(o.s + Math.max(dx, dy), W - o.x, H - o.y)); b = { ...o, s:n }; }
          else b = { ...o, x:Math.min(Math.max(o.x + dx, 0), W - o.s), y:Math.min(Math.max(o.y + dy, 0), H - o.s) };
          put(); };
        const up = () => { removeEventListener('pointermove', mv); removeEventListener('pointerup', up); };
        addEventListener('pointermove', mv); addEventListener('pointerup', up); };
      d.querySelector('#rpx').onclick = () => end(null); d.querySelector('#rpback').onclick = () => paint(); d.querySelector('#rpall').onclick = () => end(src);
      d.querySelector('#rpok').onclick = () => { const k = img.naturalWidth / img.clientWidth, c = document.createElement('canvas'), n = Math.round(b.s * k);
        c.width = c.height = n; c.getContext('2d').drawImage(img, b.x * k, b.y * k, n, n, 0, 0, n, n); end(c.toDataURL('image/png')); };
    };
    d.onclick = e => { if (e.target === d) end(null); }; d.onclose = () => { d.remove(); if (!fin) { fin = true; done(null); } };
    document.body.append(d); d.showModal(); paint();
  });
}
function refOf(c) { if (c.ref) return file(dir(), c.ref) + '?' + (c._refv || ''); const l = panel?._loras?.find(x => x.id === c.lora || x.file === c.file); return l?.base ? `/yumeko-file/${l.base.split('/').map(encodeURIComponent).join('/')}` : null; }
function pillsOf(c) { const l = panel?._loras?.find(x => x.id === c.lora || x.file === c.file || x.name === c.name); return l?.pills; }
// Folded boxes (scene, locked composition, each character), remembered per browser.
let folds; try { folds = new Set(JSON.parse(localStorage.getItem('yumekoFolds') || '[]')); } catch { folds = new Set(); }
const folded = k => folds.has(k) ? ' min' : '';
const foldBtn = k => `<button class="fold" data-fold="${k}" aria-expanded="${!folds.has(k)}" aria-label="${folds.has(k) ? 'Unfold' : 'Fold'}">${folds.has(k) ? '▸' : '▾'}</button>`;
function setFold(k, on) { on ? folds.add(k) : folds.delete(k); try { localStorage.setItem('yumekoFolds', JSON.stringify([...folds])); } catch {}
  const box = document.querySelector(`[data-box="${CSS.escape(k)}"]`), b = box?.querySelector('[data-fold]'); box?.classList.toggle('min', on);
  if (b) { b.textContent = on ? '▸' : '▾'; b.setAttribute('aria-expanded', !on); b.setAttribute('aria-label', on ? 'Unfold' : 'Fold'); } }
/* one character open at a time: opening (or clicking the head of) a card closes the others and makes her the brush, painting */
function openChar(id) { panel.characters.forEach(c => setFold('char:' + c.id, c.id !== id)); tool.erase = false;
  if (tool.active !== id) { tool.active = id; syncPanel(); const keep = captureMasks(); paintPanel(); restoreMasks(keep); }
  else if ($('erase')) { $('erase').textContent = 'Erase'; $('erase').setAttribute('aria-pressed', false); } }
document.addEventListener('click', e => { const f = e.target.closest('[data-fold]');
  if (f && f.dataset.fold.startsWith('char:') && folds.has(f.dataset.fold)) { openChar(f.dataset.fold.slice(5)); return; }
  if (f) { setFold(f.dataset.fold, !folds.has(f.dataset.fold)); return; }
  const head = e.target.closest('#chars .boxhead'); if (head && !e.target.closest('button, input, select, textarea, a')) { openChar(head.closest('[data-box]').dataset.box.slice(5)); return; }
  const a = e.target.closest('[data-foldall]'); if (a) document.querySelectorAll('.pedit [data-box]').forEach(x => setFold(x.dataset.box, a.dataset.foldall === '1')); });
function syncPanel() { if ($('prompt')) { panel.prompt = $('prompt').value; panel.negative = $('neg').value; }
  if ($('ptags')) { panel.pass_tags = $('ptags').value; panel.pass_negative = $('pneg').value; }
  document.querySelectorAll('#chars .char').forEach((w, i) => { const c = panel.characters[i], t = w.querySelector('[data-action]'), n = w.querySelector('[data-cneg]'); if (c && t) c.action = t.value; if (c && n) c.negative = n.value; }); }
function captureMasks() { return drawn; } function restoreMasks() { showMask(); }
const choosing = p => !!(p.sketches?.length || (p.busy || '').startsWith('Sketch')) && !p.sketch_picked;
function sketchGrid(p) {
  if (!choosing(p)) return p.sketch_picked && p.sketches?.length ? `<div class="row" style="margin:0 0 10px"><button id="sk-back">← Back to the sketches</button></div>` : '';
  const n = Math.max(p.sketch_count || 0, p.sketches?.length || 0) || sketchCount(), cols = skCols(n), rows = Math.ceil(n / cols);
  const cells = Array.from({ length:n }, (_, i) => { const s = p.sketches?.[i], c = i % cols, r = Math.floor(i / cols);
    const origin = `${r === 0 ? 'top' : r === rows - 1 ? 'bottom' : 'center'} ${cols === 1 ? 'center' : c === 0 ? 'left' : c === cols - 1 ? 'right' : 'center'}`;
    return s ? `<div class="skcell" role="button" tabindex="0" style="transform-origin:${origin}" data-pick="${esc(s.file)}" data-i="${i}" title="Open large · seed ${s.seed}"><img src="${file(dir(), s.file)}" alt="Sketch ${i + 1}"></div>`
      : `<div class="skcell skwait">${p.busy ? 'rendering …' : ''}</div>`; }).join('');
  return `<section class="box skbox"><div class="lockhead boxhead"><h3>Pick a sketch</h3><button id="sk-dice" ${p.busy ? 'disabled' : ''} title="New sketches, same prompt, new seeds">🎲 ${sketchCount()} new</button></div><p class="note">Same prompt, ${n} seeds. Click one to see it large, then Use this one.</p><div class="skgrid" style="--skcols:${cols}">${cells}</div></section>`;
}
// how many sketches a roll draws (the picker beside Render sketches), remembered per browser
const withLoras = () => { try { return localStorage.getItem('createLoras') === '1'; } catch { return false; } };
const sketchCount = () => { try { return Math.max(1, Math.min(16, +localStorage.getItem('createSketches') || 6)); } catch { return 6; } };
const skCols = n => n === 4 ? 2 : n <= 4 ? n : Math.ceil(n / Math.ceil(n / 4));  // 4 → 2 × 2, max 4 per row, rows as even as they go: 6 → 3 × 2, 9 → 3 × 3
// a sketch large: ←/→ switch, Enter or "Use this one" locks it in as the composition, Esc / ✕ / backdrop close
function zoomSketch(p, i) {
  const list = (p.sketches || []).filter(Boolean); if (!list.length) return;
  i = Math.max(0, list.findIndex(s => s === p.sketches[i]));
  const d = document.createElement('dialog'); d.className = 'skbig';
  d.innerHTML = `<img alt="Sketch"><div class="skbar"><button id="skprev" aria-label="Previous">←</button><button id="sknext" aria-label="Next">→</button><span></span><button id="skx">Close</button><button class="primary" id="skuse">Use this one</button></div>`;
  const img = d.querySelector('img'), info = d.querySelector('span');
  const show = () => { const s = list[i]; img.src = file(dir(), s.file); info.textContent = `Sketch ${i + 1} of ${list.length} · seed ${s.seed}`; };
  const step = k => { i = (i + k + list.length) % list.length; show(); };
  const use = async () => { const s = list[i]; d.close(); const r = await act('create_pick', { id:p.id, file:s.file }); if (r) { Object.assign(panel, r); paintPanel(); } };
  d.querySelector('#skprev').onclick = () => step(-1); d.querySelector('#sknext').onclick = () => step(1);
  d.querySelector('#skx').onclick = () => d.close(); d.querySelector('#skuse').onclick = use;
  d.onkeydown = e => { if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1); } else if (e.key === 'ArrowRight') { e.preventDefault(); step(1); } else if (e.key === 'Enter') { e.preventDefault(); use(); } };
  d.onclick = e => { if (e.target === d) d.close(); }; d.onclose = () => d.remove();
  document.body.append(d); d.showModal(); show(); d.querySelector('#skuse').focus();
}
// character snippets: one bubble per character in the Scene head; drag it into Prompt (or Negative) to drop her short text there, click it to edit.
// Plain text, no LoRA. Kept per browser; the first draft is "girl with " + her look.
const snipStore = () => { try { return JSON.parse(localStorage.getItem('createSnips') || '{}'); } catch { return {}; } };
// one short phrase, never a comma list: commas would split her into loose tags again
const SNIP_DRAFT = { Kiri:'girl with short blue bob', Nova2:'girl with very long pale aqua hair', Yumeko:'girl with white braid with red streaks', Morpheus:'girl with black pixie cut' };  // hair only: colour and cut
const SNIP_SKIP = /^(adult|mature female|girl|1girl|solo|slender.*|fair.*skin|pale skin)$/i;
const oneLine = t => t.replace(/\s*,\s*/g, ' ').replace(/\s{2,}/g, ' ').trim();
const snipOf = l => oneLine(snipStore()[l.id] ?? SNIP_DRAFT[l.name] ?? (() => { const t = (l.look || '').split(',').map(x => x.trim()).filter(x => /hair|braid|bob|ponytail|twintails|bun|pixie|bangs/i.test(x) && !SNIP_SKIP.test(x)).slice(0, 2); return t.length ? 'girl with ' + t.join(' and ') : 'girl'; })());
const snipSave = (id, v) => { try { const m = snipStore(); m[id] = v; localStorage.setItem('createSnips', JSON.stringify(m)); } catch {} };
const snipBubbles = p => { const ls = (p._loras || []).filter(l => l.name); if (!ls.length) return '';
  return `<span class="snips">${ls.map(l => `<button class="snip" draggable="true" data-snip="${esc(l.id)}" aria-pressed="false" title="Drag into Prompt · click to edit: ${esc(snipOf(l))}">${esc(l.name)}</button>`).join('')}</span>`; };
function wireSnips(p) {
  const ed = $('snipedit'); let open = null;
  app.querySelectorAll('[data-snip]').forEach(b => { const l = p._loras.find(x => x.id === b.dataset.snip);
    b.ondragstart = e => { e.dataTransfer.setData('text/x-tag', snipOf(l)); e.dataTransfer.effectAllowed = 'copy'; };  // lands like a Danbooru hit: between the nearest tags, with the marker
    b.ondragend = () => { showMark(null); document.querySelectorAll('.droptarget').forEach(x => x.classList.remove('droptarget')); };
    b.onclick = () => { app.querySelectorAll('[data-snip]').forEach(x => x.setAttribute('aria-pressed', 'false'));
      if (open === l.id) { open = null; ed.hidden = true; return; } open = l.id; b.setAttribute('aria-pressed', 'true');
      ed.hidden = false; ed.innerHTML = `<label class="note" for="snipin">${esc(l.name)}: what dragging her drops in</label><input id="snipin" value="${esc(snipOf(l))}" placeholder="girl with blond sidecut and small breasts">`;
      const inp = $('snipin'); inp.focus(); inp.oninput = () => { if (inp.value.includes(',')) { const c = inp.selectionStart; inp.value = inp.value.replace(/,/g, ''); inp.setSelectionRange(c - 1, c - 1); } snipSave(l.id, inp.value.trim()); b.title = 'Drag into Prompt · click to edit: ' + inp.value.trim(); };
      inp.onkeydown = e => { if (e.key === 'Enter' || e.key === 'Escape') { e.preventDefault(); b.click(); } }; }; });
}
function paintPanel() {
  const p = panel, top = choosing(p) ? null : p.versions.find(v => v.file === shownOf(p));  // while picking a sketch, the grid is the stage
  const book = mangaData?.books?.find(b => b.panels.includes(p.id));
  app.innerHTML = `<div class="phead"><div class="pleft">${book ? `<a class="btnlike" href="#book/${book.id}/${Math.floor(book.panels.indexOf(p.id) / 4) + 1}">← ${esc(book.title)}</a>` : ''}<input id="title" value="${esc(p.title)}" aria-label="Panel title"><button class="danger" id="delp">Delete panel</button></div>
      <div class="tagsearch hdr"><input id="ts" type="search" autocomplete="off" aria-label="Search Danbooru tags" placeholder="Danbooru tags: search, then click or drag a hit onto a field" value="${esc(tagQ)}"><div class="tsdrop"><div class="kinds" id="tk"></div><ul id="tr"></ul></div></div></div>
    <div class="pedit cflow${top ? '' : ' nov'}">${top ? `<aside class="pvers"><h3>Versions</h3><div class="versions" id="vs"></div></aside>` : ''}
    <div class="pmid">${p.busy ? `<p class="busy cbusy">${esc(p.busy)} …</p>` : ''}${p.error ? `<p class="err">${esc(p.error)}</p>` : ''}${sketchGrid(p)}${top && !choosing(p) ? `<div class="stage${tool.hide ? ' nomask' : ''}" id="stage"><button class="ico maskeye" id="eye" aria-pressed="${!!tool.hide}" title="${tool.hide ? 'Show the masks' : 'Hide the masks'}" aria-label="Hide masks"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg></button><img id="pic" src="${file(dir(), top.file)}" alt="Panel"><canvas id="cv"></canvas></div>
      <div class="row brushbar"><label class="note">Brush</label><input id="size" type="range" min="6" max="140" value="${tool.size}" style="width:140px;padding:0"><button id="erase" aria-pressed="${tool.erase}">${tool.erase ? 'Erasing' : 'Erase'}</button><button id="clear">Clear mask</button><button id="savem">Save masks</button></div>
      <section class="box magic"><h3>Fix it</h3><p class="note">Render a <b>sketch</b> (the staging, no LoRAs), then <b>Cast characters</b>: everyone keeps the sketch's place and pose and gets her own LoRA, left to right in the order listed. Then say what's wrong, naming her, and <b>Finish</b> it: larger, crisp, faces up close.</p>
      <div class="row"><button class="go" id="castb" ${p.busy || !p.characters.length ? 'disabled' : ''} title="Redraw the shown picture with every character's LoRA in her own place">Cast characters</button><button class="go" id="finb" ${p.busy ? 'disabled' : ''} title="1.5x larger, redrawn crisp with every LoRA, faces up close">Finish</button></div>
      <div class="row"><input id="magic" placeholder="Yumeko's underboob cutout is missing" aria-label="What is wrong" style="flex:1"><button class="go" id="magicb" ${p.busy ? 'disabled' : ''}>✨ Fix it</button></div></section>` : ''}
    ${p.locked ? `<section class="box${folded('lock')}" data-box="lock"><div class="lockhead boxhead">${foldBtn('lock')}<h3>Composition locked</h3><button id="unlock" ${p.busy ? 'disabled' : ''} title="Back to the scene to render a new composition">Unlock</button></div>
      <label class="note" for="ptags">Pass tags, with every character</label><textarea id="ptags" rows="2" placeholder="indoors, night, bedroom">${esc(p.pass_tags || '')}</textarea>
      <label class="note" for="pneg">Pass negative, plus each character's own</label><textarea id="pneg" rows="2">${esc(p.pass_negative || '')}</textarea>
      ${mangaData?.pass_fixed ? `<details class="fixed"><summary>Always added to every pass</summary><p><b>Positive</b> ${esc(mangaData.pass_fixed.positive)}, her description, the pass tags</p><p><b>Negative</b> the pass negative, hers, ${esc(mangaData.pass_fixed.negative)}</p></details>` : ''}</section>
` : `<section class="box${folded('scene')}" data-box="scene"><div class="lockhead boxhead">${foldBtn('scene')}<h3>Scene</h3>${snipBubbles(p)}</div><div id="snipedit" hidden></div><label class="note" for="prompt">Prompt</label><textarea id="prompt" rows="6" placeholder="2girls, sitting, eating breakfast, table, indoors, morning, upper body, from side">${esc(p.prompt)}</textarea>
      <label class="note" for="neg">Negative</label><textarea id="neg" rows="2" placeholder="text, letters, speech_bubble, watermark, signature">${esc(p.negative)}</textarea>
      <div class="row"><select id="size-p" aria-label="Size" style="width:auto"><option value="1024,1024">Square</option><option value="1216,832">Wide</option><option value="832,1216">Tall</option></select>
      <select id="skn" class="skcount" aria-label="How many sketches" title="How many sketches one roll draws">${Array.from({ length:16 }, (_, i) => `<option value="${i + 1}" ${i + 1 === sketchCount() ? 'selected' : ''}>${i + 1} ×</option>`).join('')}</select><button class="go" id="comp" ${p.busy ? 'disabled' : ''}>Render sketches</button><button id="tog" aria-pressed="${withLoras() && p.characters.length ? 'true' : 'false'}" ${p.busy || !p.characters.length ? 'disabled' : ''} title="${p.characters.length ? 'Sketches with the LoRA (0.7) and trigger of every listed character. Experimental: LoRAs together often mix identities' : 'Add a character below first'}">With LoRAs</button>${top ? '<button id="lock" title="Keep this composition; character passes then use the pass tags and negative below">Lock composition</button>' : ''}<button id="showprompt" title="The raw text the composition sends">Show prompt</button></div><div id="promptview" hidden></div>
</section>`}
    <section class="box${folded('motion')}" data-box="motion"><div class="lockhead boxhead">${foldBtn('motion')}<h3>Comes alive</h3></div>
      <label class="note" for="pmotion">Motion for <b>Animate that</b>: what moves in this panel's 2 s clip</label><textarea id="pmotion" rows="3" placeholder="Anime style. Kiri's goggles slip down her nose, Nova laughs, a slow camera push-in.">${esc(p.motion || '')}</textarea></section></div>
    <div class="pside">      <section class="box qbox${folded('qwen')}" data-box="qwen"><div class="lockhead boxhead">${foldBtn('qwen')}<h3>Qwen edit</h3></div>
        <p class="note">One sentence for the whole panel. Paint its own mask to edit only that area; without a mask Qwen gets the whole picture.</p>
        <div class="qrow"><textarea id="qtext" rows="3" placeholder="The three women laugh, a pillow flies through the air.">${esc(tool.qwen.text)}</textarea>
          <button class="qref" id="qref" title="${p.qwen_ref ? 'Your reference: click to change' : 'Optional: a reference picture, e.g. an outfit or an object'}" aria-label="Reference picture">${p.qwen_ref ? `<img src="${file(dir(), p.qwen_ref)}?${tool.qwen.refv || ''}" alt=""><i id="qrefx" title="Remove the reference" role="button">✕</i>` : '<span>+ ref</span>'}</button><input type="file" accept="image/*" id="qfile" hidden></div>
        <span class="chips row" style="margin:0">${QWEN_LORAS.map(([k, t]) => `<button class="chip" data-pqlora="${k}" aria-pressed="${tool.qwen.loras.includes(k)}">${t}</button>`).join('')}</span>
        <div class="row" style="margin:0"><button id="qbrush" aria-pressed="${tool.active === 'qwen'}">${tool.active === 'qwen' ? 'Painting the Qwen mask' : 'Paint mask'}</button><button id="qclear">Clear mask</button><button id="qrun" class="primary" ${p.busy || !top ? 'disabled' : ''}>Qwen edit</button><button id="qrunj" ${p.busy || !top ? 'disabled' : ''} title="The Qwen edit, then JANKU goes over the same area lightly (Change 0.3) so it is back in the panel's style. JANKU uses the scene tags and the LoRAs and negative chosen in the Inpaint box below. About half a minute longer; both results land in Versions.">Qwen edit + JANKU</button></div>
      </section>
      <section class="box qbox${folded('inpaint')}" data-box="inpaint"><div class="lockhead boxhead">${foldBtn('inpaint')}<h3>Inpaint</h3></div>
        <p class="note">JANKU repaints from your tags. Paint its own mask to repaint only that area; without a mask the whole picture is redone.</p>
        <textarea id="itext" rows="3" placeholder="red kimono, floral print, obi">${esc(tool.inp.text)}</textarea>
        <label class="note" for="ineg">Negative</label><textarea id="ineg" rows="2" placeholder="nsfw, extra person">${esc(tool.inp.negative ?? (p.locked ? p.pass_negative : p.negative) ?? '')}</textarea>
        <label class="note" for="iden">Change ${tool.inp.denoise}</label><input id="iden" type="range" min="0.05" max="1" step="0.05" value="${tool.inp.denoise}" style="padding:0" title="0.3 keeps the picture and refines it, 0.6 changes it, 0.9 repaints it">
        ${p._loras.length ? `<label class="note">LoRAs, up to three</label><span class="chips row" style="margin:0">${p._loras.map(l => `<button class="chip" data-ilora="${esc(l.id)}" aria-pressed="${tool.inp.loras.some(x => x.id === l.id)}" title="Her trigger is added in front of your tags">${esc(l.name)}</button>`).join('')}</span>
        ${tool.inp.loras.map(x => `<label class="note">${esc(p._loras.find(l => l.id === x.id)?.name || 'LoRA')} ${x.strength}</label><input data-istr="${esc(x.id)}" type="range" min="0.3" max="1.3" step="0.05" value="${x.strength}" style="padding:0">${p._loras.find(l => l.id === x.id)?.pills?.outfit?.[0] ? `<span class="chips row" style="margin:0"><button class="chip" data-iout="${esc(x.id)}" aria-pressed="${!!x.outfit}" title="Adds her signature outfit: ${esc(p._loras.find(l => l.id === x.id).pills.outfit[0])}">Her outfit</button></span>` : ''}`).join('')}
        ${tool.inp.loras.length ? '<p class="note">Her look (hair, eyes, marks) is added with her trigger. Turn Her outfit off to dress her in your own tags.</p>' : ''}` : ''}
        <label class="note">Pose (ControlNet)</label><span class="chips row" style="margin:0">${[['off', 'Off', 'No pose guidance'], ['keep', 'Keep pose', 'The skeleton is read from the shown version: a strong repaint keeps the pose'], ['ref', 'Change pose', 'The pose of another picture, put where your mask is; needs a mask over the character']].map(([k, t, d]) => `<button class="chip" data-ictl="${k}" aria-pressed="${tool.inp.control === k}" title="${d}">${t}</button>`).join('')}</span>
        ${tool.inp.control === 'ref' ? `<div class="row" style="margin:0;align-items:center"><button class="qref" id="iref" title="${p.pose_ref ? 'Your pose picture: click to change' : 'A picture of a person in the pose you want'}" aria-label="Pose picture">${p.pose_ref ? `<img src="${file(dir(), p.pose_ref)}?${tool.inp.refv}" alt=""><i id="irefx" title="Remove the pose picture" role="button">✕</i>` : '<span>+ pose</span>'}</button>${p.pose_ref ? `<img class="iskel" src="${file(dir(), 'pose_ref_skel.png')}?${tool.inp.refv}" alt="The pose that was read">` : ''}<span class="note" style="flex:1">A new pose needs Change 0.9 to 1 and tags for her whole look, not only the pose. Mask her as she is now; the mask grows by itself over the new arms and legs.</span><input type="file" accept="image/*" id="ifile" hidden></div>` : ''}
        ${tool.inp.control !== 'off' ? `<label class="note" for="icstr">Pose strength ${tool.inp.cstr}</label><input id="icstr" type="range" min="0.3" max="1.5" step="0.05" value="${tool.inp.cstr}" style="padding:0">` : ''}
        <label class="note">Detailer, redoes them large afterwards</label><span class="chips row" style="margin:0">${[['off', 'Off'], ['face', 'Faces'], ['hands', 'Hands'], ['both', 'Both']].map(([k, t]) => `<button class="chip" data-idet="${k}" aria-pressed="${tool.inp.detail === k}">${t}</button>`).join('')}</span>
        ${tool.inp.detail !== 'off' ? `<label class="note" for="idstr">Detail change ${tool.inp.dstr}</label><input id="idstr" type="range" min="0.1" max="0.7" step="0.05" value="${tool.inp.dstr}" style="padding:0">` : ''}
        <details class="fixcard" id="iset" ${tool.inp.open ? 'open' : ''}><summary>Settings</summary>
          <div class="row" style="margin:0"><label class="note" style="flex:1">Seed<input id="iseed" inputmode="numeric" placeholder="random" value="${esc(String(tool.inp.seed))}"></label><label class="note" style="flex:1">Steps<input id="isteps" type="number" min="10" max="60" value="${tool.inp.steps}"></label><label class="note" style="flex:1">CFG<input id="icfg" type="number" min="1" max="10" step="0.5" value="${tool.inp.cfg}"></label></div></details>
        <div class="row" style="margin:0"><button id="ibrush" aria-pressed="${tool.active === 'inpaint'}">${tool.active === 'inpaint' ? 'Painting the inpaint mask' : 'Paint mask'}</button><button id="iclear">Clear mask</button><button id="irun" class="primary" ${p.busy || !top ? 'disabled' : ''}>Inpaint</button><button id="irunq" ${p.busy || !top ? 'disabled' : ''} title="Change pose, keep the outfit: Qwen redraws her in the pose picture's pose from the picture itself, then JANKU goes over it lightly with her LoRA. About two and a half minutes. Needs Change pose and a pose picture; a mask is optional.">Inpaint with Qwen</button></div>
      </section>
      <div class="lockhead"><h3 title="The first character is painted first">Characters</h3><span class="row" style="margin:0;gap:6px"><button class="mini" data-foldall="1">Fold all</button><button class="mini" data-foldall="0">Unfold all</button></span></div><div id="chars"></div>
      <div class="row"><select id="add" aria-label="Add LoRA" style="flex:1"><option value="">Add LoRA …</option>${p._loras.map(l => `<option value="${esc(l.id)}">${esc(l.name)}</option>`).join('')}</select></div>
      ${p._loras.length ? '' : '<p class="note">Finish a set first; its LoRA appears here.</p>'}
      <div class="row"><button class="go" id="all" ${p.busy ? 'disabled' : ''}>Paint all characters</button></div>
      <p class="note">Strength 0.6 keeps the pose, 0.8 brings back marks and outfits but can break anatomy. Put the character underneath first.</p></div></div>`;
  $('title').onchange = e => { p.title = e.target.value; savePanel(); };
  /* the panel's Qwen edit */
  const Q = tool.qwen;
  $('qtext').oninput = e => { Q.text = e.target.value; };
  document.querySelectorAll('[data-pqlora]').forEach(b => b.onclick = () => { const k = b.dataset.pqlora; Q.loras = Q.loras.includes(k) ? Q.loras.filter(x => x !== k) : [...Q.loras, k]; b.setAttribute('aria-pressed', Q.loras.includes(k)); });
  $('qbrush').onclick = () => { tool.erase = false; tool.active = tool.active === 'qwen' ? panel.characters[0]?.id ?? null : 'qwen'; syncPanel(); const keep = captureMasks(); paintPanel(); restoreMasks(keep); };
  $('qclear').onclick = () => { drawn.qwen?.getContext('2d').clearRect(0, 0, drawn.qwen.width, drawn.qwen.height); showMask(); };
  $('qref').onclick = async e => { if (e.target.closest('#qrefx')) return; const img = await pickRef(panel.characters); if (!img) return;
    const r = await act('panel_qwen_ref', { id:p.id, image:img }); if (r) { p.qwen_ref = r.qwen_ref; Q.refv = Date.now(); paintPanel(); } };
  $('qfile').onchange = () => { const f0 = $('qfile').files[0]; if (!f0) return; const rd = new FileReader();
    rd.onload = async () => { const r = await act('panel_qwen_ref', { id:p.id, image:rd.result }); if (r) { p.qwen_ref = r.qwen_ref; Q.refv = Date.now(); paintPanel(); } }; rd.readAsDataURL(f0); };
  if ($('qrefx')) $('qrefx').onclick = async e => { e.stopPropagation(); const r = await act('panel_qwen_ref', { id:p.id }); if (r) { delete p.qwen_ref; paintPanel(); } };
  const runQ = janku => async () => { Q.text = $('qtext').value; if (!Q.text.trim()) return alert('Write in a sentence what should change.');
    const m = drawn.qwen, has = m && m.getContext('2d').getImageData(0, 0, m.width, m.height).data.some((v, i) => i % 4 === 3 && v); let mask = null;
    if (has) { const t = document.createElement('canvas'); t.width = m.width; t.height = m.height; const g = t.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, t.width, t.height); g.drawImage(m, 0, 0); mask = t.toDataURL('image/png'); }
    syncPanel(); await savePanel(); const r = await act('panel_qwen', { id:p.id, text:Q.text, loras:Q.loras, mask, janku, janku_loras:janku ? tool.inp.loras : [], negative:janku ? $('ineg').value : '' });
    if (r) { panel.busy = r.busy; paintPanel(); pollPanel(); } };
  $('qrun').onclick = runQ(false); $('qrunj').onclick = runQ(true);
  /* the panel's own SDXL inpaint */
  const I = tool.inp, syncI = () => { I.text = $('itext').value; I.negative = $('ineg').value; I.seed = $('iseed').value.trim(); I.steps = +$('isteps').value || 30; I.cfg = +$('icfg').value || 5; I.open = $('iset').open; };
  const repaintI = () => { syncI(); syncPanel(); const keep = captureMasks(); paintPanel(); restoreMasks(keep); };
  ['itext', 'ineg', 'iseed', 'isteps', 'icfg'].forEach(k => $(k).oninput = syncI);
  const slide = (id, key, word) => { if ($(id)) $(id).oninput = e => { I[key] = +e.target.value; e.target.previousElementSibling.textContent = word + ' ' + I[key]; }; };
  slide('iden', 'denoise', 'Change'); slide('icstr', 'cstr', 'Pose strength'); slide('idstr', 'dstr', 'Detail change');
  document.querySelectorAll('[data-ilora]').forEach(b => b.onclick = () => { const id = b.dataset.ilora, l = p._loras.find(x => x.id === id);
    if (I.loras.some(x => x.id === id)) I.loras = I.loras.filter(x => x.id !== id); else if (I.loras.length < 3) I.loras = [...I.loras, { id, strength:l.strength, outfit:!!l.pills?.outfit?.[0] }]; repaintI(); });
  document.querySelectorAll('[data-iout]').forEach(b => b.onclick = () => { const x = I.loras.find(y => y.id === b.dataset.iout); x.outfit = !x.outfit; repaintI(); });
  document.querySelectorAll('[data-istr]').forEach(r => r.oninput = e => { const x = I.loras.find(y => y.id === r.dataset.istr); x.strength = +e.target.value; e.target.previousElementSibling.textContent = (p._loras.find(l => l.id === x.id)?.name || 'LoRA') + ' ' + x.strength; });
  document.querySelectorAll('[data-ictl]').forEach(b => b.onclick = () => { I.control = b.dataset.ictl; if (I.control === 'ref' && I.denoise < 0.9) I.denoise = 1; /* below that the old pose in the picture wins over the skeleton */ repaintI(); });
  document.querySelectorAll('[data-idet]').forEach(b => b.onclick = () => { I.detail = b.dataset.idet; repaintI(); });
  $('ibrush').onclick = () => { tool.erase = false; tool.active = tool.active === 'inpaint' ? panel.characters[0]?.id ?? null : 'inpaint'; repaintI(); };
  $('iclear').onclick = () => { drawn.inpaint?.getContext('2d').clearRect(0, 0, drawn.inpaint.width, drawn.inpaint.height); showMask(); };
  if ($('iref')) { $('iref').onclick = e => { if (!e.target.closest('#irefx')) $('ifile').click(); };
    $('ifile').onchange = () => { const f0 = $('ifile').files[0]; if (!f0) return; const rd = new FileReader();
      rd.onload = async () => { syncI(); const r = await act('panel_pose_ref', { id:p.id, image:rd.result }); if (r) { p.pose_ref = r.pose_ref; I.refv = Date.now(); repaintI(); } }; rd.readAsDataURL(f0); };
    if ($('irefx')) $('irefx').onclick = async e => { e.stopPropagation(); const r = await act('panel_pose_ref', { id:p.id }); if (r) { delete p.pose_ref; repaintI(); } }; }
  const runI = qwen => async () => { syncI(); if (!I.text.trim()) return alert(qwen ? 'Write the new pose in a few words.' : 'Write in tags what should be there.');
    if (qwen && I.control !== 'ref') return alert('Inpaint with Qwen changes the pose and keeps the outfit: pick Change pose and add a pose picture.');
    const m = drawn.inpaint, has = m && m.getContext('2d').getImageData(0, 0, m.width, m.height).data.some((v, i) => i % 4 === 3 && v); let mask = null;
    if (I.control === 'ref' && !p.pose_ref) return alert('Add a pose picture first.');
    if (I.control === 'ref' && !has && !qwen) return alert('Paint a mask over the character who takes the pose.');
    if (has) { const t = document.createElement('canvas'); t.width = m.width; t.height = m.height; const g = t.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, t.width, t.height); g.drawImage(m, 0, 0); mask = t.toDataURL('image/png'); }
    syncPanel(); await savePanel();
    const r = await act('panel_inpaint', { id:p.id, text:I.text, negative:I.negative, denoise:I.denoise, loras:I.loras, control:I.control, control_strength:I.cstr, detail:I.detail, detail_strength:I.dstr, seed:I.seed, steps:I.steps, cfg:I.cfg, mask, qwen });
    if (r) { panel.busy = r.busy; paintPanel(); pollPanel(); } };
  $('irun').onclick = runI(false); $('irunq').onclick = runI(true);
  $('delp').onclick = async () => { if (confirm('Delete this panel and its versions?')) { await act('panel_delete', { id:p.id }); location.hash = book ? 'book/' + book.id : 'manga'; } };
  $('pmotion').onchange = async e => { p.motion = e.target.value; await act('panel_edit', { id:p.id, motion:p.motion }); };
  if (!p.locked) { ['prompt'].forEach(k => $(k).onchange = e => { p[k] = e.target.value; savePanel(); }); $('neg').onchange = e => { p.negative = e.target.value; savePanel(); }; }
  else { $('ptags').onchange = e => { p.pass_tags = e.target.value; savePanel(); }; $('pneg').onchange = e => { p.pass_negative = e.target.value; savePanel(); }; }
  const HEADCOUNT = /^(\d\+?|multiple|solo)[ _]?(girls?|boys?|others?)$|^solo$|^(group|crowd)$/;
  if ($('lock')) $('lock').onclick = async () => { syncPanel(); Object.assign(p, { locked:true, pass_tags:p.pass_tags || p.tags.split(',').map(t => t.trim()).filter(t => t && !HEADCOUNT.test(t.toLowerCase())).join(', '),
    pass_negative:p.pass_negative || ['nsfw', 'nude', ...p.negative.split(',').map(t => t.trim()).filter(t => t && !['loli', 'child', 'nsfw', 'nude'].includes(t.toLowerCase()))].join(', ') }); /* loli, child are locked; nsfw, nude editable */ await act('panel_edit', { id:p.id, locked:true, pass_tags:p.pass_tags, pass_negative:p.pass_negative }); paintPanel(); };
  if ($('unlock')) $('unlock').onclick = async () => { syncPanel(); p.locked = false; await act('panel_edit', { id:p.id, locked:false, pass_tags:p.pass_tags, pass_negative:p.pass_negative }); paintPanel(); };
  if (!p.locked) { $('size-p').value = `${p.width},${p.height}`; $('size-p').onchange = e => { [p.width, p.height] = e.target.value.split(',').map(Number); savePanel(); };
  const compose = () => async () => { p.prompt = $('prompt').value; p.negative = $('neg').value; await savePanel(); const r = await act('create_compose', { id:p.id, count:sketchCount(), loras:withLoras() && p.characters.length > 0 }); if (r) pollPanel(); if (r) { Object.assign(panel, r); paintPanel(); } };
  $('comp').onclick = compose(); }
  if ($('snipedit')) wireSnips(p);
  if ($('tog')) $('tog').onclick = () => { const on = !withLoras(); try { localStorage.setItem('createLoras', on ? '1' : ''); } catch {} $('tog').setAttribute('aria-pressed', String(on)); livePrompt(); };
  if ($('sk-dice')) $('sk-dice').onclick = async () => { const r = await act('create_compose', { id:p.id, count:sketchCount(), loras:withLoras() && p.characters.length > 0 }); if (r) { Object.assign(panel, r); paintPanel(); pollPanel(); } };
  app.querySelectorAll('[data-pick]').forEach(b => b.onkeydown = e => { if (e.target === b && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); b.click(); } });
  app.querySelectorAll('[data-pick]').forEach(b => b.onclick = () => zoomSketch(p, +b.dataset.i));
  if ($('sk-back')) $('sk-back').onclick = async () => { const r = await act('create_unpick', { id:p.id }); if (r) { Object.assign(panel, r); paintPanel(); } };
  app.querySelector('.skgrid')?.style.setProperty('--skar', String(p.width / p.height));
  if ($('skn')) $('skn').onchange = e => { try { localStorage.setItem('createSketches', e.target.value); } catch {} if ($('sk-dice')) $('sk-dice').textContent = `🎲 ${sketchCount()} new`; };
  // Show prompt: the raw text, live while you type in Prompt or Negative (one request a quarter second after the last key)
  const livePrompt = async () => { const v = $('promptview'); if (!v || v.hidden) return;
    const r = await post('create_prompt', { id:p.id, loras:withLoras() && p.characters.length > 0, prompt:$('prompt')?.value ?? p.prompt, negative:$('neg')?.value ?? p.negative }).catch(() => null); if (!r || v.hidden) return;
    if (!$('pvpos')) v.innerHTML = `<label class="note" id="pvhead"></label><textarea id="pvpos" readonly rows="4"></textarea><label class="note">Negative</label><textarea id="pvneg" readonly rows="3"></textarea>`;
    $('pvhead').textContent = `Positive · ${r.model}, ` + (r.loras?.length ? 'LoRAs: ' + r.loras.map(l => `${l.name} ${l.strength}`).join(', ') : 'no LoRA');
    $('pvpos').value = r.positive; $('pvneg').value = r.negative; };
  let liveT; ['prompt', 'neg'].forEach(k => $(k) && $(k).addEventListener('input', () => { clearTimeout(liveT); liveT = setTimeout(livePrompt, 250); }));
  if ($('showprompt')) $('showprompt').onclick = () => { const v = $('promptview'); v.hidden = !v.hidden; try { localStorage.setItem('createShowPrompt', v.hidden ? '' : '1'); } catch {} if (!v.hidden) livePrompt(); };
  try { if (localStorage.getItem('createShowPrompt') && $('promptview')) { $('promptview').hidden = false; livePrompt(); } } catch {}
  const run = (action, body) => async () => { const r = await act(action, { id:p.id, ...body() }); if (r) { Object.assign(panel, r); paintPanel(); } };
  if ($('castb')) $('castb').onclick = run('panel_cast', () => ({}));
  if ($('finb')) $('finb').onclick = run('panel_finish', () => ({}));
  if ($('magicb')) { $('magicb').onclick = run('panel_magic', () => ({ text:$('magic').value })); $('magic').onkeydown = e => { if (e.key === 'Enter') $('magicb').click(); }; }
  $('add').onchange = e => { const l = p._loras.find(x => x.id === e.target.value); if (!l) return; const c = { id:`c${Date.now()}`, lora:l.id, name:l.name, trigger:l.trigger, file:l.file, strength:l.strength, denoise:.62, action:l.look ? l.look + ', ' : '', negative:'', identity:true, outfit:!!l.pills?.outfit?.[0] };
    pillTags(c, l.pills?.identity, true, ['action', 'negative']); pillTags(c, l.pills?.outfit, c.outfit, ['action', 'negative'], true); p.characters.push(c); tool.active = p.characters.at(-1).id; savePanel(); paintPanel(); };
  p.characters.forEach((c, i) => { const w = document.createElement('div'); w.className = 'char' + (tool.active === c.id ? ' on' : '');
    w.classList.add('box'); if (folds.has('char:' + c.id)) w.classList.add('min'); w.dataset.box = 'char:' + c.id;
    w.innerHTML = `<div class="row boxhead">${foldBtn('char:' + c.id)}<span class="dot" style="background:${COLORS[i % 4]}"></span><b style="flex:1">${esc(c.name)}</b><span class="note">${c.masked ? 'mask saved' : 'no mask'}</span><span class="icons"><button class="ico" data-p aria-pressed="${tool.active === c.id}" title="Paint her mask" aria-label="Paint mask"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 21c3 0 5-1.5 5-4a3 3 0 0 0-3-3c-1.5 0-2.5 1-2.5 2.5 0 2-1 3-1.5 4.5zM8.5 14.5 20 3l1 1-11.5 11.5"/></svg></button><button class="ico" data-pass ${p.busy ? 'disabled' : ''} title="Redo her pass" aria-label="Redo this pass"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 12a8 8 0 1 1-2.3-5.6M20 4v5h-5"/></svg></button><button class="ico" data-reset title="Reset description and negative to her defaults (click twice)" aria-label="Reset to defaults"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 12a8 8 0 1 0 2.3-5.6M4 4v5h5"/><path d="M12 8v4l3 2"/></svg></button><button class="ico danger" data-r title="Remove her from this panel" aria-label="Remove"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg></button></span></div>
      ${pillsOf(c) ? `<div class="row chips cpills"><button class="chip" data-pill="identity" aria-pressed="${c.identity === true}" title="Her calibrated negatives go into her negative below">Identity</button><button class="chip" data-pill="outfit" aria-pressed="${!!c.outfit}" title="Her signature outfit and its trigger go into her description, its negatives into hers">Her outfit</button></div>` : ''}
      <textarea data-action id="act-${c.id}" rows="3" placeholder="Her look and what she does here">${esc(c.action)}</textarea>
      <label class="note" for="cneg-${c.id}">Her negative, added to the panel's in her pass</label><textarea data-cneg id="cneg-${c.id}" rows="2" placeholder="e.g. black hair, long sleeves">${esc(c.negative || '')}</textarea>
      <label class="note">LoRA ${c.strength}</label><input data-str type="range" min=".3" max="1.3" step=".05" value="${c.strength}" style="padding:0">
      <label class="note">Change ${c.denoise}</label><input data-den type="range" min=".3" max=".95" step=".01" value="${c.denoise}" style="padding:0">
      <details class="fixcard" data-fixd ${tool.fix.open.has(c.id) ? 'open' : ''}><summary>Fix a spot on her</summary>
        <p class="note">Uses her card as set above. Paint her mask over just the spot, add a few words, Fix: only that spot of the shown version is repainted.</p>
        <span class="chips row" style="margin:0">${[['small', 'Small change', 'Face, smile, a detail: light, keeps the shapes'], ['sharp', 'Resharpen', 'Blurry face, hair or outfit: repainted larger and lightly with detail tags; shapes and expression stay'], ['outfit', 'Her outfit', 'Washes the old colours out first; adds her outfit tags if the pill is off'], ['pose', 'Pose', 'An arm, a hand, posture: strong; mask the old and the new place'], ['qwen', 'Qwen edit', 'A plain sentence; Qwen-Image-Edit repaints the masked spot, with a reference picture']].map(([k, t, d]) => `<button class="chip" data-preset="${k}" aria-pressed="${fx(c).preset === k}" title="${d}">${t}</button>`).join('')}</span>
        ${fx(c).preset === 'qwen' ? `<div class="qrow"><textarea data-ftext rows="3" placeholder="She wears the orange cropped jacket from the reference.">${esc(fx(c).text)}</textarea>
          <button class="qref" data-qref title="${c.ref ? 'Your reference: click to change' : 'Her base image: click to use your own picture, e.g. an outfit or an object'}" aria-label="Reference picture">${refOf(c) ? `<img src="${refOf(c)}" alt="">` : '<span>+ ref</span>'}${c.ref ? '<i data-qrefx title="Back to her base image" role="button">✕</i>' : ''}</button><input type="file" accept="image/*" data-qfile hidden></div>
        <span class="chips row" style="margin:0">${QWEN_LORAS.map(([k, t]) => `<button class="chip" data-qlora="${k}" aria-pressed="${fx(c).loras.includes(k)}">${t}</button>`).join('')}</span>`
        : `<textarea data-ftext rows="2" placeholder="${fx(c).preset === 'outfit' ? 'Optional: extra outfit tags' : fx(c).preset === 'pose' ? 'arm up, hand on own head, stretching' : fx(c).preset === 'sharp' ? 'Optional: adds to detailed face, eyes, hair, clothes, body, sharp focus' : 'smile, open mouth'}">${esc(fx(c).text)}</textarea>
        <label class="note">Fix change ${fx(c).denoise}</label><input data-fden type="range" min="0.2" max="1" step="0.05" value="${fx(c).denoise}" style="padding:0">`}
        <div class="row" style="margin:0"><button data-fbrush aria-pressed="${tool.active === c.id}">${tool.active === c.id ? 'Painting her mask' : 'Paint mask'}</button><button data-fclear>Clear mask</button><button data-frun class="primary" ${p.busy ? 'disabled' : ''}>Fix</button></div>
      </details>
`;
    w.querySelector('[data-p]').onclick = () => openChar(c.id);
    /* fix a spot on her: her own mask, painted over just the spot; preset, words and Change per character */
    const F = fx(c), d = w.querySelector('[data-fixd]');
    d.ontoggle = () => { d.open ? tool.fix.open.add(c.id) : tool.fix.open.delete(c.id); };
    w.querySelectorAll('[data-preset]').forEach(b => b.onclick = () => { F.text = w.querySelector('[data-ftext]').value; F.preset = b.dataset.preset; F.denoise = { small:0.5, sharp:0.35, outfit:0.85, pose:0.9, qwen:1 }[F.preset]; paintPanel(); });
    w.querySelectorAll('[data-qlora]').forEach(b => b.onclick = () => { const k = b.dataset.qlora; F.text = w.querySelector('[data-ftext]').value; F.loras = F.loras.includes(k) ? F.loras.filter(x => x !== k) : [...F.loras, k]; paintPanel(); });
    const qf = w.querySelector('[data-qfile]');
    if (qf) { w.querySelector('[data-qref]').onclick = async e => { if (e.target.closest('[data-qrefx]')) return; const img = await pickRef([c]); if (!img) return;
        F.text = w.querySelector('[data-ftext]').value; syncPanel(); await savePanel();
        const r = await act('panel_ref', { id:panel.id, character:c.id, image:img }); if (r) { c.ref = r.characters.find(x => x.id === c.id)?.ref; c._refv = Date.now(); paintPanel(); } };
      qf.onchange = () => { const f0 = qf.files[0]; if (!f0) return; const rd = new FileReader(); rd.onload = async () => { F.text = w.querySelector('[data-ftext]').value; syncPanel(); await savePanel();
        const r = await act('panel_ref', { id:panel.id, character:c.id, image:rd.result }); if (r) { c.ref = r.characters.find(x => x.id === c.id)?.ref; c._refv = Date.now(); paintPanel(); } }; rd.readAsDataURL(f0); };
      const x = w.querySelector('[data-qrefx]'); if (x) x.onclick = async e => { e.stopPropagation(); const r = await act('panel_ref', { id:panel.id, character:c.id }); if (r) { delete c.ref; paintPanel(); } }; }
    w.querySelector('[data-ftext]').oninput = e => { F.text = e.target.value; };
    w.querySelector('[data-fden]') && (w.querySelector('[data-fden]').oninput = e => { F.denoise = +e.target.value; e.target.previousElementSibling.textContent = 'Fix change ' + F.denoise; });
    w.querySelector('[data-fbrush]').onclick = () => openChar(c.id);
    w.querySelector('[data-fclear]').onclick = () => { const m = maskOf(c.id); m?.getContext('2d').clearRect(0, 0, m.width, m.height); showMask(); };
    w.querySelector('[data-frun]').onclick = () => run();
    const run = async (over) => { const m = maskOf(c.id); if (!m || !m.getContext('2d').getImageData(0, 0, m.width, m.height).data.some((v, i) => i % 4 === 3 && v)) return alert('Paint the spot on her first.');
      const t = document.createElement('canvas'); t.width = m.width; t.height = m.height; const g = t.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, t.width, t.height); g.drawImage(m, 0, 0);
      F.text = w.querySelector('[data-ftext]').value; syncPanel(); await savePanel();
      const r = await act('panel_fix', { id:panel.id, character:c.id, preset:F.preset, text:F.text, denoise:F.denoise, loras:F.loras, ...over, mask:t.toDataURL('image/png') });
      if (r) { panel.busy = r.busy; paintPanel(); pollPanel(); } };
    /* reset: what she gets when added (her look, identity pill on, outfit off); first click arms, second within 3 s resets */
    w.querySelector('[data-reset]').onclick = e => { const x = e.currentTarget;
      if (!x.classList.contains('armed')) { x.classList.add('armed'); x.title = 'Click again to reset'; clearTimeout(x._t); x._t = setTimeout(() => { x.classList.remove('armed'); x.title = 'Reset description and negative to her defaults (click twice)'; }, 3000); return; }
      clearTimeout(x._t); const l = p._loras?.find(y => y.id === c.lora || y.file === c.file || y.name === c.name);
      Object.assign(c, { action:l?.look ? l.look + ', ' : '', negative:'', identity:true, outfit:false }); pillTags(c, l?.pills?.identity, true, ['action', 'negative']); savePanel(); paintPanel(); };
    w.querySelector('[data-r]').onclick = () => { p.characters.splice(i, 1); savePanel(); paintPanel(); };
    w.querySelector('[data-action]').onchange = e => { c.action = e.target.value; savePanel(); };
    w.querySelector('[data-cneg]').onchange = e => { c.negative = e.target.value; savePanel(); };
    w.querySelectorAll('[data-pill]').forEach(b => b.onclick = () => { const k = b.dataset.pill, on = !c[k];  // cards from before the pills have neither flag: both start off
      c.action = w.querySelector('[data-action]').value; c.negative = w.querySelector('[data-cneg]').value;
      pillTags(c, pillsOf(c)?.[k], on, ['action', 'negative'], k === 'outfit'); c[k] = on; savePanel(); paintPanel(); });
    w.querySelector('[data-str]').onchange = e => { c.strength = +e.target.value; savePanel(); paintPanel(); };
    w.querySelector('[data-den]').onchange = e => { c.denoise = +e.target.value; savePanel(); paintPanel(); };
    w.querySelector('[data-pass]').onclick = async () => { syncPanel(); await savePanel(); await saveMasks(); const r = await act('panel_pass', { id:p.id, character:c.id, from:shownOf(p) }); if (r) { Object.assign(panel, r); paintPanel(); } };
    $('chars').append(w); });
  tagSearch(Object.fromEntries([['prompt', 'prompt'], ['neg', 'negative'], ['ptags', 'pass_tags'], ['pneg', 'pass_negative'], ['itext', 'inp:text'], ['ineg', 'inp:negative'], ...p.characters.map(c => ['act-' + c.id, c.id]), ...p.characters.map(c => ['cneg-' + c.id, 'neg:' + c.id])]), (key, value) => {
    const cn = key.startsWith('neg:') && p.characters.find(x => x.id === key.slice(4)), c = p.characters.find(x => x.id === key); if (key.startsWith('inp:')) { tool.inp[key.slice(4)] = value; return; } if (cn) cn.negative = value; else if (c) c.action = value; else p[key] = value; savePanel(); });
  $('all').onclick = async () => { syncPanel(); await savePanel(); await saveMasks(); const comp = p.composition; const r = await act('panel_pass', { id:p.id, all:true, from:comp }); if (r) { Object.assign(panel, r); paintPanel(); } };
  if (top) setupPainter();
}
function setupPainter() {
  const img = $('pic'), cv = $('cv');
  showBusy();
  const ready = () => { cv.width = img.naturalWidth; cv.height = img.naturalHeight; showMask(); };
  img.complete ? ready() : img.onload = ready;
  let down = false, last = null;
  const at = e => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height]; };
  const stroke = e => { const m = maskOf(tool.active); if (!m) return; const c = m.getContext('2d'), [x, y] = at(e); c.globalCompositeOperation = tool.erase ? 'destination-out' : 'source-over'; c.strokeStyle = c.fillStyle = '#fff'; c.lineCap = 'round'; c.lineWidth = tool.size * cv.width / cv.getBoundingClientRect().width;
    c.beginPath(); c.moveTo(...(last || [x, y])); c.lineTo(x, y); c.stroke(); last = [x, y]; showMask(); };
  cv.onpointerdown = e => { if (!tool.active) return; down = true; last = null; cv.setPointerCapture(e.pointerId); stroke(e); };
  // the brush itself as the cursor: its real size, in the colour of the character being painted (dashed when erasing)
  const ring = document.createElement('i'); ring.className = 'brushring'; cv.parentElement.append(ring);
  const showRing = e => { const i = panel.characters.findIndex(x => x.id === tool.active), qw = tool.active === 'qwen' || tool.active === 'inpaint'; if (i < 0 && !qw) { ring.hidden = true; cv.style.cursor = 'not-allowed'; return; }
    const r = cv.getBoundingClientRect(), col = tool.active === 'inpaint' ? '#ffb070' : qw ? '#e8e0ff' : COLORS[i % 4]; cv.style.cursor = 'none'; ring.hidden = false;
    Object.assign(ring.style, { width:tool.size + 'px', height:tool.size + 'px', left:(e.clientX - r.left) + 'px', top:(e.clientY - r.top) + 'px',
      borderColor:tool.erase ? '#fff' : col, borderStyle:tool.erase ? 'dashed' : 'solid', background:tool.erase ? 'transparent' : col + '40' }); };
  cv.onpointermove = e => { showRing(e); if (down) stroke(e); };
  cv.onpointerleave = () => { ring.hidden = true; };
  cv.onpointerup = () => { down = false; };
  $('size').oninput = e => tool.size = +e.target.value;
  $('eye').onclick = () => { tool.hide = !tool.hide; $('stage').classList.toggle('nomask', tool.hide); $('eye').setAttribute('aria-pressed', tool.hide);
    $('eye').title = tool.hide ? 'Show the masks' : 'Hide the masks'; };  // hidden masks can't be painted on; show them to paint again
  $('erase').onclick = () => { tool.erase = !tool.erase; $('erase').textContent = tool.erase ? 'Erasing' : 'Erase'; $('erase').setAttribute('aria-pressed', tool.erase); };
  $('clear').onclick = () => { const m = maskOf(tool.active); m?.getContext('2d').clearRect(0, 0, m.width, m.height); showMask(); };
  $('savem').onclick = saveMasks;
  // newest first, numbered from the oldest; the shown one is framed. Clicking only picks, the order never changes.
  const vs = $('vs'), order = [...panel.versions].reverse(), num = Object.fromEntries(order.map((v, i) => [v.file, i + 1]));
  panel.versions.forEach(v => { const b = document.createElement('button'), on = v.file === shownOf(panel); b.setAttribute('aria-pressed', on);
    b.title = on ? 'Shown in the manga; the next pass paints on this one' : 'Use this version in the manga and paint on it next';
    b.innerHTML = `<img src="${file(dir(), v.file)}" alt=""><span>${num[v.file]}. ${esc(v.label === 'Composition' || v.kind === 'fix' ? v.label : v.label + ' pass')}${v.denoise ? `, ${v.denoise}` : ''}${v.from && num[v.from] ? ` <small>on ${num[v.from]}</small>` : ''}</span>${on ? '<em>In the manga</em>' : ''}<i class="vdel" role="button" tabindex="0" title="Delete this version" aria-label="Delete version ${num[v.file]}">✕</i>`;
    /* two clicks: the first arms the cross (red, "Delete?"), a second within 3 s deletes */
    b.querySelector('.vdel').onclick = async e => { e.stopPropagation(); const x = e.currentTarget; if (panel.busy) return;
      if (!x.classList.contains('armed')) { x.classList.add('armed'); x.textContent = 'Delete?'; x.title = 'Click again to delete; the file goes to the panel trash folder';
        clearTimeout(x._t); x._t = setTimeout(() => { x.classList.remove('armed'); x.textContent = '✕'; x.title = 'Delete this version'; }, 3000); return; }
      clearTimeout(x._t);
      const r = await act('panel_version_delete', { id:panel.id, file:v.file }); if (r) { panel.versions = r.versions; panel.current = r.current; panel.composition = r.composition; paintPanel(); } };
    b.onclick = async () => { if (on) return; const r = await act('panel_edit', { id:panel.id, current:v.file }); if (r) { panel.current = r.current; paintPanel(); } }; vs.append(b); });
}
function maskOf(id) { if (!id || !$('cv')) return null; if (!drawn[id]) { const c = document.createElement('canvas'); c.width = $('cv').width; c.height = $('cv').height; drawn[id] = c;
    const c0 = panel.characters.find(x => x.id === id); if (c0?.masked) { const im = new Image(); im.onload = () => { const t = document.createElement('canvas'); t.width = c.width; t.height = c.height; const g = t.getContext('2d'); g.drawImage(im, 0, 0, c.width, c.height); const d = g.getImageData(0, 0, c.width, c.height); for (let i = 0; i < d.data.length; i += 4) { d.data[i + 3] = d.data[i]; d.data[i] = d.data[i + 1] = d.data[i + 2] = 255; } c.getContext('2d').putImageData(d, 0, 0); showMask(); }; im.src = file(dir(), `mask_${id}.png`) + '?' + Date.now(); } }
  return drawn[id]; }
function showMask() { const cv = $('cv'); if (!cv) return; const c = cv.getContext('2d'); c.clearRect(0, 0, cv.width, cv.height);
  [...panel.characters, { id:'qwen' }, { id:'inpaint' }].forEach((ch, i) => { const q = ch.id === 'qwen' || ch.id === 'inpaint', m = q ? drawn[ch.id] : maskOf(ch.id); if (!m) return; const t = document.createElement('canvas'); t.width = cv.width; t.height = cv.height; const g = t.getContext('2d'); g.drawImage(m, 0, 0); g.globalCompositeOperation = 'source-in'; g.fillStyle = ch.id === 'inpaint' ? '#ffb070' : q ? '#e8e0ff' : COLORS[i % 4]; g.fillRect(0, 0, t.width, t.height); c.drawImage(t, 0, 0); }); }
async function saveMasks() { for (const ch of panel.characters) { const m = drawn[ch.id]; if (!m) continue; const t = document.createElement('canvas'); t.width = m.width; t.height = m.height; const g = t.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, t.width, t.height); g.drawImage(m, 0, 0); if (!m.getContext('2d').getImageData(0, 0, m.width, m.height).data.some((v, i) => i % 4 === 3 && v)) continue;
    const r = await act('panel_mask', { id:panel.id, character:ch.id, mask:t.toDataURL('image/png') }); if (r) ch.masked = true; } }

// ---------- routes ----------
if (parent !== window) document.body.classList.add('embedded');  // inside Morpheus the top bar carries this navigation
return { open: openPanel, stop: () => clearTimeout(timer) };
})();
