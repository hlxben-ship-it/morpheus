// glia-avatar.js: the player for glia-agent packages (pixel-art agents made in Morpheus' Soulforge).
// One file, no dependencies, plain ES module. It draws an agent alive on a canvas: breathing, hair and cloak in the
// air, blinks and glances, moods (a mouth, eyes, a blush, a gesture of the head, an arm, an icon), speech on the
// mouth, and her worlds behind her with their light, weather and fog.
//
//   import { GliaAvatar } from './glia-avatar.js';
//   const glia = await GliaAvatar.load(document.querySelector('#avatar'), './glia/');
//   glia.mood('happy'); glia.say('Hello!'); glia.think(true); glia.scene('dream');
//
// Morpheus renders its own agents with this same file, so what you see there is what a product gets.

export const VERSION = '2.0.0';
export const FORMATS = ['glia-agent/1', 'glia-agent/2'];  // read; older ones are migrated in memory
export const FORMAT = 'glia-agent/2';                     // what migrate() returns

// ---------- the format ----------
// Each step turns one version's manifest into the next. An export never changes on disk; the player upgrades what
// it read. A newer format than this player knows fails loudly: half a render would hide the problem.
const MIGRATIONS = {
  'glia-agent/1': m => {
    const scene = m.scene || {};
    const worlds = {};
    if (scene.world_sky) worlds.home = { name: `${m.name}'s world`, light: scene.light || null,
      layers: { sky: scene.world_sky, far: scene.world_far, near: scene.world_near } };
    const { mood_recipes, ...rest } = m;
    return { ...rest, format: 'glia-agent/2', moods: mood_recipes || {},
      scene: { default: worlds.home ? 'home' : null, cut: scene.cut || null, emblem: scene.emblem || null, worlds } };
  },
};
export function migrate(manifest) {
  let m = manifest;
  for (let k = 0; k < 10 && m.format !== FORMAT; k++) {
    const step = MIGRATIONS[m.format];
    if (!step) throw new Error(`glia-avatar ${VERSION} can't read "${m.format}" (it reads ${FORMATS.join(', ')}). Use a newer glia-avatar.js.`);
    m = step(m);
  }
  return m;
}

// ---------- shared motion: one loop for every scene on the page ----------
// reduced motion: the scene and the figure hold still; blinks and expressions stay
const calm = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
let lean = 0, leanTo = 0, gust = 0, gustAt = performance.now() + 9000;
const ptr = { x: -1e4, y: -1e4, t: 0 };
addEventListener('pointermove', e => { leanTo = (e.clientX / innerWidth - .5) * 2; Object.assign(ptr, { x: e.clientX, y: e.clientY, t: performance.now() }); }, { passive: true });
const scenes = new Set();  // the mounted scenes: { el, back, front, light, weather, figure }
const ease = x => x * x * (3 - 2 * x);
const DRIFT = { sky: [6, .05, 0], far: [16, .07, 1.3], near: [30, .09, 2.1] };
// the air in front of her: the theme's own and, if asked for, a weather on top
export const WEATHER = {
  snow: { n: 40, col: ['#ffffff', '#dfefff'], vy: 9, vx: -1.5, sway: 2.2, big: .35 },
  petals: { n: 22, col: ['#ffb7d5', '#ff9ec4', '#ffe0ee'], vy: 6, vx: 4, sway: 3, petal: 1 },
  embers: { n: 26, col: ['#ffb347', '#ff7a2f', '#ffe08a'], vy: -7, vx: 1.5, sway: 1.8, twinkle: 1, glow: 1 },
  rain: { n: 50, col: ['#a9c8ff', '#d6e6ff'], vy: 40, vx: -5, streak: 4 },
  leaves: { n: 16, col: ['#e0782f', '#c9542a', '#f0b040'], vy: 7, vx: 5, sway: 3.5, petal: 1 },
  fireflies: { n: 14, col: ['#e6ff8a', '#fff3a0'], vy: -.6, vx: .8, twinkle: 1, wander: 1 },
  dust: { n: 30, col: ['#fff6d8', '#ffe3a8'], vy: -.6, vx: 2.2, sway: 1, twinkle: 1 },
};
(function drift(now) {
  requestAnimationFrame(drift);
  const t = now / 1000;
  lean += (leanTo - lean) * .04;
  // a gust now and then: it swells and fades over three seconds, and everything in the air feels it at once
  const g = (now - gustAt) / 3000;
  gust = g < 0 ? 0 : g < 1 ? Math.sin(Math.PI * g) ** 2 : 0;
  if (g >= 1) gustAt = now + 12000 + Math.random() * 14000;
  const dt = Math.min(.1, t - (drift.last || t)); drift.last = t;
  if (calm) return;
  for (const s of scenes) {
    if (!s.el.isConnected) { if (!s.keep) scenes.delete(s); continue; }
    const w = s.el.clientWidth / 340;  // the same swing at every card size
    for (const [l, [amp, sp, ph]] of Object.entries(DRIFT)) {
      const el = s.el.querySelector('.ga-' + l);
      // a mirrored layer slides mirrored too: its lean is turned back so every layer leans the same way
      if (el) el.style.backgroundPosition = `calc(${el.dataset.x || 50}% + ${Math.round(w * (amp * Math.sin(t * sp * 2 * Math.PI + ph) - (el.style.transform ? -1 : 1) * lean * amp * .6))}px) 100%`;
    }
    const fig = s.host.querySelector('.ga-live, .ga-still');
    if (fig) fig.style.translate = `${Math.round(-lean * 7 * w)}px 0`;
    const m = s.el.querySelector('.ga-motif');
    if (m) m.style.transform = `translate(${Math.round(-lean * 4 * w)}px, ${Math.round(w * 5 * Math.sin(t * .9))}px)`;
    sky(s, now, dt);  // first: it clears the back canvas
    air(s, t, dt);
  }
})(0);
// now and then a shooting star crosses a night sky, behind the far layer
function sky(s, now, dt) {
  const cv = s.back, g = cv.getContext('2d');
  if (s.sky.day) { g.clearRect(0, 0, cv.width, cv.height); if (s.storm) storm(s, now); return; }
  const st = s.star ||= { next: now + 2000 + Math.random() * 9000, s: null };
  if (!st.s && now > st.next) {
    const dir = Math.random() < .5 ? 1 : -1;
    st.s = { x: dir > 0 ? Math.random() * 60 : 79 + Math.random() * 60, y: 4 + Math.random() * 40, vx: dir * (55 + Math.random() * 30), vy: 16 + Math.random() * 12, life: 0, max: .7 + Math.random() * .5 };
  }
  g.clearRect(0, 0, cv.width, cv.height);
  if (s.storm) storm(s, now);
  const z = st.s;
  if (!z) return;
  z.life += dt; z.x += z.vx * dt; z.y += z.vy * dt;
  const a = Math.sin(Math.PI * Math.min(1, z.life / z.max));
  g.globalAlpha = a * .35; g.fillStyle = '#dff0ff'; g.fillRect(Math.round(z.x) - 1, Math.round(z.y) - 1, 3, 3);
  for (let k = 0; k < 15; k++) {
    g.globalAlpha = a * (1 - k / 15) ** 1.4;
    g.fillStyle = k ? '#bfe3ff' : '#ffffff';
    g.fillRect(Math.round(z.x - z.vx / 90 * k), Math.round(z.y - z.vy / 90 * k), 1, 1);
  }
  g.globalAlpha = 1;
  if (z.life > z.max) { st.s = null; st.next = now + (s.sky.nonear ? 2500 : 7000) + Math.random() * (s.sky.nonear ? 5000 : 14000); }
}
// a storm: now and then the sky behind the far layers flashes twice, and the flash lights her too
function storm(s, now) {
  const st = s.flash ||= { next: now + 3000 + Math.random() * 8000 };
  const t = now - st.next, flash = t < 0 ? 0 : t < 70 ? .55 : t < 160 ? 0 : t < 260 ? .35 * (1 - (t - 160) / 100) : -1;
  if (flash < 0) { st.next = now + 6000 + Math.random() * 12000; return; }
  const g = s.back.getContext('2d');
  if (flash) { g.globalAlpha = flash; g.fillStyle = '#e8e6ff'; g.fillRect(0, 0, s.back.width, s.back.height * .75); g.globalAlpha = 1; }
  const her = s.host.querySelector('.ga-live, .ga-still');
  if (her) her.style.filter = flash > 0 ? `brightness(${1 + flash * .7}) saturate(${1 - flash * .4})` : '';
}
function air(s, t, dt) {
  const cv = s.front, kinds = [s.light?.motes, WEATHER[s.weather]].filter(Boolean);
  const W = cv.width, H = cv.height, front = cv.getContext('2d'), back = s.back.getContext('2d');
  const spawn = (p, L, any) => Object.assign(p, { L, x: Math.random() * W, y: any ? Math.random() * H : (L.vy > 0 ? -4 : H + 2),
    c: L.col[Math.random() * L.col.length | 0], ph: Math.random() * 7, sp: .6 + Math.random() * .8, z: Math.random() });
  const ps = s.motes ||= kinds.flatMap(L => Array.from({ length: L.n }, () => spawn({}, L, true)));
  front.clearRect(0, 0, W, H);
  for (const p of ps) {
    const g = p.z < .33 ? back : front;  // the farthest third behind her
    const L = p.L, sway = L.sway || (L.wander ? 1.6 : 0);
    p.x += (L.vx * p.sp + sway * Math.sin(t * .7 + p.ph) + gust * (L.streak ? 10 : 16) * p.sp) * dt - lean * .05 * p.z;
    p.y += (L.vy * p.sp + (L.wander ? 1.2 * Math.cos(t * .5 + p.ph * 1.3) : 0)) * dt;
    if (p.y < -6 || p.y > H + 6 || p.x < -6 || p.x > W + 6) spawn(p, L);
    const a = L.twinkle ? Math.max(0, Math.sin(t * (1.1 + p.sp) + p.ph)) ** 2 * .95 : .4 + .45 * p.z;
    if (a < .03) continue;
    const x = Math.round(p.x), y = Math.round(p.y);
    g.fillStyle = p.c;
    g.globalAlpha = a;
    if (L.streak) { g.fillRect(x, y, 1, L.streak + (p.z * 2 | 0)); continue; }
    if (L.petal) { const f = Math.sin(t * 3 + p.ph) > 0; g.fillRect(x, y, f ? 2 : 1, f ? 1 : 2); continue; }
    if (L.bubble) { g.globalAlpha = a * .8; g.fillRect(x, y, 1, 1); if (p.z > .6) { g.globalAlpha = a * .35; g.fillRect(x + 1, y, 1, 1); g.fillRect(x, y - 1, 1, 1); } continue; }
    if (L.big && p.z > 1 - L.big) { g.fillRect(x, y, 2, 2); continue; }
    if (L.wander || L.glow) { g.globalAlpha = a * .22; g.fillRect(x - 1, y - 1, 3, 3); g.globalAlpha = a; }
    g.fillRect(x, y, 1, 1);
  }
  front.globalAlpha = 1; back.globalAlpha = 1;
}

// ---------- styles: injected once, every class prefixed ga- so a product's own CSS stays untouched; the box
// itself (size, corners, background) is :where()d, so any rule of the product's own wins over it ----------
const CSS = `
:where(.ga){position:relative;overflow:hidden;aspect-ratio:var(--ga-ar,832/1216);background:#07060f}
.ga>.ga-live,.ga>.ga-still{position:relative;z-index:1;width:100%;height:100%;display:block;image-rendering:pixelated;object-fit:cover}
.ga>.ga-wait{position:absolute;inset:0;z-index:2;width:100%;height:100%;image-rendering:pixelated;object-fit:cover;transition:opacity .35s ease-out}.ga>.ga-wait.ga-gone{opacity:0}
.ga>img.ga-ov{position:absolute;inset:0;z-index:2;width:100%;height:100%;object-fit:cover;opacity:0;pointer-events:none;image-rendering:pixelated}
.ga>img.ga-ov.on{opacity:1}.ga>img.ga-ov.blush{transition:opacity 1.6s ease-in-out}.ga>img.ga-ov.smirk{transition:opacity .45s ease-in-out}
.ga-scene{position:absolute;inset:0;overflow:hidden;pointer-events:none;background:#07060f}
.ga-scene>*,.ga-front{animation:ga-in .6s ease-out both}
.ga-scene>.ga-far{animation-delay:.12s}.ga-scene>.ga-near{animation-delay:.3s}.ga-front{animation-delay:.4s}
.ga-scene>.ga-near,.ga-scene>.ga-far{animation-name:ga-in-up}
@keyframes ga-in-up{from{opacity:0;translate:0 3%}}@keyframes ga-in{from{opacity:0}}
.ga-scene i{position:absolute;inset:0;background-repeat:no-repeat;image-rendering:pixelated;will-change:background-position,transform}
.ga-sky,.ga-far,.ga-near{background-size:auto 100%;background-position:50% 100%}
.ga-scene .ga-far{inset:-17% 0 17%}.ga-scene .ga-near{inset:-7% 0 7%}
.ga-scene::after{content:'';position:absolute;inset:auto 0 0;height:8%;background:#05040c}
.ga-scene .ga-motif{inset:2% 4% auto;height:58%;background-size:contain;background-position:50% 50%;animation:ga-motif 4.6s ease-in-out infinite}
@keyframes ga-motif{0%,100%{filter:drop-shadow(0 0 6px color-mix(in srgb,var(--glow) 35%,transparent)) brightness(.96)}50%{filter:drop-shadow(0 0 16px color-mix(in srgb,var(--glow) 70%,transparent)) brightness(1.08)}}
.ga-scene.ga-aurora .ga-sky{animation:ga-aurora 11s ease-in-out infinite}
@keyframes ga-aurora{50%{filter:brightness(.95) saturate(1.25) hue-rotate(var(--shim))}}
.ga-scene.ga-lava .ga-near{animation:ga-lava 3.2s ease-in-out infinite}.ga-scene.ga-lava .ga-far{animation:ga-lava 4.7s ease-in-out infinite reverse}
@keyframes ga-lava{50%{filter:brightness(1.28) saturate(1.2)}}
.ga-scene .ga-rays{inset:-20%;background:repeating-conic-gradient(from 180deg at calc(50% - var(--side) * 38%) 0%,rgba(255,244,214,.16) 0 4deg,transparent 4deg 11deg);mix-blend-mode:screen;animation:ga-rays 14s ease-in-out infinite alternate;transform-origin:50% 0}
@keyframes ga-rays{from{opacity:.55;rotate:-2deg}to{opacity:1;rotate:2deg}}
.ga-scene canvas{position:absolute;inset:0;width:100%;height:100%;image-rendering:pixelated}
.ga-front{position:absolute;inset:0;z-index:3;pointer-events:none}
.ga-front canvas{position:absolute;inset:0;width:100%;height:100%;image-rendering:pixelated}
.ga-front::before{content:'';position:absolute;inset:0;background:radial-gradient(120% 90% at 50% 38%,transparent 55%,var(--vig,rgba(4,3,12,.55)))}
.ga-front::after{content:'';position:absolute;inset:auto -30% 0;height:34%;animation:ga-fog 9s ease-in-out infinite,ga-bank 23s ease-in-out infinite alternate;background:radial-gradient(40% 55% at 30% 70%,var(--fog),transparent 70%),radial-gradient(35% 50% at 75% 80%,var(--fog),transparent 70%),linear-gradient(transparent,var(--fog))}
@keyframes ga-bank{to{translate:12% 0}}@keyframes ga-fog{50%{opacity:.75;transform:translateY(3%)}}
@media (prefers-reduced-motion:reduce){.ga-scene *,.ga-front::after{animation:none!important}}`;
function style() {
  if (document.getElementById('glia-avatar-css')) return;
  document.head.append(Object.assign(document.createElement('style'), { id: 'glia-avatar-css', textContent: CSS }));
}

// ---------- worlds ----------
// a scene spec: a theme name, or { theme, skyTheme, weather, motif: { scale, dx }, sky|far|near: { flip, x, hue, alt } }
function sceneSpec(spec) {
  const s = typeof spec === 'string' ? { theme: spec } : { ...spec };
  const layer = l => ({ flip: false, x: 50, hue: 0, alt: '', ...(s[l] || {}) });
  return { skyTheme: s.theme, weather: '', ...s, motif: { scale: 1, dx: 0, ...(s.motif || {}) }, sky: layer('sky'), far: layer('far'), near: layer('near') };
}
function sceneDOM(m, res, s) {
  const worlds = m.scene?.worlds || {}, W = worlds[s.theme], S = worlds[s.skyTheme] || W, L = W.light || {}, SL = S.light || {};
  const src = (w, l, alt) => res(w.layers[l + alt] || w.layers[l]);
  const lay = l => {
    const w = l === 'sky' ? S : W, v = s[l], alt = l === 'sky' && s.skyTheme !== s.theme ? '' : v.alt;
    if (!w.layers[l]) return '';
    return `<i class="ga-${l}" data-x="${v.x}" style="background-image:url('${src(w, l, alt)}');${v.flip ? 'transform:scaleX(-1);' : ''}filter:${{ sky: 'brightness(.82) saturate(.9) ', far: 'brightness(.9) saturate(.82) ', near: 'brightness(.78) ' }[l]}${v.hue ? `hue-rotate(${v.hue}deg)` : ''};"></i>`;
  };
  const emblem = m.scene?.emblem;
  return `<div class="ga-scene${SL.shimmer ? ' ga-aurora' : ''}${L.lava ? ' ga-lava' : ''}" style="--shim:${SL.shimmer || 0}deg">${lay('sky')}<canvas class="ga-back" width="139" height="203"></canvas>${SL.day ? `<i class="ga-rays" style="--side:${s.sky.flip ? -1 : 1}"></i>` : ''}${lay('far')}${L.nonear ? '' : lay('near')}${emblem ? `<i class="ga-motif" style="background-image:url('${res(emblem)}');--glow:${L.glow};scale:${s.motif.scale.toFixed(3)};translate:${s.motif.dx.toFixed(1)}% 0"></i>` : ''}</div><div class="ga-front" style="--fog:${L.fog}${L.vig ? `;--vig:${L.vig}` : ''}"><canvas width="139" height="203"></canvas></div>`;
}
// the light that falls on her: the ground's, its key and fill from the sky above it, mirrored with a mirrored sky
function lookOf(m, s) {
  const worlds = m.scene?.worlds || {};
  const L = worlds[s.theme]?.light || {}, S = worlds[s.skyTheme]?.light || L;
  const look = { key: [225, 200, 255], fill: [120, 150, 255], side: 1, rim: .7, amb: [200, 185, 255, .14], ...L, key: S.key || L.key || [225, 200, 255], fill: S.fill || L.fill || [120, 150, 255], side: S.side || L.side || 1 };
  look.side *= s.sky.flip ? -1 : 1;
  return look;
}

// ---------- moods: icons over her head, the gestures of her head ----------
// one character per art pixel (# outline, o fill, w shine)
export const ICONS = {
  heart: ['#ec5b9b', ['.##.##.', '#ow#oo#', '#ooooo#', '#ooooo#', '.#ooo#.', '..#o#..', '...#...']],
  bang: ['#ffd54a', ['###', '#w#', '#o#', '#o#', '#o#', '###', '...', '###', '#o#', '###']],
  ask: ['#f4f1ff', ['.####.', '#wooo#', '####o#', '..#oo#', '..#o#.', '..###.', '......', '..###.', '..#o#.', '..###.']],
  vein: ['#ff4a4a', ['.##.##.', '#o#.#o#', '##...##', '.......', '##...##', '#o#.#o#', '.##.##.']],
  drop: ['#6cc4ff', ['..#..', '.#o#.', '.#o#.', '#owo#', '#ooo#', '#ooo#', '.###.']],
  note: ['#ffe066', ['..####', '..#oo#', '..#..#', '..#..#', '###.##', '#o#.#o', '###.##']],
  star: ['#ffd84a', ['...#...', '..#w#..', '###o###', '#ooooo#', '.#ooo#.', '#oo#oo#', '##...##']],
};
const bump = u => Math.sin(Math.PI * Math.min(1, Math.max(0, u)));
const hold = (u, a = .18) => ease(Math.min(1, u / a, (1 - u) / a));
// the head's offset (art pixels) and roll (radians) over the gesture's time u in 0..1
export const GESTURES = {
  nod: { len: 1.0, f: u => ({ y: 1.3 * Math.max(0, Math.sin(u * Math.PI * 4)) * (1 - u * .5), r: 0 }) },
  shake: { len: 1.3, f: u => ({ x: .7 * Math.sin(u * Math.PI * 6) * (1 - u) ** 1.2, yaw: 1.6 * Math.sin(u * Math.PI * 6) * (1 - u) ** 1.2, r: .015 * Math.sin(u * Math.PI * 6 + .5) * (1 - u) }) },
  tilt: { len: 3.0, f: u => ({ r: .055 * hold(u), y: .3 * hold(u) }) },
  ponder: { len: 3.4, f: u => ({ r: -.06 * hold(u, .25), y: -.4 * hold(u, .25), x: .3 * hold(u, .25), yaw: .8 * hold(u, .25) }) },
  laugh: { len: 2.0, f: u => ({ y: -1.1 * Math.abs(Math.sin(u * Math.PI * 7)) * (1 - u) ** .7, r: .03 * bump(u) }) },
  jolt: { len: 1.6, f: u => ({ y: u < .12 ? -2 * ease(u / .12) : -2 * (1 - ease((u - .12) / .88)) ** 2, r: 0 }) },
  droop: { len: 3.8, f: u => ({ y: 1.1 * hold(u, .3), r: .04 * hold(u, .3) }) },
  duck: { len: 3.0, f: u => ({ y: .9 * hold(u), r: -.05 * hold(u), x: -.2 * hold(u), yaw: -.8 * hold(u) }) },
  beat: { len: .35, f: u => ({ y: .6 * bump(u), r: 0 }) },  // a word stressed in speech
  hop: { len: 1.4, f: u => ({ y: -1.4 * Math.abs(Math.sin(u * Math.PI * 3)) * (1 - u * .4), r: .02 * Math.sin(u * Math.PI * 3) }) },
  cock: { len: .9, f: u => ({ r: .045 * bump(u), y: 0 }) },  // a question
};
const MOUTHS = ['smile', 'frown', 'open1', 'open2', 'open3'], ARMS = ['wave', 'chin', 'heart'];
const loadImg = src => new Promise((ok, no) => Object.assign(new Image(), { crossOrigin: 'anonymous', onload() { ok(this); }, onerror: () => no(new Error(`glia-avatar: can't load ${src}`)), src }));

// ---------- the still figure: the picture with its overlays, a blink, a smirk, a blush now and then ----------
// for many agents at once (a gallery) or a product that wants no canvas work
function still(host, m, res, cutout, opts) {
  const o = m.idle?.overlays || {};
  host.insertAdjacentHTML('beforeend', `<img class="ga-still" src="${res(cutout && m.scene?.cut ? m.scene.cut : m.art.portrait)}" alt="${opts.alt ?? m.name}">`
    + ['blush', 'smirk', 'half', 'closed'].filter(n => o[n]).map(n => `<img class="ga-ov ${n}" src="${res(o[n])}" alt="">`).join(''));
  const timers = [];
  const later = (f, ms) => timers.push(setTimeout(() => host.isConnected && f(), ms));
  const rand = (a, b) => a + Math.random() * (b - a);
  const show = (n, on) => host.querySelector(`img.ga-ov.${n}`)?.classList.toggle('on', on);
  const blink = again => {
    show('half', true);
    later(() => { show('half', false); show('closed', true); }, 50);
    later(() => { show('closed', false); show('half', true); }, 150);
    later(() => show('half', false), 210);
    later(() => blink(Math.random() < .12), again ? 420 : rand(5000, 9000));
  };
  const smirk = () => { show('smirk', true); later(() => show('smirk', false), rand(2500, 4000)); later(smirk, rand(25000, 45000)); };
  const blush = () => { show('blush', true); later(() => show('blush', false), rand(3000, 5000)); later(blush, rand(35000, 60000)); };
  later(blink, rand(1000, 5000)); later(smirk, rand(6000, 30000)); later(blush, rand(12000, 40000));
  return { stop: () => timers.forEach(clearTimeout) };
}

// ---------- the living figure ----------
// The base with its expression overlays, warped every frame by the rig. Breathing lifts the upper body, hair and
// cloak sway with a lag toward their tips, and a pixel that the figure moves off of shows the clean background
// plate (or, on a scene, the world behind her). Every shift is a whole art pixel, so it stays crisp pixel art.
async function live(host, m, res, cut, look, opts) {
  const idle = m.idle, a0 = m.art.anchors, frames = idle.overlays;
  const load = f => loadImg(res(f));
  const [base, plate, rig, ...ovs] = await Promise.all([m.art.portrait, idle.plate, idle.rig, ...['blush', 'smirk', 'half', 'closed'].map(n => frames[n])].map(load));
  const ov = Object.fromEntries(['blush', 'smirk', 'half', 'closed'].map((n, i) => [n, ovs[i]]));
  for (const n of ['glance', 'glance_l', ...MOUTHS, ...ARMS]) ov[n] = frames[n] ? await load(frames[n]).catch(() => null) : null;
  const W = base.naturalWidth, H = base.naturalHeight, B = m.art.pixel || 6, N = W * H;
  const eyes = a0.eyes, mouth = a0.mouth, chest = a0.chest, waist = a0.waist;
  const pixels = img => { const c = new OffscreenCanvas(W, H).getContext('2d', { willReadFrequently: true }); c.drawImage(img, 0, 0); return c.getImageData(0, 0, W, H).data; };
  const plate32 = new Uint32Array(pixels(plate).buffer), r = pixels(rig);
  const fig = new Uint8Array(N), hairW = new Float32Array(N), cloakW = new Float32Array(N), breathW = new Float32Array(N);
  const zone = [];  // the pixels that can change: the figure and a margin around it for what it moves into
  for (let i = 0; i < N; i++) {
    fig[i] = r[i * 4] > 128;
    hairW[i] = r[i * 4 + 1] / 255;
    cloakW[i] = r[i * 4 + 2] / 255;
    const y = (i / W) | 0, chin = mouth[3];
    // the chest and shoulders rise; the head barely does (riding along as one block read as the picture pushed)
    breathW[i] = ease(Math.min(1, Math.max(0, (waist - y) / (waist - chest))))
      * (y > chest ? 1 : y < chin ? .25 : .25 + .75 * ease((y - chin) / (chest - chin)));
  }
  // her rim: on a scene, what a moving head uncovers there is the world behind her, not her own unmoved pixel
  // (that drew a second hair outline on a head shake). Inside her the unmoved pixel stays, it hides the seams.
  const rim = new Uint8Array(N);
  for (let y = 0; y < H; y++) {
    let last = -1e9;
    for (let x = 0; x < W; x++) { const i = y * W + x; if (!fig[i]) last = x; else if (x - last <= 2 * B) rim[i] = 1; }
    last = 1e9;
    for (let x = W - 1; x >= 0; x--) { const i = y * W + x; if (!fig[i]) last = x; else if (last - x <= 2 * B) rim[i] = 1; }
  }
  // the margin around the figure, with the sway of the nearest figure pixel in its row
  const pad = B * 3, inZone = new Uint8Array(N);
  for (let y = 0; y < H; y++) {
    for (const dir of [1, -1]) {
      let near = -1e9, h = 0, c = 0;
      for (let k = 0; k < W; k++) {
        const x = dir > 0 ? k : W - 1 - k, i = y * W + x;
        if (fig[i]) { near = k; h = hairW[i]; c = cloakW[i]; inZone[i] = 1; }
        else if (k - near <= pad) { inZone[i] = 1; hairW[i] = Math.max(hairW[i], h); cloakW[i] = Math.max(cloakW[i], c); }
      }
    }
    for (let x = 0; x < W; x++) for (let k = 1; k <= 4 * B && y + k < H; k++) if (fig[(y + k) * W + x]) inZone[y * W + x] = 1;
  }
  const arms = {};
  for (const name of ARMS) {
    if (!ov[name]) continue;
    const arm = arms[name] = { idx: [], rgb: [], on: [], thr: [], q: 0, to: 0 };
    const a = pixels(ov[name]);
    let top = H, bot = 0;
    for (let i = 0; i < N; i++) if (a[i * 4 + 3] === 255) { const y = (i / W) | 0; top = Math.min(top, y); bot = Math.max(bot, y); }
    for (let i = 0; i < N; i++) {
      const al = a[i * 4 + 3];
      if (al !== 255 && al !== 128) continue;
      const y = (i / W) | 0, x = i - y * W;
      arm.idx.push(i); arm.rgb.push(0xff000000 | a[i * 4 + 2] << 16 | a[i * 4 + 1] << 8 | a[i * 4]); arm.on.push(al === 255);
      arm.thr.push(al === 255 ? .97 * (bot - y) / Math.max(1, bot - top) : .3);  // a clean wipe up from the elbow
      for (let dy = -2 * B; dy <= 2 * B; dy += B) for (let dx = -2 * B; dx <= 2 * B; dx += B) {
        const yy = y + dy, xx = x + dx;
        if (yy >= 0 && yy < H && xx >= 0 && xx < W) inZone[yy * W + xx] = 1;
      }
    }
    arm.top = top; arm.bot = bot;
  }
  const arm = arms.wave || { idx: [], on: [] };  // the waving one: its hand swings
  for (let i = 0; i < N; i++) if (inZone[i]) zone.push(i);
  const figBase = fig.slice(), posed = new Uint8Array(N), armLit = new Float32Array(N);
  // the motion is worked out per art pixel (a B×B block) and every screen pixel in it moves alike
  const BW = Math.ceil(W / B), BH = Math.ceil(H / B), NB = BW * BH;
  const bHair = new Float32Array(NB), bCloak = new Float32Array(NB), bBreath = new Float32Array(NB), bRow = new Float32Array(NB);
  const blockOf = new Int32Array(N);
  for (let i = 0; i < N; i++) {
    const y = (i / W) | 0, b = ((y / B) | 0) * BW + (((i - y * W) / B) | 0);
    blockOf[i] = b;
    bHair[b] += hairW[i]; bCloak[b] += cloakW[i]; bBreath[b] += breathW[i];
  }
  for (let b = 0; b < NB; b++) { bHair[b] /= B * B; bCloak[b] /= B * B; bBreath[b] /= B * B; bRow[b] = ((b / BW) | 0) * B + B / 2; }
  const bdx = new Int16Array(NB), bdy = new Int16Array(NB);
  const ew = eyes[2] - eyes[0], hx = (eyes[0] + eyes[2]) / 2, chin = mouth[3];
  const neck = chin + ew * .32, piv = [hx, chin + ew * .5];
  const bHead = new Float32Array(NB), bHang = new Float32Array(NB), bX = new Float32Array(NB), bArm = new Float32Array(NB), bFace = new Float32Array(NB), bBrow = new Float32Array(NB);
  for (let b = 0; b < NB; b++) {
    const x = (b % BW) * B + B / 2, y = bRow[b], top = eyes[1];
    const across = 1 - Math.min(1, Math.abs(Math.abs(x - hx) - ew * .27) / (ew * .26));  // over each eye
    const up = y < top && y > top - ew * .42 ? 1 - Math.abs(y - (top - ew * .17)) / (ew * .25) : 0;
    bBrow[b] = Math.max(0, across) * Math.max(0, Math.min(1, up * 1.6));
  }
  {
    const fy = (eyes[1] + chin) / 2, rx = ew * .78, ry = (chin - eyes[1]) * .9 + ew * .35;
    for (let b = 0; b < NB; b++) {
      const d = (((b % BW) * B + B / 2 - hx) / rx) ** 2 + ((bRow[b] - fy) / ry) ** 2;
      bFace[b] = d < 1 ? (1 - d) ** .6 : 0;
    }
  }
  for (let k = 0; k < arm.idx.length; k++) if (arm.on[k]) {
    const i = arm.idx[k], y = (i / W) | 0;
    bArm[blockOf[i]] = Math.max(0, Math.min(1, (arm.top + (arm.bot - arm.top) * .55 - y) / ((arm.bot - arm.top) * .55))) ** 1.4;
  }
  for (let b = 0; b < NB; b++) {
    const x = (b % BW) * B + B / 2, y = bRow[b];
    bX[b] = x;
    const across = Math.abs(x - hx) < ew * 1.3 ? 1 : Math.max(0, 1 - (Math.abs(x - hx) - ew * 1.3) / (ew * .4));
    bHead[b] = across * (y < chin ? 1 : Math.max(0, 1 - (y - chin) / (neck - chin)));
    bHang[b] = bHair[b] > .02 || hairW[Math.min(N - 1, (y | 0) * W + (x | 0))] > .02 ? Math.max(0, 1 - Math.max(0, y - neck) / (ew * 3.2)) : 0;
  }
  // on a scene its light falls on her: a rim of the key light along her edges on its side (the fill on the other),
  // whole art pixels deep, and her colours graded toward the scene's ambient
  const lit = new Float32Array(N);
  look = { ...look };
  {
    const hue = ([r, g, b]) => { const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn || 1;
      return (mx === r ? (g - b) / d + 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4) * 60 % 360; };
    const px = pixels(base);
    let r = 0, g = 0, b = 0, n = 0;
    for (let i = 0; i < N; i += 7) if (fig[i]) { r += px[i * 4]; g += px[i * 4 + 1]; b += px[i * 4 + 2]; n++; }
    const d = Math.abs(hue([r / n, g / n, b / n]) - hue(look.key));
    if (n && Math.min(d, 360 - d) < 35) [look.key, look.fill] = [look.fill, look.key];  // red on red: the cool light wins
  }
  if (cut) {
    const bFig = new Uint8Array(NB), cnt = new Uint16Array(NB);
    for (let i = 0; i < N; i++) if (fig[i]) cnt[blockOf[i]]++;
    for (let b = 0; b < NB; b++) bFig[b] = cnt[b] > B * B / 2;
    const depth = [0, 1, .38, .12], rimB = new Float32Array(NB);
    for (let b = 0; b < NB; b++) {
      if (!bFig[b]) continue;
      const bx = b % BW;
      let w = 0;
      for (let d = 1; d <= 3; d++) {
        if (bx - d * look.side >= 0 && bx - d * look.side < BW && !bFig[b - d * look.side]) { w = Math.max(w, depth[d]); }
        else if (bx + d * look.side >= 0 && bx + d * look.side < BW && !bFig[b + d * look.side]) { w = -depth[d] * .7; }
        else if (b - d * BW >= 0 && !bFig[b - d * BW]) w = Math.max(w, depth[d] * .8);
        if (w) break;
      }
      rimB[b] = w;
    }
    for (let i = 0; i < N; i++) lit[i] = rimB[blockOf[i]] * look.rim;
    for (const a of Object.values(arms)) {  // a raised arm gets its own rim, against her posed outline
      const on = new Uint8Array(N);
      for (let i = 0; i < N; i++) on[i] = figBase[i];
      a.idx.forEach((i, k) => { on[i] = a.on[k] ? 1 : 0; });
      a.lit = a.idx.map((i, k) => {
        if (!a.on[k]) return 0;
        const y = (i / W) | 0, x = i - y * W, open = (dx, dy) => { const xx = x + dx, yy = y + dy; return xx >= 0 && xx < W && yy >= 0 && yy < H && !on[yy * W + xx]; };
        for (let d = 1; d <= 3; d++) {
          if (open(-d * B * look.side, 0)) return depth[d] * look.rim;
          if (open(d * B * look.side, 0)) return -depth[d] * .7 * look.rim;
          if (open(0, -d * B)) return depth[d] * .8 * look.rim;
        }
        return 0;
      });
    }
  }
  const faceRect = [Math.max(0, Math.floor((hx - ew * 1.3) / B) * B), Math.max(0, Math.floor((eyes[1] - ew * .7) / B) * B),
    Math.min(W, Math.ceil((hx + ew * 1.3) / B) * B), Math.min(H, Math.ceil((chin + 6 * B) / B) * B)];
  for (const a of Object.values(arms)) {
    let x0 = W, y0 = H, x1 = 0, y1 = 0;
    for (const i of a.idx) { const y = (i / W) | 0, x = i - y * W; x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x + 1); y1 = Math.max(y1, y + 1); }
    a.box = [x0, y0, x1, y1]; a.was = 0;
    a.blocks = [...new Set(a.idx.filter((i, k) => a.on[k]).map(i => blockOf[i]))];
  }
  const canvas = Object.assign(document.createElement('canvas'), { width: W, height: H, className: 'ga-live' });
  if (opts.alt !== '') { canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', opts.alt ?? m.name); }
  host.querySelector('.ga-scene') ? host.querySelector('.ga-scene').after(canvas) : host.prepend(canvas);
  const ctx = canvas.getContext('2d');
  const out = ctx.createImageData(W, H), out32 = new Uint32Array(out.data.buffer);
  const comp = new OffscreenCanvas(W, H).getContext('2d', { willReadFrequently: true });
  let src32, dirty = true, mouthNow = null, eyesNow = null, mood = {}, gaze = 0, lookSide = null, frameNo = 0, box = host.getBoundingClientRect();
  const acts = [], icons = [];
  const state = { blush: 0, smirk: 0, glance: 0, glance_l: 0, half: 0, closed: 0 };
  if (opts.poke !== false) host.addEventListener('click', () => {  // a poke: a smirk and a blush
    const now = performance.now();
    timeline.push([now, 'smirk', 1], [now + 1800, 'smirk', 0], [now, 'blush', 1], [now + 2600, 'blush', 0]);
    nextSmirk = Math.max(nextSmirk, now + 15000); nextBlush = Math.max(nextBlush, now + 20000);
  });
  function compose() {  // the base with the expression overlays that are on right now
    comp.clearRect(0, 0, W, H);
    comp.drawImage(base, 0, 0);
    const draw = (n, a) => { if (a && ov[n]) { comp.globalAlpha = Math.min(1, a); comp.drawImage(ov[n], 0, 0); comp.globalAlpha = 1; } };
    draw('blush', Math.max(state.blush, mood.blush || 0));
    if (mouthNow && mouthNow !== 'smirk' && ov[mouthNow]) draw(mouthNow, 1); else draw('smirk', mouthNow === 'smirk' ? 1 : state.smirk);
    for (const n of ['glance', 'glance_l']) draw(n, Math.max(state[n], eyesNow === n || (!eyesNow && lookSide === n) ? 1 : 0));
    if (eyesNow === 'half' || eyesNow === 'closed') draw(eyesNow, 1);
    if (eyesNow === 'wink' && ov.closed) { comp.save(); comp.beginPath(); comp.rect(hx, 0, W - hx, H); comp.clip(); draw('closed', 1); comp.restore(); }
    draw('half', state.half); draw('closed', state.closed);
    if (mood.gloom) {  // every other art-pixel column, from the hairline down to the eyes, deepening downward
      const x0 = Math.round((hx - ew * .55) / B) * B, x1 = hx + ew * .55, y0 = eyes[1] - ew * .3, y1 = eyes[1] + ew * .14;
      for (let x = x0; x < x1; x += 2 * B) for (let y = Math.round(y0 / B) * B; y < y1; y += B) {
        comp.globalAlpha = .2 + .55 * (y - y0) / (y1 - y0);
        comp.fillStyle = '#28306a'; comp.fillRect(x, y, B, B);
      }
      comp.globalAlpha = 1;
    }
    // only what can change is read back and relit: the face, and an arm while it moves
    let rx0 = 0, ry0 = 0, rx1 = W, ry1 = H;
    if (src32) {
      [rx0, ry0, rx1, ry1] = faceRect;
      for (const a of Object.values(arms)) if (a.q || a.was) { rx0 = Math.min(rx0, a.box[0]); ry0 = Math.min(ry0, a.box[1]); rx1 = Math.max(rx1, a.box[2]); ry1 = Math.max(ry1, a.box[3]); }
      const part = new Uint32Array(comp.getImageData(rx0, ry0, rx1 - rx0, ry1 - ry0).data.buffer), w = rx1 - rx0;
      for (let y = ry0; y < ry1; y++) src32.set(part.subarray((y - ry0) * w, (y - ry0 + 1) * w), y * W + rx0);
    } else src32 = new Uint32Array(comp.getImageData(0, 0, W, H).data.buffer);
    for (const a of Object.values(arms)) a.was = a.q;
    for (const a of Object.values(arms)) for (let k = 0; k < a.idx.length; k++) if (a.thr[k] >= a.q) { fig[a.idx[k]] = figBase[a.idx[k]]; posed[a.idx[k]] = 0; }
    for (const a of Object.values(arms)) if (a.q) for (let k = 0; k < a.idx.length; k++) {  // the posed arm over her
      const i = a.idx[k];
      if (a.thr[k] < a.q) { fig[i] = a.on[k] ? 1 : 0; posed[i] = a.on[k] ? 2 : 1; armLit[i] = a.lit?.[k] || 0; if (a.on[k] || !cut) src32[i] = a.rgb[k]; }
    }
    if (cut) for (let y = ry0; y < ry1; y++) for (let i = y * W + rx0, e = y * W + rx1; i < e; i++) {  // on a scene only she is drawn, lit by it
      if (!fig[i]) { src32[i] = 0; continue; }
      const s = posed[i] ? armLit[i] : lit[i];
      if (!s && !look.amb[3]) continue;
      const v = src32[i], t = look.amb[3];
      let r = v & 255, g = v >> 8 & 255, bl = v >> 16 & 255;
      r += (look.amb[0] * r / 255 - r) * t; g += (look.amb[1] * g / 255 - g) * t; bl += (look.amb[2] * bl / 255 - bl) * t;
      if (s) { const c = s > 0 ? look.key : look.fill, a = Math.abs(s); r += (c[0] - r) * a; g += (c[1] - g) * a; bl += (c[2] - bl) * a; }
      src32[i] = (v & 0xff000000) | (bl << 16) | (g << 8) | r;
    }
    out32.set(src32);
  }
  // smirk and blush fade (seconds for a full fade); the blink cuts, as a real one does
  const target = { blush: 0, smirk: 0, glance: 0, glance_l: 0, half: 0, closed: 0 }, fade = { blush: 1.6, smirk: .45 };
  const set = (n, v) => {
    if (n === 'mouth') { if (mouthNow !== v) { mouthNow = v; dirty = true; } return; }
    if (n === 'eyes') { if (eyesNow !== v) { eyesNow = v; dirty = true; } return; }
    if (n === 'gesture') {
      const now = performance.now();
      acts.push([now, v, GESTURES[v]?.len || 1, ['tilt', 'cock', 'droop'].includes(v) && Math.random() < .5 ? -1 : 1]);  // a tilt goes either way
      if (['shake', 'tilt', 'ponder', 'duck'].includes(v) && eyesNow !== 'closed' && Math.random() < .7)
        [[0, 'half', 1], [45, 'half', 0], [45, 'closed', 1], [130, 'closed', 0], [130, 'half', 1], [190, 'half', 0]].forEach(([d, k, x]) => timeline.push([now + d, k, x]));
      return;
    }
    if (n === 'mood') { mood = v; browTo = v.brow || 0; dirty = true; return; }
    if (n === 'arm') { for (const [k, a] of Object.entries(arms)) a.to = k === v ? 1 : 0; return; }
    target[n] = v; if (!fade[n]) { state[n] = v; dirty = true; }
  };
  // the hair on the head's spring: it chases the head, overshoots, swings back
  const lag = { x: 0, y: 0, r: 0 }, vel = { x: 0, y: 0, r: 0 };
  const moods = m.moods || {};
  function emote(name) {
    const e = moods[name];
    if (!e) return false;
    const now = performance.now();
    for (let k = timeline.length - 1; k >= 0; k--) if (['mouth', 'eyes', 'mood'].includes(timeline[k][1])) timeline.splice(k, 1);
    if (e.arm && arms[e.arm]) timeline.push([now, 'arm', e.arm], [now + e.hold - 300, 'arm', 0]);
    timeline.push([now, 'mouth', e.mouth || null], [now, 'eyes', e.eyes || null], [now, 'mood', { blush: e.blush || 0, gloom: e.gloom || 0, brow: e.brow || 0 }],
      [now + e.hold, 'mouth', null], [now + e.hold, 'eyes', null], [now + e.hold + 400, 'mood', {}]);
    if (e.chuckle) for (let k = 1; k < 9; k++) timeline.push([now + k * 230, 'mouth', k % 2 ? 'open2' : 'open3']);
    if (e.gesture) timeline.push([now, 'gesture', e.gesture]);
    if (e.icon && ICONS[e.icon]) { icons.length = 0; icons.push({ k: e.icon, t0: now + 120, life: Math.min(2600, e.hold) }); }
    if (e.eyes || e.brow > 0) nextBlink = Math.max(nextBlink, now + e.hold * (e.eyes === 'closed' || e.brow > 0 ? 1 : .4));  // wide eyes hold open
    nextGlance = Math.max(nextGlance, now + e.hold + 4000);
    return e.hold;
  }
  // speech, as game dialogue animates it: one open mouth per syllable, a half-shut mouth between, shut between the
  // words, rests at the stops; a cheerful mood rests on a smile. A stressed word nods, a question cocks the head
  function say(text, after, pace = 1) {
    const now = performance.now();
    for (let k = timeline.length - 1; k >= 0; k--) if (timeline[k][1] === 'mouth') timeline.splice(k, 1);
    const rest = ['happy', 'laugh', 'wave', 'yes', 'shy'].includes(after) && ov.smile ? 'smile' : null;
    if (breathPhase > .45) breathPhase = 0;  // a breath in before the first word
    let t = now + 120, n = 0;
    for (const tok of String(text).slice(0, 360).match(/[\p{L}']+|[.!?]+|[,;:]|\s+/gu) || []) {
      if (/^\p{L}/u.test(tok)) {
        const syl = Math.max(1, (tok.toLowerCase().match(/[aeiouyäöü]+/g) || []).length);
        const stress = tok.length > 6 || /^[A-ZÄÖÜ]/.test(tok);
        for (let k = 0; k < syl; k++) {
          const big = stress && k === 0 ? 'open3' : Math.random() < .55 ? 'open2' : 'open1';
          timeline.push([t, 'mouth', big], [t, 'syllable', big === 'open3']); t += (105 + Math.random() * 35) * pace;
          if (k < syl - 1) { timeline.push([t, 'mouth', 'open1']); t += 55 * pace; }
        }
        timeline.push([t, 'mouth', rest]); t += 70 * pace;
        if (stress && ++n % 3 === 0) timeline.push([t - 200, 'gesture', 'beat']);
      } else if (/[,;:]/.test(tok)) t += 200 * pace;
      else if (/[.!?]/.test(tok)) {
        if (tok.includes('?')) timeline.push([t, 'gesture', 'cock']);
        if (tok.includes('!')) timeline.push([t - 150, 'gesture', 'beat']);
        t += 360 * pace;
      }
    }
    timeline.push([t, 'mouth', null]);
    if (after) timeline.push([t + 150, 'emote', after]);
    speakTill = t;
    return t - now;
  }
  let listenTill = 0, speakTill = 0, brow = 0, browTo = 0;
  function listen() {  // one attentive tilt and a small smile per burst of typing
    const now = performance.now();
    if (now > listenTill + 1500 && !acts.length) timeline.push([now, 'gesture', 'tilt']);
    if (!mouthNow || mouthNow === 'smile') { set('mouth', 'smile'); for (let k = timeline.length - 1; k >= 0; k--) if (timeline[k][1] === 'mouth' && timeline[k][2] === null && timeline[k].listen) timeline.splice(k, 1); const e = [now + 2200, 'mouth', null]; e.listen = 1; timeline.push(e); }
    listenTill = now;
  }
  let thinkOn = false, nextThink = 0;
  function think(on) {
    on = !!on;
    if (on === thinkOn) return;
    thinkOn = on;
    const now = performance.now();
    if (on) nextThink = now;
    else {  // back to rest: the hand down, the face plain
      for (let k = timeline.length - 1; k >= 0; k--) if (['mouth', 'eyes', 'mood', 'arm'].includes(timeline[k][1])) timeline.splice(k, 1);
      timeline.push([now, 'arm', 0], [now, 'mouth', null], [now, 'eyes', null], [now, 'mood', {}]);
      icons.length = 0;
    }
  }
  function hush() {
    const now = performance.now();
    for (let k = timeline.length - 1; k >= 0; k--) if (['mouth', 'syllable', 'gesture'].includes(timeline[k][1])) timeline.splice(k, 1);
    set('mouth', null);
    speakTill = now;
  }
  // sparkles: plus-shaped stars glowing up and fading in the background around her
  const sparks = Array.from({ length: 6 }, () => ({}));
  function place(s, fresh) {
    for (let k = 0; k < 50; k++) {
      const x = (Math.random() * (W / B - 4) + 2 | 0) * B, y = (Math.random() * (chest / B) | 0) * B;
      if (!fig[y * W + x] && !fig[(y + B) * W + x]) {
        Object.assign(s, { x, y, life: 2.5 + Math.random() * 2.5, t: fresh ? -Math.random() * 6 : -Math.random() * 3 });
        return;
      }
    }
  }
  sparks.forEach(s => place(s, true));
  const rand = (a, b) => a + Math.random() * (b - a);
  const t0 = performance.now();
  let nextBlink = t0 + rand(1500, 3500), nextSmirk = t0 + rand(12000, 20000), nextBlush = t0 + rand(20000, 30000),
    nextGlance = t0 + rand(6000, 10000), nextMoment = t0 + rand(14000, 24000);
  const timeline = [];  // [when, what, value]
  let last = 0, stopped = false;
  const BREATH = 5.2;  // a breath: in quickly, out slower, a short rest; its clock drifts so it never feels metronomic
  let breathPhase = 0, sigh = false, swing = 0;
  const breathCurve = p => p < .38 ? ease(p / .38) : p < .86 ? 1 - ease((p - .38) / .48) : 0;
  const hairRow = new Float32Array(H), cloakRow = new Float32Array(H);
  function frame(now) {
    if (stopped) return;
    if (!host.isConnected) { if (opts.keep) requestAnimationFrame(frame); return; }
    requestAnimationFrame(frame);
    if (now > nextBlink) {
      const slow = Math.random() < .15 ? 3 : 1;  // now and then a slow, fond blink
      const b = [[0, 'half', 1], [70 * slow, 'half', 0], [70 * slow, 'closed', 1], [70 * slow + 100 * slow ** 1.6, 'closed', 0], [70 * slow + 100 * slow ** 1.6, 'half', 1], [140 * slow + 100 * slow ** 1.6, 'half', 0]];
      b.forEach(([d, n, v]) => timeline.push([now + d, n, v]));
      nextBlink = now + (Math.random() < .12 ? 420 : rand(5000, 9000));  // now and then a double blink
    }
    if (ov.glance && now > nextGlance) {  // a glance aside: there and back, often with a blink on the return
      const hold = rand(1100, 2000);
      const way = ov.glance_l && Math.random() < .5 ? 'glance_l' : 'glance';
      timeline.push([now, way, 1], [now + hold, way, 0]);
      if (Math.random() < .3) timeline.push([now + 260, way, 0], [now + 520, way, 1]);  // a double take
      if (Math.random() < .5) [[0, 'half', 1], [50, 'half', 0], [50, 'closed', 1], [150, 'closed', 0], [150, 'half', 1], [210, 'half', 0]].forEach(([d, n, v]) => timeline.push([now + hold - 60 + d, n, v]));
      nextGlance = now + hold + rand(15000, 30000);
      nextBlink = Math.max(nextBlink, now + hold + 1500);
    }
    if (now > nextMoment && !acts.length && !mouthNow) {
      const r = Math.random();
      if (r < .4) timeline.push([now, 'mouth', 'smile'], [now, 'gesture', 'tilt'], [now + 2600, 'mouth', null]);
      else if (r < .7) timeline.push([now, 'mouth', 'smile'], [now, 'gesture', 'beat'], [now + 520, 'gesture', 'beat'], [now + 1040, 'gesture', 'beat'], [now + 1900, 'mouth', null]);  // a little hum
      else timeline.push([now, 'gesture', 'ponder']);
      nextMoment = now + rand(22000, 45000);
    }
    if (thinkOn && now > nextThink && now > speakTill) { emote('thinking'); nextThink = now + 4200; }
    if (now > nextSmirk) { const hold = rand(2500, 4000); timeline.push([now, 'smirk', 1], [now + hold, 'smirk', 0]); nextSmirk = now + hold + rand(25000, 45000); }
    if (now > nextBlush) { const hold = rand(3000, 5000); timeline.push([now, 'blush', 1], [now + hold, 'blush', 0]); nextBlush = now + hold + rand(35000, 60000); }
    for (let k = 0; k < timeline.length; k++) if (timeline[k][0] <= now) {
      const [, n, v] = timeline.splice(k--, 1)[0];
      if (n === 'emote') emote(v); else if (n === 'syllable') opts.onSyllable?.(v); else set(n, v);
    }
    const t = now / 1000, dt = last ? Math.min(.1, t - last) : 0; last = t;
    for (const n in fade) if (state[n] !== target[n]) {
      const d = dt / fade[n];
      state[n] = target[n] > state[n] ? Math.min(target[n], state[n] + d) : Math.max(target[n], state[n] - d);
      dirty = true;
    }
    for (const a of Object.values(arms)) if (a.q !== a.to) {  // an arm rises in about a quarter second, lowers slower
      a.q = a.to > a.q ? Math.min(1, a.q + dt / .26) : Math.max(0, a.q - dt / .34);
      dirty = true;
    }
    if (dirty) {  // the fade's curve, not its linear clock
      const lin = { ...state };
      for (const n in fade) state[n] = ease(lin[n]);
      compose();
      Object.assign(state, lin);
    }
    dirty = false;
    const was = breathPhase;
    breathPhase = (breathPhase + dt / (BREATH * (1 + .08 * Math.sin(t * .37)) * (sigh ? 1.5 : 1))) % 1;
    if (breathPhase < was) sigh = Math.random() < .12;  // a new breath: now and then a sigh
    const breath = calm ? 0 : breathCurve(breathPhase) * (sigh ? 1.6 : 1);
    // the sway: a slow drift of air, never one clean sine; the tips follow the root later, as cloth and hair do
    swing = swing * Math.exp(-dt * 1.4) + Math.abs(leanTo - lean) * dt * 9;
    const airy = calm ? 0 : (.55 + .25 * Math.sin(t * .41) + .2 * Math.sin(t * .93 + 1.7)) * (1 + 1.1 * gust + Math.min(1.2, swing));
    for (let y = 0; y < H; y++) {
      hairRow[y] = airy * (3.2 * Math.sin(t * 1.63 - y * .009) + 1.3 * Math.sin(t * 2.71 - y * .017 + .6));
      cloakRow[y] = airy * (4.5 * Math.sin(t * 1.21 - y * .005 + .8) + 1.6 * Math.sin(t * 2.07 - y * .011 + 2.1));
    }
    const breathLate = breathCurve((breathPhase + .94) % 1);  // the head rides a beat behind the shoulders
    const pose = calm ? { x: 0, y: 0, r: 0 } : { x: 2.2 * Math.sin(t * .31) + 1.3 * Math.sin(t * .73 + 2), y: 0, r: .012 * Math.sin(t * .23 + 1) + .008 * Math.sin(t * .57) };
    // the pointer near her and moving: her eyes go to it and her head turns a little after them
    if (!calm && opts.pointer !== false && now - ptr.t < 2500 && now > speakTill) {
      if (!(frameNo++ % 12)) box = host.getBoundingClientRect();
      const rel = (ptr.x - box.left - box.width / 2) / box.width, up = (ptr.y - box.top) / box.height;
      const near = Math.abs(rel) < 1.6 && up > -.3 && up < 1.3;
      gaze += ((near ? Math.max(-1, Math.min(1, rel * 1.4)) : 0) - gaze) * Math.min(1, dt * 3);
    } else gaze += (0 - gaze) * Math.min(1, dt * 1.5);
    const side = gaze > .45 ? 'glance' : gaze < -.45 ? 'glance_l' : null;
    if (side !== lookSide) { lookSide = side; dirty = true; }
    pose.x += gaze * B * .3; pose.yaw = gaze * B * .7; pose.r += gaze * .012;
    if (now < speakTill && !calm) {  // a speaker's head is never still
      const k = Math.min(1, (speakTill - now) / 400, (now - speakTill + 99999) / 400);
      pose.y += k * B * .35 * Math.abs(Math.sin(t * 5.3)); pose.r += k * .014 * Math.sin(t * 1.7); pose.yaw += k * B * .4 * Math.sin(t * 1.1);
    }
    let lift = 0;
    for (let k = acts.length - 1; k >= 0; k--) {
      const u = (now - acts[k][0]) / 1000 / acts[k][2];
      if (u >= 1) { acts.splice(k, 1); continue; }
      const g = GESTURES[acts[k][1]].f(Math.max(0, u)), mm = acts[k][3] || 1;
      pose.x += (g.x || 0) * B * mm; pose.y += (g.y || 0) * B; pose.r += (g.r || 0) * mm; pose.yaw += (g.yaw || 0) * B * mm; lift += (g.y || 0) * B;
    }
    for (const k of ['x', 'y', 'r']) {  // underdamped: a quick head leaves the hair swinging
      vel[k] += ((pose[k] - lag[k]) * 70 - vel[k] * 6.5) * dt;
      lag[k] += vel[k] * dt;
    }
    for (let b = 0; b < NB; b++) {
      const y = bRow[b], late = ease(Math.min(1, Math.max(0, (chest - y) / (chest - mouth[3]))));
      const h = bHead[b], hang = bHang[b] * (1 - h);
      const ox = bX[b] - piv[0], oy = y - piv[1];
      const hdx = pose.x - oy * pose.r, hdy = pose.y + ox * pose.r;
      const ldx = lag.x - oy * lag.r, ldy = lag.y + ox * lag.r;
      const tip = h * bHair[b] * .45;  // strands on the head trail it too, at most an art pixel
      bdy[b] = -Math.round((breath + (breathLate - breath) * late) * bBreath[b] * 3) + Math.round(h * hdy + hang * ldy + tip * Math.max(-B, Math.min(B, ldy - hdy)) + (1 - h) * bBreath[b] * lift * .4);
      bdx[b] = Math.round(bHair[b] * hairRow[y] + bCloak[b] * cloakRow[y] + h * hdx + hang * ldx + tip * Math.max(-B, Math.min(B, ldx - hdx)) + bFace[b] * pose.yaw
        + (arm.q === 1 && bArm[b] ? bArm[b] * B * 1.3 * Math.sin(t * 8.5) : 0));
    }
    brow += (browTo - brow) * Math.min(1, dt * 14);  // brows jump fast and settle
    if (Math.abs(brow) > .02) for (let b = 0; b < NB; b++) if (bBrow[b]) bdy[b] -= Math.round(bBrow[b] * brow * B);
    for (const a of Object.values(arms)) if (a.q > 0 && a.q < 1) {
      const off = Math.round((1 - ease(a.q)) * 3 * B);
      for (const b of a.blocks) bdy[b] += off;
    }
    for (let k = 0; k < zone.length; k++) {
      const i = zone[k], y = (i / W) | 0, x = i - y * W, b = blockOf[i];
      const sx = x - bdx[b], sy = y - bdy[b], q = sy * W + sx;
      // a raised arm is in front: hair swaying onto it slides behind it; only the waving hand moves over it
      if (posed[i] === 2 && !(arm.q && bArm[b]) && !(sx >= 0 && sx < W && sy >= 0 && sy < H && posed[q] === 2)) out32[i] = src32[i];
      else if (sx >= 0 && sx < W && sy >= 0 && sy < H && fig[q]) out32[i] = src32[q];
      else out32[i] = fig[i] && !cut ? plate32[i] : cut && rim[i] ? 0 : src32[i];
    }
    ctx.putImageData(out, 0, 0);
    ctx.fillStyle = cut ? `rgb(${look.key.map(c => c + (255 - c) * .55 | 0)})` : '#e9e2ff';  // on a scene, its light
    for (const sp of sparks) {
      sp.t += dt;
      if (sp.t > sp.life) { place(sp); continue; }
      if (sp.t < 0) continue;
      const a = Math.sin(Math.PI * sp.t / sp.life) ** 2;
      ctx.globalAlpha = a; ctx.fillRect(sp.x, sp.y, B, B);
      ctx.globalAlpha = a * .55;
      ctx.fillRect(sp.x - B, sp.y, B, B); ctx.fillRect(sp.x + B, sp.y, B, B); ctx.fillRect(sp.x, sp.y - B, B, B); ctx.fillRect(sp.x, sp.y + B, B, B);
    }
    ctx.globalAlpha = 1;
    for (let k = icons.length - 1; k >= 0; k--) {
      const ic = icons[k], age = (now - ic.t0) / 1000;
      if (age > ic.life / 1000) { icons.splice(k, 1); continue; }
      if (age < 0) continue;
      const [col, rows] = ICONS[ic.k];
      const pop = age < .28 ? Math.sin(age / .28 * Math.PI) : 0;  // it hops up as it appears, then bobs
      const C = 2 * B;  // an icon pixel is two art pixels: at one it read as a speck
      const x0 = Math.min(W - C * rows[0].length - B, Math.round((hx + ew * .85 + pose.x) / B) * B);
      const y0 = Math.max(5 * B, Math.round((eyes[1] - ew * 1.35 + pose.y - pop * B * 2 + Math.sin(age * 5) * B * .5) / B) * B);
      ctx.globalAlpha = Math.min(1, age / .08, (ic.life / 1000 - age) / .3);
      rows.forEach((row, ry) => [...row].forEach((c, rx) => {
        if (c === '.') return;
        ctx.fillStyle = c === '#' ? '#1a1020' : c === 'w' ? '#ffffff' : col;
        ctx.fillRect(x0 + rx * C, y0 + ry * C, C, C);
      }));
      ctx.globalAlpha = 1;
    }
  }
  requestAnimationFrame(frame);
  return { emote, say, listen, think, hush, stop() { stopped = true; } };
}

// ---------- mounting ----------
const asRoot = r => r.endsWith('/') ? r : r + '/';
export const GliaAvatar = {
  VERSION, FORMAT, FORMATS, migrate, WEATHER, ICONS, GESTURES,

  /** Fetch <root>agent.json and mount her into el. */
  async load(el, root, opts = {}) {
    root = asRoot(root);
    const res = await fetch(root + 'agent.json');
    if (!res.ok) throw new Error(`glia-avatar: no agent.json at ${root}`);
    return GliaAvatar.mount(el, await res.json(), { root, ...opts });
  },

  /**
   * Mount a manifest into el. opts:
   *   root      where the manifest's paths are relative to (default './')
   *   resolve   (path) => url, instead of root
   *   scene     a world name, a scene spec, or false for none; default the package's default (her own world)
   *   live      false: the still picture with blinks only (cheap, for galleries); default true
   *   pointer   false: her eyes don't follow the pointer
   *   poke      false: a click on her doesn't make her smirk
   *   keep      true: keep animating while el is moved around the DOM
   *   alt       accessible name, default her name; '' marks her decorative
   *   onSyllable(big)  called per spoken syllable, e.g. for a voice of your own
   */
  async mount(el, manifest, opts = {}) {
    style();
    const m = migrate(manifest);
    const res = opts.resolve || (p => (opts.root ? asRoot(opts.root) : './') + p);
    let figure = null, stillLoop = null, sc = null;
    el.classList.add('ga');
    if (m.art.size) el.style.setProperty('--ga-ar', `${m.art.size[0]}/${m.art.size[1]}`);
    const avatar = {
      manifest: m,
      get moods() { return Object.keys(m.moods || {}); },
      get worlds() { return Object.keys(m.scene?.worlds || {}); },
      mood(name) { return figure?.emote(name) || 0; },
      say(text, opts2 = {}) { return figure?.say(text, opts2.mood, opts2.pace) || 0; },
      think(on = true) { figure?.think(on); },
      listen() { figure?.listen(); },
      hush() { figure?.hush(); },
      async scene(spec) { await draw(spec); },
      /** The frame on screen as a PNG data URL (her figure only; with scene: false the whole picture). */
      snapshot() { return el.querySelector('.ga-live')?.toDataURL('image/png') || null; },
      destroy() { figure?.stop(); stillLoop?.stop(); if (sc) scenes.delete(sc); el.replaceChildren(); el.classList.remove('ga'); },
    };
    async function draw(spec) {
      figure?.stop(); stillLoop?.stop(); if (sc) scenes.delete(sc);
      figure = stillLoop = sc = null;
      el.replaceChildren();
      const worlds = m.scene?.worlds || {};
      if (spec === undefined) spec = m.scene?.default ?? null;
      if (spec && !worlds[typeof spec === 'string' ? spec : spec.theme]) spec = worlds.home ? 'home' : Object.keys(worlds)[0] || null;
      const s = spec && m.scene?.cut ? sceneSpec(spec) : null;
      if (s) {
        el.insertAdjacentHTML('afterbegin', sceneDOM(m, res, s));
        const front = el.querySelector('.ga-front');
        el.append(front);  // in front of her
        const light = worlds[s.theme].light || {}, skyL = worlds[s.skyTheme]?.light || light;
        sc = { el: el.querySelector('.ga-scene'), host: el, back: el.querySelector('.ga-back'), front: front.querySelector('canvas'),
          light, sky: skyL, storm: skyL.storm || s.weather === 'rain', weather: s.weather, keep: opts.keep };
        scenes.add(sc);
        el.style.setProperty('--ga-glow', light.glow || '');
      }
      const canLive = opts.live !== false && m.idle?.rig && m.idle?.plate && typeof OffscreenCanvas !== 'undefined';
      if (canLive) {
        // her picture at once while the live figure is worked out, then faded over to it: no empty box, no pop
        const wait = Object.assign(new Image(), { className: 'ga-wait', alt: '', src: res(s && m.scene?.cut ? m.scene.cut : m.art.portrait) });
        const front = el.querySelector('.ga-front');
        front ? front.before(wait) : el.append(wait);
        try {
          figure = await live(el, m, res, !!s, lookOf(m, s || sceneSpec('none')), opts);
          requestAnimationFrame(() => { wait.classList.add('ga-gone'); setTimeout(() => wait.remove(), 400); });
          return;
        } catch (e) { console.warn(e); wait.remove(); }
      }
      stillLoop = still(el, m, res, !!s, opts);
      el.querySelector('.ga-front') && el.append(el.querySelector('.ga-front'));
    }
    await draw(opts.scene === false ? null : opts.scene);
    return avatar;
  },
};
export default GliaAvatar;
