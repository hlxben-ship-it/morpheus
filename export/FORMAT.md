# glia-agent format

A glia-agent package is one folder: `agent.json` plus images. Every path in the manifest is relative to that
folder. `glia-avatar.js` is the reference player; this page is for rendering her your own way (Unity, native,
a terminal) or checking what a field means.

Units: **image pixels** for positions and sizes (the portrait's pixel grid, origin top left), **ms** for times,
colours as `[r, g, b]` 0–255 or CSS strings. Boxes are `[x0, y0, x1, y1]`.

## agent.json (`glia-agent/2`)

| Field | Meaning |
|---|---|
| `format` | `"glia-agent/2"`. A reader that doesn't know the value must refuse, not guess. |
| `id`, `name`, `title` | Slug, display name, optional epithet (`""` if none). |
| `persona.soul` | Who she is, a short paragraph. |
| `persona.voice` | How she talks. |
| `persona.system_prompt` | Ready to use as an LLM system prompt (soul + voice). |
| `art.style`, `art.pixel` | Pixel art; one art pixel = `pixel` image pixels. Move and scale only in whole art pixels, nearest-neighbour. |
| `art.size` | `[w, h]` of the portrait and every overlay. |
| `art.portrait` | The full picture, opaque. |
| `art.bust` | `{file, box}`: head and shoulders crop, and where it was cut from the portrait. |
| `art.anchors` | `eyes`, `mouth`: boxes; `chest`, `waist`: y lines (may lie below the picture). The player places gestures, icons and the breath around them. |
| `idle.overlays` | Name → full-size transparent PNG laid over the portrait. Eyes: `half`, `closed`, `glance`, `glance_l`. Mouths: `smirk`, `smile`, `frown`, `open1`–`open3` (small to wide, for speech). `blush`. Arm poses: `wave`, `chin`, `heart` (alpha 255 = arm, 128 = the body the arm uncovers). Any may be missing. |
| `idle.rig` | Optional. R = her figure mask (>128), G = hair sway weight, B = cloth sway weight. Needed for the live, moving figure. |
| `idle.plate` | Optional. The background without her, shown where she moves away. |
| `moods` | Name → recipe, see below. |
| `loops` | Repeating sequences (`thinking`): `every` ms, `steps` of `{at, eyes, mouth, gesture, blink}`. |
| `scene.default` | World to show first, normally `"home"` (her own); `null` = no backdrop. |
| `scene.cut` | Her figure cut out (transparent), drawn over a world. |
| `scene.emblem` | Her sigil, drawn behind her in a world. Optional. |
| `scene.worlds` | Name → `{name, light, layers}`. `layers`: `sky`, `far`, `near` back to front (`near` optional), each maybe with a `…2` variant to pick from. |
| `…light` | `key`/`fill` colours and `side` (−1 key from the left, 1 right), `rim` strength, `amb` `[r,g,b,strength]` tint over her, `fog`, `glow` (CSS colour), and optional effects: `motes` (floating particles), `shimmer` (aurora hue), `day`, `storm`, `lava`, `vig`, `nonear`. Ignore what you don't render. |
| `colors` | A dialogue box in her world's colours: `frame`, `fill`, `name`. |
| `models` | Which image models made her. |
| `voice_seed` | A stable number per look, for a voice of your own (pitch, blip). |
| `created_by`, `updated` | Producer and date. |

### Mood recipe

```json
"happy": { "mouth": "smile", "blush": 0.7, "gesture": "tilt", "hold": 3200, "icon": "note" }
```

| Key | Meaning |
|---|---|
| `mouth`, `eyes` | Overlay names (above) held for the mood; `eyes: "wink"` = `closed` on one eye only. |
| `blush` | 0–1 strength of the blush overlay. |
| `arm` | Arm pose overlay, wiped in from the elbow and out before `hold` ends. |
| `gesture` | Head motion: `tilt`, `cock`, `nod`, `shake`, `jolt`, `droop`, `duck`, `hop`, `laugh`, `ponder`. |
| `hold` | How long the mood lasts, ms. Then back to idle. |
| `icon` | A small pixel symbol by her head: `note`, `bang`, `drop`, `vein`, `ask`, `heart`, `star`. |
| `brow` | 1 raised (wide eyes), −1 lowered. |
| `gloom` | 1: shaded stripes over the upper face (sad). |
| `chuckle` | Alternating open mouths during the mood. |

Minimal rendering: draw `portrait`, lay over `mouth`/`eyes`/`blush` overlays for `hold` ms. Everything else is
polish.

## Versions and migrations

A reader keeps a chain of small steps, each turning one format into the next, and runs it until it reaches its
own (`migrate()` in `glia-avatar.js`, `agents.migrate()` in Morpheus). Old packages keep working with a newer
player; an unknown newer format fails with a message.

### `glia-agent/1` → `/2`

- `mood_recipes` became `moods` (one mood table; `/1` also had a list).
- `scene.world_sky/far/near` + `scene.light` moved to `scene.worlds.home = {name, light, layers}`; `scene.default`
  added. `/2` also ships every shared world and `stills/`.
- `title` is now empty instead of a fragment of the soul.
