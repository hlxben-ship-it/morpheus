# Morpheus

AI image studio — generate, edit, face swap — running locally on the AMD Radeon RX 7900 XT (20 GB),
driven from the web UI, the command line or by Glia. v0.2 (formerly Imageer).

## Web UI

```
start_morpheus.ps1        # serves http://127.0.0.1:5196 and opens the browser
```

Laid out like a chat app: past **sessions** on the left (kept in the browser's `localStorage`), the
current session in the middle with every result inline as a clickable thumbnail, and a composer at
the bottom. Top right: **Ghost** mode and the light/dark switch (follows the system until chosen).
The look is 16-bit cyberpunk: Silkscreen and VT323 pixel type for the chrome, Inter for text, neon
cyan and magenta, stepped corners, CRT scanlines (never over pictures); logo and favicon in `web/`.

The header shows ComfyUI's state (idle, thinking while Claude writes the prompt, rendering with its queue,
off), VRAM in use and GPU load. Both come from the Windows GPU performance counters, so they count every
program, like Task Manager; a PowerShell sampler started by the server reads them each second and ends
with it. Hovering VRAM lists the top 3 programs on the card plus everything else, so the rows add up
(per process it reads "Local Usage", what is resident; "Dedicated Usage" counts the desktop's mapped buffers
again). Under ComfyUI, and when hovering its state, come its loaded models with their kind and how much of
each sits on the card. ComfyUI has no route for that, so `comfy_ext/morpheus_status.py` adds
`/morpheus/loaded`; the server copies it into `ComfyUI/custom_nodes` whenever it starts ComfyUI.
The GPU temperature comes from AMD's driver library (`atiadlxx.dll`, ADL PMLog: edge, hotspot, memory, fan),
since Windows has no counter for it; hovering shows all four, and it hides on cards without ADL. Sessions with a job running get a red REC dot in the list.

The server starts ComfyUI and renders a tiny throwaway picture at boot, so the first real prompt
doesn't pay the model load. Every render shows its step count and a live, pixelated preview; a
finished picture arrives with a scanline reveal. Jobs run in the background on the server, so a
reload mid-render picks the result up again.

A new session starts as one of three kinds (Alt+1–3 on the empty session); once the first message
is sent the kind is fixed and shown as a badge:

- **Chat** — describe the picture; Claude (Claude Code CLI, `claude -p --model sonnet`) writes the
  English prompt and picks model, style and format, then the server renders. A follow-up like
  "make it night" **edits** the session's latest picture (FLUX.2 Klein); "now a different scene"
  generates a new one. Each session keeps its own Claude conversation. A picture can ride along
  (paperclip, paste or drop): Claude looks at it and either edits exactly that picture or generates
  a new one in its style. With a reference picture, generate runs a **refine loop**: Claude compares
  each result with the reference and re-prompts until it is happy, at most 5 rounds.
- **Manual** — the exact prompt and settings, no Claude in between.
- **Guise** — the face from a photo onto the largest face in a video section (`faceswap.py`,
  `.venv-faceswap`): High (HyperSwap 256 + GPEN 512), Premium (GPEN 1024), Fast (inswapper 128);
  optional hair recolouring and Real-ESRGAN ×2/×4 upscaling (capped at 4K). A finished Guise ends
  the session: **Run again with changes** or **New Guise**.

Crossing over is manual: **Edit** on any result continues in the current chat (or opens a new chat
session from Manual, Guise or the Gallery); **Variant** loads a result's prompt and settings into a
Manual session. Dropped or pasted pictures go to a chat, videos to a Guise session. Every result also
has **Again** (same settings, new seed), **Copy prompt** and **Open**. **Gallery** (bottom left) shows
all saved pictures; a click opens them large, with the raw version for pixel styles. **Ctrl+K** opens
a command palette (new sessions, sessions, gallery, ghost, theme); `/` jumps to the input.

Edit models live in `EDIT_MODELS` in `morpheus.py`:

| Model | Status | Time |
|---|---|---|
| FLUX.2 [klein] 4B | installed (same files as text-to-image), source as reference latent | ~20–30 s once loaded |

The output keeps the source's aspect ratio at about 1 MP; the time is sampling, not text encoding,
and halving the size only halves it. CLI: `morpheus.py "instruction" --edit path\to\image.png`.

### Ghost mode

The **Ghost** button (top right) turns on a mode where nothing is stored — for trying things out.
All modes work in a separate, unsaved session; the accent turns neon pink and ghost results get
dashed frames.

- Pictures and edits stay in the server's memory (last 40, `GHOST` in `server.py`), served from
  `/output/ghost/` with `Cache-Control: no-store`. No sidecar, no gallery entry.
- ComfyUI saves to its temp folder via `PreviewImage` instead of `ComfyUI/output`, and the file is
  deleted as soon as it is read; an edit's staged source in `ComfyUI/input` is deleted too.
- Ghost uploads stay in the page and in memory, never in `output/uploads`.
- Chat runs with `--no-session-persistence` (no session file); the page sends the ghost
  conversation along each turn instead, and none of it goes to `localStorage`.
- Turning ghost mode off, or restarting the server, discards everything.

Claude runs from `%TEMP%\imageer-chat` (outside `C:\dev`, so no project CLAUDE.md is loaded; still the
pre-rename name, so old chat sessions resume). It has no tools, except Read when a picture is
attached; the picture is copied into that folder for the turn and deleted right after.

## Command line

```
start_comfyui.ps1                                   # once per session, serves :8188
.venv\Scripts\python.exe morpheus.py "sleeping Link in bed at night" --style pixel --out C:\dev\talk_zelda
```

Prints one JSON line with the picture paths, seed and timing. Every picture also gets a `.json`
sidecar in `output/` (prompt, style, model, size, seed) — that is what the web gallery reads.

## Models

Measured 2026-10-30 at 1024², ComfyUI warm; "first" is the first render of that model, "next" a new
prompt with the model loaded.

| Model | Use | First / next |
|---|---|---|
| Z-Image-Turbo | **default**: photoreal, good at painting styles | 18 s / 18 s |
| FLUX.2 [klein] 4B | alternative look, fastest warm; also the edit model | 25 s / 13 s |
| NoobAI-XL 1.1 (SDXL) | anime / illustration, prompt as Danbooru tags; quality tags are prepended | 22 s / 14 s |

The chat uses Z-Image, also for lettering: short texts (signs, titles, neon) come out right; long
multi-line text sometimes repeats a word, then **Again** helps.

Qwen-Image-2512 (lettering, 175 s / 106 s) and Qwen-Image-Edit Rapid AIO (~12 min per edit) were
removed on 2026-10-30: loading Qwen-Image filled RAM plus page file (86 of 87 GB) and crashed
ComfyUI, and the edit model streamed its 28 GB through the 20 GB card.

`MODELS` in `morpheus.py` is the registry. A model is selectable once it has a workflow and all its
files exist under `ComfyUI/models`; the others show as "Not installed" in the UI.

## Styles

| Style | Prompt prefix | Post-processing |
|---|---|---|
| `none` | – | – |
| `pixel` | 16-bit SNES sprite | 128 px grid, 32 colours |
| `gameboy` | 8-bit Game Boy, green shades | 80 px grid, 4 colours |
| `photo` | professional photograph | – |

`--grid` and `--colors` override the pixel settings; `--seed` makes a result reproducible.

## Stack

- ComfyUI (cloned in `ComfyUI/`), Python 3.12 venv in `.venv/`
- PyTorch 2.9.1 for ROCm 7.2.1 on Windows, from `repo.radeon.com/rocm/windows/rocm-rel-7.2.1/`
- ComfyUI is started with `--reserve-vram` (VRAM held by other programs + 1 GB) and
  `--preview-method latent2rgb` for the live preview

## Gotchas

- **`av` is pinned to 17.0.0.** Windows Smart App Control blocks the DLLs of newer PyAV builds
  (17.0.1, 19.x); older ones lack APIs ComfyUI imports at startup. Don't let pip upgrade it.
- **Models as fp8.** With models at full precision, ComfyUI streams weights between RAM and VRAM and
  a picture takes minutes instead of seconds.
- **Z-Image's text encoder runs on the GPU** (a new prompt 59 s → ~20 s).
- **Reserve VRAM for the desktop.** Without `--reserve-vram`, a render that spills over what Windows
  and other apps hold slowed from 1.2 s to 81 s per step.
- The first runs after a fresh install are slower while ROCm warms up its kernels.
