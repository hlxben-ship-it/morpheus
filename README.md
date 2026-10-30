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

The server starts ComfyUI and renders a tiny throwaway picture at boot, so the first real prompt
doesn't pay the model load. Every render shows its step count and a live, pixelated preview; a
finished picture arrives with a scanline reveal. Jobs run in the background on the server, so a
reload mid-render picks the result up again.

The composer has four modes (Alt+1–4), each with its settings behind **Settings**:

- **Chat** — describe the picture; Claude (Claude Code CLI, `claude -p --model sonnet`) writes the
  English prompt and picks model, style and format, then the server renders. A follow-up like
  "now at sunrise" makes a new picture from an adjusted prompt. Each session keeps its own Claude
  conversation, and every message carries the session's latest picture, also one made in another
  mode. A picture can ride along with a message (paperclip, paste or drop, or **To chat** on any
  result): Claude looks at it and either **edits** exactly that picture or **generates** a new one in
  its style. With a reference picture, generate runs a **refine loop**: Claude compares each result
  with the reference and re-prompts until it is happy, at most 5 rounds; every round stays in the
  thread with its critique.
- **Manual** — the exact prompt and settings, no Claude in between. **Variant** on a result loads
  its prompt and settings back in.
- **Edit** — change an existing picture by instruction ("make it night"). The source is an upload
  (drag & drop) or any result via its **Edit** button; uploads land in `output/uploads/`, and an
  edited picture links back to its source ("Before"). The next edit builds on the last result.
- **EasySwap** — the face from a photo onto the largest face in a video section (`faceswap.py`,
  `.venv-faceswap`): High (HyperSwap 256 + GPEN 512), Premium (GPEN 1024), Fast (inswapper 128);
  optional hair recolouring and Real-ESRGAN ×2/×4 upscaling (capped at 4K).

Every result has **Again** (same settings, new seed), **Edit**, **Variant**, **To chat**, **Copy
prompt** and **Open**. **Gallery** (bottom left) shows all saved pictures; a click opens them large,
with the raw version for pixel styles. **Ctrl+K** opens a command palette (modes, sessions, gallery,
ghost, theme); `/` jumps to the input.

Edit models live in `EDIT_MODELS` in `morpheus.py`:

| Model | Status | Time |
|---|---|---|
| FLUX.2 [klein] 4B | installed (same files as text-to-image), source as reference latent | ~20–30 s once loaded |
| Qwen-Image-Edit Rapid AIO (Phr00t, NSFW v23) | one 28 GB checkpoint `checkpoints/Qwen-Rapid-AIO-NSFW-v23.safetensors` (model, encoder, VAE, accelerators); 4 steps, euler_ancestral/beta | ~12 min every time: 28 GB doesn't fit the 20 GB card, so ComfyUI streams weights |

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
| RealVisXL V5 (SDXL) | photorealistic people and scenes; overlaps with Z-Image | 25 s / 17 s |
| Qwen-Image-2512 + Lightning LoRA | best prompt adherence and lettering | 175 s / 106 s |
| Chroma1-HD | FLUX-based; real cfg, 26 steps, so slow | 104 s / 90 s |

The chat uses Z-Image unless the picture needs lettering or the user asks for top quality.

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
- **Z-Image's text encoder runs on the GPU** (a new prompt 59 s → ~20 s). Qwen-Image keeps its
  encoder on the CPU, next to its 19 GB model.
- **Reserve VRAM for the desktop.** Without `--reserve-vram`, a render that spills over what Windows
  and other apps hold slowed from 1.2 s to 81 s per step.
- The first runs after a fresh install are slower while ROCm warms up its kernels.
