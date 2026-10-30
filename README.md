# Morpheus

AI image studio — generate, edit, face swap — running locally on the AMD Radeon RX 7900 XT (20 GB),
driven from the web UI, the command line or by Glia. v0.1 (formerly Imageer).

## Web UI

```
start_morpheus.ps1        # serves http://127.0.0.1:5196 and opens the browser
```

Laid out like a chat app: past **sessions** on the left (kept in the browser's `localStorage`), the
current session in the middle with every result inline, and a composer at the bottom. Top right:
**Geist** (ghost mode) and the light/dark switch (follows the system until chosen). Branding: deep
indigo, dream violet, neon pink, moon cream; Montserrat for headings, IBM Plex Mono for text;
logo and favicon in `web/`.

The composer has four modes, each with its settings behind **Einstellungen**:

- **Chat** — describe the picture in German; Claude (Claude Code CLI, `claude -p --model sonnet`)
  writes the English prompt and picks model, style and format, then the server renders. A follow-up
  like "jetzt bei Sonnenaufgang" makes a new image from an adjusted prompt. Each session keeps its
  own Claude conversation.
- **Manuell** — the exact prompt and settings, no Claude in between. **Variante** on a result
  loads its prompt and settings back in.
- **Bearbeiten** — change an existing picture by instruction ("make it night"). The source is an
  upload (drag & drop) or any result via its **Bearbeiten** button; uploads land in
  `output/uploads/`, and an edited image links back to its source ("Vorher"). The next edit builds on
  the last result.
- **EasySwap** — the face from a photo onto the largest face in a video section (`faceswap.py`,
  `.venv-faceswap`): Hoch (HyperSwap 256 + GPEN 512), Premium (GPEN 1024), Schnell (inswapper 128);
  optional hair recolouring and Real-ESRGAN ×2/×4 upscaling (capped at 4K).

**Galerie** (bottom left) shows all saved images; a click opens them large, with the raw version
for pixel styles.

Edit models live in `EDIT_MODELS` in `morpheus.py`:

| Model | Status | Time |
|---|---|---|
| FLUX.2 [klein] 4B | installed (same files as text-to-image), source as reference latent | ~20 s once loaded |
| Qwen-Image-Edit Rapid AIO (Phr00t, NSFW v23) | one 28 GB checkpoint `checkpoints/Qwen-Rapid-AIO-NSFW-v23.safetensors` (model, encoder, VAE, accelerators); 4 steps, euler_ancestral/beta | ~12 min every time: 28 GB doesn't fit the 20 GB card, so ComfyUI streams weights |

The output keeps the source's aspect ratio at about 1 MP; the time is sampling, not text encoding,
and halving the size only halves it. CLI: `morpheus.py "instruction" --edit path\to\image.png`.

### Geist (ghost mode)

The **Geist** button (top right) turns on a mode where nothing is stored — for trying things out.
All modes work in a separate, unsaved session; the accent turns neon pink and ghost results get
dashed frames.

- Images and edits stay in the server's memory (last 40, `GHOST` in `server.py`), served from
  `/output/ghost/` with `Cache-Control: no-store`. No sidecar, no gallery entry.
- ComfyUI saves to its temp folder via `PreviewImage` instead of `ComfyUI/output`, and the file is
  deleted as soon as it is read; an edit's staged source in `ComfyUI/input` is deleted too.
- Ghost uploads stay in the page and in memory, never in `output/uploads`.
- Chat runs with `--no-session-persistence` (no session file); the page sends the ghost
  conversation along each turn instead, and none of it goes to `localStorage`.
- Turning ghost mode off, or restarting the server, discards everything.

Claude runs from `%TEMP%\imageer-chat` (outside `C:\dev`, so no project CLAUDE.md is loaded; still the pre-rename
name, so old chat sessions resume) with no tools.
`server.py` starts ComfyUI by itself with the first image if it isn't running.

## Command line

```
start_comfyui.ps1                                   # once per session, serves :8188
.venv\Scripts\python.exe morpheus.py "sleeping Link in bed at night" --style pixel --out C:\dev\talk_zelda
```

Prints one JSON line with the image paths, seed and timing. Every image also gets a `.json`
sidecar in `output/` (prompt, style, model, size, seed) — that is what the web gallery reads.

## Models

| Model | Use | First image / later |
|---|---|---|
| Z-Image-Turbo | fastest, weakest | ~70 s / ~10 s |
| FLUX.2 [klein] 4B | quick drafts | ~26 s / a few s |
| Qwen-Image-2512 + Lightning LoRA | best quality, text in images | ~3 min / shorter |
| RealVisXL V5 (SDXL) | photorealistic people and scenes | ~30 s |
| NoobAI-XL 1.1 (SDXL) | anime / illustration, prompt as Danbooru tags; quality tags are prepended | ~25 s |
| Chroma1-HD | FLUX-based, versatile; real cfg, 26 steps, so slower | ~1–2 min |

FLUX.2's text encoder must stay on the GPU (on the CPU it did not finish in 13 min); Z-Image and
Qwen keep theirs on the CPU to leave VRAM to the larger diffusion models.

`MODELS` in `morpheus.py` is the registry. A model is selectable once it has a workflow and all its
files exist under `ComfyUI/models`; the others show as "Nicht installiert" in the UI.

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
- Model: Z-Image-Turbo (Apache 2.0) — `z_image_turbo_bf16`, text encoder `qwen_3_4b`, VAE `ae`

## Gotchas

- **`av` is pinned to 17.0.0.** Windows Smart App Control blocks the DLLs of newer PyAV builds
  (17.0.1, 19.x); older ones lack APIs ComfyUI imports at startup. Don't let pip upgrade it.
- **Model as fp8, text encoder on the CPU.** With both on the GPU at full precision, ComfyUI streams
  weights between RAM and VRAM and a picture takes 3–6 min instead of ~10 s of sampling.
- Encoding a new prompt on the CPU takes about a minute; the first run after a start also loads the
  text encoder into RAM. Repeating a prompt with a new seed skips encoding.
- The first runs after a fresh install are slower while ROCm warms up its kernels.
