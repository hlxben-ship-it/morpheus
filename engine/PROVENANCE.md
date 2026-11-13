# engine/ — provenance

**The Morpheus video engine is a copy of WanGP.** Before adding or changing anything here, look how WanGP does it first:
its `defaults/*.json`, `profiles/`, the docs inside `models/<family>/*_handler.py`, `postprocessing/`.

| | |
|---|---|
| Source | WanGP by DeepBeepMeep, https://github.com/deepbeepmeep/WanGP |
| Version | **v19.29**, repo commit **`49f128de`** |
| Reference checkout | `C:/dev/wangp` (kept read-only; its spike scripts `morpheus_bench.py` / `morpheus_upscale.py` are the origin of `worker.py`) |
| Licence | WanGP Community License 2.0, full text in `wangp/LICENSE.txt`. Own use and modification are fine; no selling, white-labelling, paid API / SaaS. Keep this repo private and the notices in the files. |
| Copied | 2026-11-13, Glia seed `seed_2026_11_13_playground_wangp_native` (G1: copied as-is, cut, slim later) |

## What is WanGP's and what is ours

| Path | Origin |
|---|---|
| `wangp/wgp.py` | WanGP `wgp.py`. **One edit:** `family_handlers` lists only `models.ltx2.ltx2_handler` and `models.minimax_h3.minimax_h3_handler` (marked `# Morpheus`) |
| `wangp/shared/` | WanGP `shared/`, whole (slim later) |
| `wangp/mmgp/` | WanGP's in-tree `mmgp/` (memory manager: the reason 22B models run on 20 GB VRAM + 32 GB RAM) |
| `wangp/models/ltx2/` | WanGP `models/ltx2/` (LTX 2.3 two-stage pipeline, GGUF loading, distill LoRA, latent upscaler, Gemma encoder, audio) |
| `wangp/models/minimax_h3/` | WanGP `models/minimax_h3/` (H3 HyperFlow Ref2VA, Qwen3-VL-4B encoder, LightVAE, audio VAE) |
| `wangp/models/wan/`, `wangp/models/ideogram4/` | WanGP, only because `minimax_h3` imports modules from them |
| `wangp/models/__init__.py`, `model_metadata.py`, `help_shared.py`, `_settings.json` | WanGP `models/` top level |
| `wangp/postprocessing/` | WanGP `postprocessing/`, whole (RelayVSR, SeedVR2, the LTX upsampler, the upsampler catalog) |
| `wangp/preprocessing/` | WanGP `preprocessing/`, whole (imported at start) |
| `wangp/plugins/`, `plugins.json`, `profiles/`, `setup_config.json`, `requirements.txt` | WanGP, whole |
| `wangp/defaults/` | WanGP `defaults/`, only `ltx2*.json` and `minimax_h3*.json` (+ `ReadMe.txt`) |
| `wangp/finetunes/ltx2_22B_dev_q6.json` | **ours**: LTX 2.3 dev Q6_K GGUF as a WanGP finetune (from the spike) |
| `wangp/wgp_config.json` | WanGP's config from the spike, **paths changed**: checkpoints in `../../models/{ltx,h3,upscale}`, LoRAs in `../../models/loras`, output `../../output/engine` |
| `wangp/ffmpeg_bins/` | downloaded by WanGP on the first start (not in git) |
| `jobs.py`, `worker.py`, `runner.py`, `__init__.py` | **ours**: Playground snapshot → WanGP settings; the one-job worker process; the server-side launcher |
| `.venv/` | not in git: **hard links** to `C:/dev/wangp/venv` (Python 3.12.10, torch 2.9.1+rocm7.2.1). A *copy* is blocked by Smart App Control (torch's DLLs), hard links keep the same trusted files and cost no disk space |

## Models (G12)

One folder, `C:/dev/morpheus/models/` (not in git): `ltx/` (dev Q6 GGUF, VAE / audio VAE / vocoder / connector / projection,
spatial + temporal upscalers, Gemma 3 12B encoder), `h3/` (HyperFlow Flash 14B int8, audio VAE, `minimax_h3/` LightVAE + latent
upscaler, `P-H3-Image-Qwen3-VL-4B/`), `upscale/` (`RelayVSR/`, `SeedVR2/`), `loras/ltx2/` (distill LoRA 1.1, owner LoRAs).
All are **hard links** to the files in `C:/dev/wangp/ckpts` and `ComfyUI/models` (ComfyUI still reads its LTX files): deleting
the WanGP copies later frees nothing until the last link is gone, and loses nothing.
