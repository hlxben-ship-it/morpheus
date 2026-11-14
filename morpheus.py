"""morpheus — generate an image from a prompt via the local ComfyUI server.

Usage:
  python morpheus.py "sleeping Link in bed" --style pixel --out C:\\dev\\talk_zelda
  python morpheus.py "a cozy cabin at dusk" --style none --width 1344 --height 768
  python morpheus.py "make it night, warm light in the windows" --edit output\\cabin_raw.png

Styles add a prompt prefix and, for pixel art, a post-processing pass that snaps the
image onto a real pixel grid with a limited palette (diffusion models only fake it).
The web UI (server.py) uses the same run() as the command line.
"""
import shutil
import hashlib
import argparse
import asyncio
import json
import random
import re
import threading
import time
import uuid
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent
SERVER = "http://127.0.0.1:8188"
COMFY_MODELS = ROOT / "ComfyUI" / "models"
OUTPUT = ROOT / "output"
COMFY_INPUT = ROOT / "ComfyUI" / "input"
COMFY_TEMP = ROOT / "ComfyUI" / "temp"

STYLES = {
    "none": {"label": "No style", "prefix": "", "pixel": None},
    "pixel": {
        "label": "Retro 16-bit",
        "prefix": "16-bit SNES pixel art sprite, retro video game style, clean flat colors, "
                  "crisp outlines, limited palette, plain dark background. ",
        "pixel": {"grid": 128, "colors": 32},
    },
    "gameboy": {
        "label": "Game Boy",
        "prefix": "8-bit Game Boy pixel art, four shades of green, retro handheld game. ",
        "pixel": {"grid": 80, "colors": 4},
    },
    "photo": {"label": "Photo", "prefix": "professional photograph, natural light, sharp focus. ", "pixel": None},
}


def zimage_workflow(prompt: str, width: int, height: int, seed: int, steps: int, loras: list | None = None) -> dict:
    """ComfyUI API graph for Z-Image-Turbo (distilled: few steps, cfg 1).

    The text encoder runs on the CPU and the diffusion model loads as fp8: on the 20 GB card,
    both on the GPU at full precision forces ComfyUI to stream weights and is ~25x slower.
    loras are LORAS keys (None = the defaults), chained after the model.
    """
    graph, model = lora_chain({}, ["1", 0], lora_keys("z-image-turbo", loras))
    return graph | {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model, "shift": 3.0}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0],
            "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "res_multistep",
            "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "morpheus"}},
    }


def flux2_klein_workflow(prompt: str, width: int, height: int, seed: int, steps: int) -> dict:
    """FLUX.2 [klein] 4B, distilled: 4 steps, cfg 1, after ComfyUI's image_flux2_klein_text_to_image template.

    It shares the Qwen3-4B text encoder file with Z-Image. Unlike Z-Image, the encoder stays on the
    GPU: FLUX.2's text encoding on the CPU did not finish in 13 minutes, and model (7.7 GB) plus
    encoder (8 GB) fit the 20 GB card together.
    """
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-4b.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "11": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": width, "height": height}},
        "12": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0], "cfg": 1.0}},
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["8", 0], "guider": ["12", 0], "sampler": ["9", 0], "sigmas": ["11", 0], "latent_image": ["7", 0]}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": "morpheus"}},
    }


def sdxl_workflow(ckpt: str, sampler: str, scheduler: str, cfg: float, negative: str, prefix: str = "", clip_skip: int = 1,
                  hires: float = 1.0):
    """Build a workflow function for an SDXL (or SD 1.5) checkpoint (model, text encoders and VAE in one file).

    negative is the default; a call can pass its own (Manual's Negative field). build.negative exposes the default.
    clip_skip 2 reads the text encoder's second-to-last layer, which Pony was trained with.
    hires above 1 is a hires fix: the picture is scaled up in latent space and sampled again at half strength, for
    SD 1.5, whose small native size looks soft. width and height stay the first pass, the output is hires times larger.
    """
    default, default_prefix = negative, prefix

    def build(prompt: str, width: int, height: int, seed: int, steps: int, negative: str | None = None,
              prefix: str | None = None, loras: list | None = None) -> dict:
        negative = default if negative is None else negative
        prefix = default_prefix if prefix is None else prefix
        clip = ["1", 1] if clip_skip == 1 else ["2", 0]
        graph = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": prefix + prompt}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": negative}},
            "7": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "8": {"class_type": "KSampler", "inputs": {
                "model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0],
                "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0}},
            "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
            "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "morpheus"}},
        }
        if clip_skip != 1:
            graph["2"] = {"class_type": "CLIPSetLastLayer", "inputs": {"clip": ["1", 1], "stop_at_clip_layer": -clip_skip}}
        if hires != 1:
            graph["11"] = {"class_type": "LatentUpscaleBy", "inputs": {"samples": ["8", 0], "upscale_method": "nearest-exact", "scale_by": hires}}
            graph["12"] = {"class_type": "KSampler", "inputs": {
                "model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["11", 0],
                "seed": seed, "steps": max(10, steps // 2), "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "denoise": 0.5}}
            graph["9"]["inputs"]["samples"] = ["12", 0]
        graph, model = lora_chain(graph, ["1", 0], loras or [])
        graph["8"]["inputs"]["model"] = model
        if "12" in graph:
            graph["12"]["inputs"]["model"] = model
        triggers = "".join(LORAS[k]["trigger"] + ", " for k in loras or [] if LORAS[k].get("trigger"))
        graph["5"]["inputs"]["text"] = triggers + graph["5"]["inputs"]["text"]
        return graph
    build.negative, build.prefix = default, default_prefix
    return build


def flux2_klein_edit_workflow(prompt: str, image: str, width: int, height: int, seed: int, steps: int) -> dict:
    """FLUX.2 [klein] 4B as an editor, after ComfyUI's image_flux2_klein_image_edit template.

    The source image is VAE-encoded and attached to both conditionings as a reference latent;
    the model follows the instruction and keeps what it does not mention.
    """
    graph = flux2_klein_workflow(prompt, width, height, seed, steps)
    graph.update({
        "20": {"class_type": "LoadImage", "inputs": {"image": image}},
        "21": {"class_type": "ImageScale", "inputs": {"image": ["20", 0], "upscale_method": "lanczos",
                                                      "width": width, "height": height, "crop": "center"}},
        "22": {"class_type": "VAEEncode", "inputs": {"pixels": ["21", 0], "vae": ["3", 0]}},
        "23": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["5", 0], "latent": ["22", 0]}},
        "24": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["6", 0], "latent": ["22", 0]}},
    })
    graph["12"]["inputs"].update({"positive": ["23", 0], "negative": ["24", 0]})
    return graph


def qwen_edit_workflow(prompt: str, image: str, width: int, height: int, seed: int, steps: int, loras: list | None = None) -> dict:
    """Qwen-Image-Edit-2511 (20B, Q5_K_M GGUF) with the 4-step Lightning LoRA, after ComfyUI's image_qwen_image_edit_2511
    template. Its Qwen2.5-VL encoder reads the picture along with the instruction, which keeps who she is through
    an outfit change better than Klein, at the cost of loading 22 GB in turns.

    loras are LORAS keys (None = the defaults), chained after the model. Without a speed LoRA (Lightning) it samples
    the base model's way: 40 steps at cfg 4, about ten times slower.
    """
    loras = lora_keys("qwen-image-edit-2511", loras)
    graph, model = lora_chain({"1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q5_K_M.gguf"}}},
                              ["1", 0], loras)
    fast = any(LORAS[k].get("speed") for k in loras)
    if not fast:
        steps = 40
    return graph | {
        "16": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model, "shift": 3.1}},
        "17": {"class_type": "CFGNorm", "inputs": {"model": ["16", 0], "strength": 1.0}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "20": {"class_type": "LoadImage", "inputs": {"image": image}},
        "21": {"class_type": "ImageScale", "inputs": {"image": ["20", 0], "upscale_method": "lanczos",
                                                      "width": width, "height": height, "crop": "center"}},
        "5": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["2", 0], "prompt": prompt, "vae": ["3", 0], "image1": ["21", 0]}},
        "6": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["2", 0], "prompt": "", "vae": ["3", 0], "image1": ["21", 0]}},
        "25": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["5", 0], "reference_latents_method": "index_timestep_zero"}},
        "26": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["6", 0], "reference_latents_method": "index_timestep_zero"}},
        "22": {"class_type": "VAEEncode", "inputs": {"pixels": ["21", 0], "vae": ["3", 0]}},
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["17", 0], "positive": ["25", 0], "negative": ["26", 0], "latent_image": ["22", 0],
            "seed": seed, "steps": steps, "cfg": 1.0 if fast else 4.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "morpheus"}},
    }


SDXL_NEGATIVE ="lowres, blurry, out of focus, deformed, bad anatomy, bad hands, extra limbs, mutated, watermark, text, logo, signature"

# A model is usable when all its files are present under ComfyUI/models. Entries without a
# workflow are listed so the UI can show what could be added, but cannot be selected yet.
# Every workflow ends in node "10" (SaveImage), which generate() reads the result from.
MODELS = {
    "z-image-turbo": {
        "category": "realism",
        "family": "Z-Image",
        "label": "Z-Image-Turbo",
        "note": "Fastest, Apache 2.0",
        "files": ["diffusion_models/z_image_turbo_bf16.safetensors", "text_encoders/qwen_3_4b.safetensors", "vae/ae.safetensors"],
        "steps": 8,
        "workflow": zimage_workflow,
    },
    "flux2-klein-4b": {
        "category": "realism",
        "family": "FLUX.2",
        "label": "FLUX.2 [klein] 4B",
        "note": "Balanced, Apache 2.0",
        "files": ["diffusion_models/flux-2-klein-4b.safetensors", "text_encoders/qwen_3_4b.safetensors", "vae/flux2-vae.safetensors"],
        "steps": 4,
        "workflow": flux2_klein_workflow,
    },
    "noobai-xl": {
        "category": "manga",
        "family": "SDXL",
        "label": "NoobAI-XL 1.1 (SDXL)",
        "note": "Anime and illustration, understands Danbooru tags",
        "files": ["checkpoints/NoobAI-XL-v1.1.safetensors"],
        "steps": 28,
        "workflow": sdxl_workflow("NoobAI-XL-v1.1.safetensors", "euler_ancestral", "normal", 5.0,
                                  "worst quality, low quality, old, early, " + SDXL_NEGATIVE,
                                  prefix="masterpiece, best quality, newest, absurdres, highres, "),
    },
    # civitai.com/models/1277670 v5.0 (Illustrious/NoobAI/RouWei finetune); author: Euler a, cfg 3-5, 25-30 steps,
    # "nsfw" in the negative for safe pictures
    "janku-v5": {
        "category": "manga",
        "family": "SDXL",
        "label": "JANKU v5.0 (SDXL)",
        "note": "Polished modern anime, Danbooru tags, artist styles via \"by name\"",
        "files": ["checkpoints/JANKUTrainedChenkinNoobai_v50.safetensors"],
        "steps": 30,
        "manual_only": True,  # the chat doesn't offer it
        "workflow": sdxl_workflow("JANKUTrainedChenkinNoobai_v50.safetensors", "euler_ancestral", "normal", 5.0,
                                  "nsfw, worst quality, low quality, " + SDXL_NEGATIVE,
                                  prefix="masterpiece, best quality, absurdres, "),
    },
    # civitai.com/models/1025051 Illustrij v21 (Illustrious merge, VAE baked in); the author's v20 settings: Euler a, 30 steps,
    # cfg 7, their quality tags. File needs a Civitai login to download.
    "illustrij-v21": {
        "category": "manga",
        "family": "SDXL",
        "label": "Illustrij v21 (SDXL)",
        "note": "Hybrid anime and comic, 2.5D toward semi-real, Danbooru tags",
        "files": ["checkpoints/Illustrijv21.safetensors"],
        "steps": 30,
        "manual_only": True,  # the chat doesn't offer it
        "workflow": sdxl_workflow("Illustrijv21.safetensors", "euler_ancestral", "normal", 7.0,
                                  "worst quality, bad quality, " + SDXL_NEGATIVE,
                                  prefix="masterpiece, best quality, "),
    },
    # civitai.com/models/4384 DreamShaper 8, SD 1.5: trained at 512 px, so "pixels" renders the picked format at about
    # 0.4 MP (1024 px would double bodies and faces); DPM++ 2M Karras, cfg 7
    "dreamshaper-8": {
        "category": "realism",
        "family": "SD 1.5",
        "label": "DreamShaper 8 (SD 1.5)",
        "note": "All-rounder: fantasy, portraits, semi-real; small native size",
        "files": ["checkpoints/dreamshaper_8.safetensors"],
        "steps": 30,
        "pixels": 512 * 768,
        "manual_only": True,
        "workflow": sdxl_workflow("dreamshaper_8.safetensors", "dpmpp_2m", "karras", 7.0,
                                  "worst quality, low quality, " + SDXL_NEGATIVE, hires=1.5),
    },
    # civitai.com/models/376130 Nova Anime XL IL v19.0 (Illustrious: NoobAI + Illustrious 2.0 + ChenkinNoob merge);
    # the author's Illustrious settings: Euler a, cfg 4-6, 20-30 steps, their quality tags and negative
    "nova-anime-xl": {
        "category": "manga",
        "family": "SDXL",
        "label": "Nova Anime XL v19 (SDXL)",
        "note": "Anime / 2.5D, high contrast, detailed backgrounds; Danbooru tags",
        "files": ["checkpoints/novaAnimeXL_ilV190.safetensors"],
        "steps": 28,
        "manual_only": True,
        "workflow": sdxl_workflow("novaAnimeXL_ilV190.safetensors", "euler_ancestral", "normal", 5.0,
                                  "modern, recent, old, oldest, cartoon, graphic, text, painting, crayon, graphite, abstract, "
                                  "glitch, deformed, mutated, ugly, disfigured, long body, lowres, bad anatomy, bad hands, "
                                  "missing fingers, extra digits, fewer digits, cropped, very displeasing, "
                                  "(worst quality, bad quality:1.2), sketch, jpeg artifacts, signature, watermark, username, "
                                  "simple background, conjoined, bad ai-generated",
                                  prefix="masterpiece, best quality, amazing quality, 4k, very aesthetic, high resolution, "
                                         "ultra-detailed, absurdres, newest, "),
    },
    # Meina V12, SD 1.5 anime; author: DPM++ 2M Karras 20-60 steps, cfg 4-9, CLIP skip 2, 512x768. Their hires.fix isn't
    # available here, so this uses their negative for going without it
    "meina-v12": {
        "category": "manga",
        "family": "SD 1.5",
        "label": "MeinaMix V12 (SD 1.5)",
        "note": "Soft bright anime; faces want a close framing; small native size",
        "files": ["checkpoints/MeinaMix_V12.safetensors"],
        "steps": 30,
        "pixels": 512 * 768,
        "manual_only": True,
        "workflow": sdxl_workflow("MeinaMix_V12.safetensors", "dpmpp_2m", "karras", 7.0,
                                  "(worst quality:1.6, low quality:1.6), (zombie, sketch, interlocked fingers, comic)", clip_skip=2, hires=1.5),
    },
    # RealCartoon3D V18, SD 1.5 3D cartoon; author: Euler a, VAE baked in, 512x768 or 512x904. Author names no steps or cfg, so
    # 30 and 7 are the usual SD 1.5 values. Their easynegative and badhandv4 embeddings aren't installed and are left out
    "realcartoon3d-v18": {
        "category": "manga",
        "family": "SD 1.5",
        "label": "RealCartoon3D V18 (SD 1.5)",
        "note": "3D cartoon look; add Realistic or Cartoon at the front to lean either way; small native size",
        "files": ["checkpoints/RealCartoon3D_V18.safetensors"],
        "steps": 30,
        "pixels": 512 * 768,
        "manual_only": True,
        "workflow": sdxl_workflow("RealCartoon3D_V18.safetensors", "euler_ancestral", "normal", 7.0,
                                  "(bad quality:1.3), (worst quality:1.3), watermark, (blurry)", hires=1.5),
    },
    # CetusMix Coda 1, SD 1.5 anime; author: DPM++ 2M Karras 20+ steps, cfg 4-8, CLIP skip 2. Their Pastel VAE, hires.fix and
    # bad_prompt_v2 embedding aren't available, so this uses the baked VAE and spells out that embedding's kind of negative
    "cetusmix-coda1": {
        "category": "manga",
        "family": "SD 1.5",
        "label": "CetusMix Coda 1 (SD 1.5)",
        "note": "Pastel anime; can look soft without an upscale; small native size",
        "files": ["checkpoints/CetusMix_Coda1.safetensors"],
        "steps": 25,
        "pixels": 512 * 768,
        "manual_only": True,
        "workflow": sdxl_workflow("CetusMix_Coda1.safetensors", "dpmpp_2m", "karras", 6.0,
                                  "lowres, bad anatomy, bad hands, (worst quality, low quality:1.4), blurry, text, watermark",
                                  clip_skip=2, hires=1.5),
    },
    # Hassaku XL V3.4, Illustrious line, Fair AI Public License 1.0-SD; author: Euler a, cfg 6, 832x1216 most recommended, "signature"
    # in the negative. Tag order: number of characters, character, the rest. No metadata or franchise tags. Author names no steps
    "hassaku-v3-4": {
        "category": "manga",
        "family": "SDXL",
        "label": "Hassaku XL V3.4",
        "note": "Anime tags: 1girl, character, the rest; no franchise or highres tags",
        "files": ["checkpoints/hassakuV3_4.safetensors"],
        "steps": 30,
        "pixels": 832 * 1216,
        "manual_only": True,
        "workflow": sdxl_workflow("hassakuV3_4.safetensors", "euler_ancestral", "normal", 6.0, "signature, " + SDXL_NEGATIVE),
    },
    # HC Tai Pony v1.1, an SDXL Pony merge with its VAE baked in (header: hardcore_hentai_pony_v1.1). Author's text is for SD 1.5
    # sizes (512x768), which don't suit SDXL, so sizes, score tags and CLIP skip 2 follow Pony. From the author: DPM++ 2M Karras, 25-50 steps,
    # cfg 7, Danbooru tags or sentences, tag only what is visible, generic negative
    "hc-tai-pony": {
        "category": "manga",
        "family": "Pony",
        "label": "HC Tai Pony v1.1 (SDXL)",
        "note": "Explicit-leaning Pony: Danbooru tags or sentences, tag only what shows",
        "files": ["checkpoints/hc_tai_pony_v1.1.safetensors"],
        "steps": 30,
        "manual_only": True,
        "workflow": sdxl_workflow("hc_tai_pony_v1.1.safetensors", "dpmpp_2m", "karras", 7.0,
                                  "score_4, score_5, score_6, worst quality, low quality, bad quality, normal quality, bad anatomy, monochrome",
                                  prefix="score_9, score_8_up, score_7_up, ", clip_skip=2),
    },
    # civitai.com/models/4468 Counterfeit V3.0, SD 1.5 anime; natural language or tags, CLIP skip 2 like the NovelAI-era
    # anime models, DPM++ 2M Karras, cfg 7; small native size like the other SD 1.5 models
    "counterfeit-v3": {
        "category": "manga",
        "family": "SD 1.5",
        "label": "Counterfeit V3.0 (SD 1.5)",
        "note": "Classic SD 1.5 anime, free composition; small native size",
        "files": ["checkpoints/counterfeitV30_v30_pruned_fp16.safetensors"],
        "steps": 30,
        "pixels": 512 * 768,
        "manual_only": True,
        "workflow": sdxl_workflow("counterfeitV30_v30_pruned_fp16.safetensors", "dpmpp_2m", "karras", 7.0,
                                  "(worst quality, low quality:1.4), " + SDXL_NEGATIVE,
                                  prefix="masterpiece, best quality, ", clip_skip=2, hires=1.5),
    },
    # civitai.com/models/15003 CyberRealistic Final, SD 1.5 photoreal; author: DPM++ 2M Karras, 30 steps, cfg 5, 512x768
    "cyberrealistic": {
        "category": "realism",
        "family": "SD 1.5",
        "label": "CyberRealistic Final (SD 1.5)",
        "note": "Photoreal people and portraits; small native size",
        "files": ["checkpoints/cyberrealistic_final_pruned_fp16.safetensors"],
        "steps": 30,
        "pixels": 512 * 768,
        "manual_only": True,
        "workflow": sdxl_workflow("cyberrealistic_final_pruned_fp16.safetensors", "dpmpp_2m", "karras", 5.0,
                                  "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, "
                                  "cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, "
                                  "username, blurry", hires=1.5),
    },
    # civitai.com/models/372465 Pony Realism v2.2 (Pony finetune, photoreal); author: DPM++ SDE / Euler a, 30+ steps,
    # cfg 6-7, CLIP skip 2, score_9/8_up/7_up before, score_4/5/6 in the negative, "female"/"male" over "woman"/"man"
    "pony-realism": {
        "category": "realism",
        "family": "Pony",
        "label": "Pony Realism v2.2 (SDXL)",
        "note": "Photoreal Pony: Danbooru tags, score tags, female / male",
        "files": ["checkpoints/ponyRealism_V22.safetensors"],
        "steps": 30,
        "manual_only": True,
        "workflow": sdxl_workflow("ponyRealism_V22.safetensors", "euler_ancestral", "normal", 6.5,
                                  "score_4, score_5, score_6, rating_explicit, " + SDXL_NEGATIVE,
                                  prefix="score_9, score_8_up, score_7_up, ", clip_skip=2),
    },
    # civitai.com/models/257749 V6 "start with this one"; score tags lead the prompt, CLIP skip 2, cfg 7, 25 steps
    "pony-v6": {
        "category": "manga",
        "family": "Pony",
        "label": "Pony Diffusion V6 XL (SDXL)",
        "note": "Cartoon, anime, furry; score tags + source_anime / rating_safe",
        "files": ["checkpoints/ponyDiffusionV6XL_v6StartWithThisOne.safetensors"],
        "steps": 25,
        "manual_only": True,
        "workflow": sdxl_workflow("ponyDiffusionV6XL_v6StartWithThisOne.safetensors", "euler_ancestral", "normal", 7.0,
                                  "score_4, score_3, score_2, score_1, rating_explicit, " + SDXL_NEGATIVE,
                                  prefix="score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, ",
                                  clip_skip=2),
    },
}


NOCT_NEGATIVE = ("artifacts, gpt-image, washed-out colors, low quality, low resolution, AI slop, deviantart, sloppy lines, rough sketch, "
                 "blurry, indistinct, missing fingers, badly drawn hands, wrong number of fingers")


def noct_edit_workflow(prompt: str, image: str, width: int, height: int, seed: int, steps: int) -> dict:
    """Noct Q V4 (an uncensored merge of Qwen Image 2.1, int8) as an edit: the author's settings, 25 steps at cfg 3 with
    their negative. The encoder sizes the picture to about 1024 x 1024 pixels at multiples of 32 and hands over the latent."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "NoctQ_V4_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "20": {"class_type": "LoadImage", "inputs": {"image": image}},
        "5": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": prompt, "negative_prompt": NOCT_NEGATIVE, "vae": ["3", 0],
                                                                "resolution": 1024, "images.image_1": ["20", 0]}},
        "8": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["5", 2],
                                                   "seed": seed, "steps": steps, "cfg": 3.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "morpheus"}},
    }


# LoRAs under ComfyUI/models/loras, each for one model. Manual shows a checkbox per installed LoRA of the picked
# model; "default" ones are on unless the request says otherwise (chat, Soulforge, CLI). "speed" marks a step
# distillation: the workflow drops to few steps only while one of those is on.
LORAS = {
    # Trained in the Yumeko spike: 31 JANKU pictures, 2000 steps, UNet only. "trigger" goes in front of the prompt while it is on
    "zzsorc-v1": {
        "label": "Gothic Sorceress",
        "note": "Trained character with moon earrings; hair behind ear shows them best",
        "file": "zzsorc_v1.safetensors",
        "model": "janku-v5",
        "strength": 1.0,
        "trigger": "zzsorc",
    },
    "qwen-lightning": {
        "label": "Lightning 4-step",
        "note": "About 10x faster; off = 40 full steps",
        "file": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "model": "qwen-image-edit-2511",
        "strength": 1.0,
        "default": True,
        "speed": True,
    },
    # lrzjason/Anything2Real on Hugging Face; prompt as "change the picture 1 to realistic photograph, <description>"
    "qwen-anything2real": {
        "label": "Anything2Real",
        "note": "Anime, art, cartoon → photo; prompt: change the picture 1 to realistic photograph, …",
        "file": "anything2real_2601_A_final_patched.safetensors",
        "model": "qwen-image-edit-2511",
        "strength": 0.85,
        "default": False,
    },
    # civitai.com/models/667086 "NSFW MASTER", Z-Image Turbo V2; author: weight 0.8. Never on by default: the chat
    # only ever runs the defaults.
    "zimage-nsfw-master": {
        "label": "NSFW Master",
        "note": "Realistic nude anatomy, adults only; strength 0.8",
        "file": "NSFW_master_ZIT_000017532.safetensors",
        "model": "z-image-turbo",
        "strength": 0.8,
        "default": False,
    },
    # fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA; prompt as "<sks> [azimuth] [elevation] [distance]"
    "qwen-multiple-angles": {
        "label": "Multiple Angles",
        "note": "New camera angle; prompt: <sks> front-left quarter view low-angle shot medium shot",
        "file": "qwen-image-edit-2511-multiple-angles-lora.safetensors",
        "model": "qwen-image-edit-2511",
        "strength": 1.0,
        "default": False,
    },
    # fal/Qwen-Image-Edit-2511-SFW
    "qwen-sfw": {
        "label": "SFW",
        "note": "Generate SFW images",
        "file": "qwen-image-edit-2511-sfw-lora.safetensors",
        "model": "qwen-image-edit-2511",
        "strength": 1.0,
        "default": False,
    },
}


def lora_chain(graph: dict, model: list, loras: list) -> tuple[dict, list]:
    """Chain LoraLoaderModelOnly nodes (ids 40, 41, ...) after model; returns the graph and the last model output."""
    for i, k in enumerate(loras):
        graph[f"4{i}"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": model, "lora_name": LORAS[k]["file"], "strength_model": LORAS[k]["strength"]}}
        model = [f"4{i}", 0]
    return graph, model


def has_loras(model: str) -> bool:
    """Whether the model's workflow takes loras (it has entries in LORAS)."""
    return any(l["model"] == model for l in LORAS.values())


def lora_installed(key: str) -> bool:
    return (COMFY_MODELS / "loras" / LORAS[key]["file"]).exists()


def lora_keys(model: str, wanted: list | None = None) -> list:
    """The LoRAs to use for model: the wanted ones (None = its defaults), installed and meant for it, in LORAS order."""
    return [k for k, l in LORAS.items() if l["model"] == model and lora_installed(k)
            and (k in wanted if wanted is not None else l.get("default"))]


# Edit models take a source image plus an instruction. Same rules as MODELS; the workflow gets the
# image's name in ComfyUI/input and an output size that follows the source's aspect ratio.
EDIT_MODELS = {
    "noct-q-v4": {
        "family": "Qwen",
        "manual_note": "Edits a source picture, nothing filtered; write full sentences; slow, 25 steps",
        "label": "Noct Q V4 (uncensored)",
        "note": "Qwen Image 2.1 merge, nothing filtered; write full sentences; slow, 25 steps",
        "files": ["diffusion_models/NoctQ_V4_int8_convrot.safetensors", "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
                  "vae/qwen_image_2.1_vae_bf16.safetensors"],
        "steps": 25,
        "manual_only": True,  # the chat doesn't offer it
        "workflow": noct_edit_workflow,
    },
    "flux2-klein-4b-edit": {
        "family": "FLUX.2",
        "manual_note": "Edits a source picture: fast, leaves the rest alone",
        "label": "FLUX.2 [klein] 4B",
        "note": "Fast, leaves the rest of the picture alone",
        "files": MODELS["flux2-klein-4b"]["files"],
        "steps": 4,
        "workflow": flux2_klein_edit_workflow,
    },
    "qwen-image-edit-2511": {
        "family": "Qwen",
        "manual_note": "Edits a source picture: restyles, keeps the person",
        "label": "Qwen-Image-Edit 2511",
        "note": "Best at outfit and detail changes, keeps the person and style; slower",
        "files": ["unet/qwen-image-edit-2511-Q5_K_M.gguf",
                  "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "vae/qwen_image_vae.safetensors"],
        "steps": 4,
        "pixel_art": True,  # keeps a pixel-art source's grid and flat colours on its own (Soulforge skips its restyle)
        "workflow": qwen_edit_workflow,
    },
}


def default_prefix(key: str) -> str:
    """The quality tags the model puts before the prompt ("" for none). Manual passes prefix="" and shows them instead."""
    return getattr((MODELS.get(key) or {}).get("workflow"), "prefix", "")


def default_negative(key: str) -> str | None:
    """The model's default negative prompt, or None for models without one (the distilled ones run at cfg 1)."""
    return getattr((MODELS.get(key) or {}).get("workflow"), "negative", None)


def installed(key: str) -> bool:
    m = MODELS.get(key) or EDIT_MODELS[key]
    return m["workflow"] is not None and all((COMFY_MODELS / f).exists() for f in m["files"])


def call(path: str, data: dict | None = None) -> dict:
    req = urllib.request.Request(SERVER + path, data=json.dumps(data).encode() if data else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def comfy_up() -> bool:
    try:
        call("/system_stats")
        return True
    except OSError:
        return False


def watch(client: str, progress, ready: threading.Event, done: threading.Event) -> None:
    """Follow one client's ComfyUI websocket: sampler steps as progress(step=, total=), and the live
    preview frames (JPEG, sent when ComfyUI runs with --preview-method) as progress(preview=bytes)."""
    import aiohttp  # ComfyUI's own dependency, so it is in the venv

    async def run():
        async with aiohttp.ClientSession() as http, http.ws_connect(f"{SERVER.replace('http', 'ws')}/ws?clientId={client}") as ws:
            ready.set()
            while not done.is_set():
                try:
                    msg = await ws.receive(timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("type") == "progress":
                        progress(step=data["data"]["value"], total=data["data"]["max"])
                elif msg.type == aiohttp.WSMsgType.BINARY and msg.data[:4] == b"\x00\x00\x00\x01":
                    progress(preview=msg.data[8:])  # event 1 = preview image, then 4 bytes format, then the image
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    return
    try:
        asyncio.run(run())
    except Exception:
        pass  # progress is a nicety; the render itself is followed through /history
    finally:
        ready.set()


def generate(graph: dict, ghost: bool = False, progress=None) -> Image.Image:
    """Run a graph and return its image. Ghost runs save to ComfyUI's temp folder instead of its output
    folder, and the file is deleted again as soon as it is read. progress, if given, gets the live steps
    and preview frames (see watch)."""
    if ghost:
        graph = {**graph, "10": {"class_type": "PreviewImage", "inputs": {"images": graph["10"]["inputs"]["images"]}}}
    client, done = uuid.uuid4().hex, threading.Event()
    if progress:
        ready = threading.Event()
        threading.Thread(target=watch, args=(client, progress, ready, done), daemon=True).start()
        ready.wait(3)
    try:
        try:
            return wait(call("/prompt", {"prompt": graph, "client_id": client})["prompt_id"], ghost)
        except RuntimeError as e:
            if "has no attribute 'model_size'" not in str(e):
                raise
            # a dead model left in ComfyUI's list (see comfy_ext/morpheus_status.py): clear it, try once more
            call("/morpheus/cleanup", {"now": True})  # call() sends an empty dict as a GET
            return wait(call("/prompt", {"prompt": graph, "client_id": client})["prompt_id"], ghost)
    finally:
        done.set()


def wait(prompt_id: str, ghost: bool) -> Image.Image:
    while True:
        hist = call(f"/history/{prompt_id}").get(prompt_id)
        if hist and hist.get("status", {}).get("completed"):
            break
        if hist and hist.get("status", {}).get("status_str") == "error":
            raise RuntimeError(f"ComfyUI error: {json.dumps(hist['status'])}")
        time.sleep(0.5)
    img = hist["outputs"]["10"]["images"][0]
    query = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]})
    with urllib.request.urlopen(f"{SERVER}/view?{query}") as r:
        result = Image.open(BytesIO(r.read())).convert("RGB")
    if ghost:
        (COMFY_TEMP / img["subfolder"] / img["filename"]).unlink(missing_ok=True)
    return result


def pixelate(img: Image.Image, grid: int, colors: int) -> Image.Image:
    """Snap to a real pixel grid: average down, cut the palette, scale back up with hard edges."""
    w, h = img.size
    small = img.resize((grid, round(grid * h / w)), Image.Resampling.BOX)
    # Fast octree keeps small saturated areas (a green cap, blond hair) that median cut drops for background shades.
    small = small.quantize(colors=colors, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).convert("RGB")
    return small.resize((w, h), Image.Resampling.NEAREST)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "image"


def run(prompt: str, style: str = "none", model: str = "z-image-turbo", width: int = 1024, height: int = 1024,
        seed: int | None = None, steps: int | None = None, out: Path = OUTPUT, name: str | None = None,
        grid: int | None = None, colors: int | None = None, ghost: bool = False, progress=None,
        negative: str | None = None, loras: list | None = None, prefix: str | None = None) -> dict:
    """Generate one image, save it (plus raw and a metadata sidecar) and return the metadata.

    A ghost run writes nothing: the metadata carries the images themselves under "pictures" instead.
    """
    if not installed(model):
        raise ValueError(f"Model '{model}' is not installed.")
    m, st = MODELS[model], STYLES[style]
    seed = seed if seed is not None else random.randrange(2**32)
    if m.get("pixels"):
        width, height = edit_size(width, height, m["pixels"])  # an older model's native size, same aspect ratio
    # Model and time in the name: the same prompt and seed on another model must not overwrite an earlier image.
    name = name or f"{time.strftime('%Y%m%d-%H%M%S')}_{model}_{slug(prompt)[:30]}_{seed}"

    started = time.time()
    # only a model with a negative prompt (SDXL) takes one; None = its default
    neg = {"negative": negative} if negative is not None and default_negative(model) is not None else {}
    if has_loras(model):
        neg["loras"] = lora_keys(model, loras)
    if prefix is not None and default_prefix(model):
        neg["prefix"] = prefix
    img = generate(m["workflow"](st["prefix"] + prompt, width, height, seed, steps or m["steps"], **neg), ghost, progress)
    final = pixelate(img, grid or st["pixel"]["grid"], colors or st["pixel"]["colors"]) if st["pixel"] else img
    meta = {"image": f"{name}.png" if st["pixel"] else f"{name}_raw.png", "raw": f"{name}_raw.png", "prompt": prompt,
            "style": style, "model": model, "width": width, "height": height, "seed": seed,
            "seconds": round(time.time() - started, 1), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    if "negative" in neg:
        meta["negative"] = negative
    if "loras" in neg:
        meta["loras"] = neg["loras"]
    return keep(meta, final, img, out, name, ghost)


def keep(meta: dict, final: Image.Image, raw: Image.Image, out: Path, name: str, ghost: bool) -> dict:
    """Save images and sidecar under out/, or for a ghost run hand the images back in the metadata."""
    if ghost:
        return {**meta, "pictures": {"image": final, "raw": raw}}
    out.mkdir(parents=True, exist_ok=True)
    raw.save(out / meta["raw"])
    if meta["image"] != meta["raw"]:
        final.save(out / meta["image"])
    (out / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def edit_size(width: int, height: int, pixels: int = 1024 * 1024) -> tuple[int, int]:
    """Keep the source's aspect ratio at about one megapixel, both sides a multiple of 16."""
    scale = (pixels / (width * height)) ** 0.5
    return max(16, round(width * scale / 16) * 16), max(16, round(height * scale / 16) * 16)


def run_edit(prompt: str, source: Path | Image.Image, model: str = "flux2-klein-4b-edit", seed: int | None = None,
             steps: int | None = None, out: Path = OUTPUT, name: str | None = None, ghost: bool = False,
             source_ref: str | None = None, progress=None, loras: list | None = None) -> dict:
    """Edit an existing image by instruction, save the result and a metadata sidecar, return the metadata.

    The source is a file or, for ghost runs, an image held in memory. A ghost run writes nothing that outlives it.
    """
    if not installed(model):
        raise ValueError(f"Model '{model}' is not installed.")
    m = EDIT_MODELS[model]
    seed = seed if seed is not None else random.randrange(2**32)
    name = name or f"{time.strftime('%Y%m%d-%H%M%S')}_{model}_{slug(prompt)[:30]}_{seed}"

    # ComfyUI's LoadImage only reads from its own input folder; a ghost source is removed again right after.
    src = (source if isinstance(source, Image.Image) else Image.open(source)).convert("RGB")
    width, height = edit_size(*src.size)
    staged = COMFY_INPUT / "morpheus" / f"{name}.png"
    staged.parent.mkdir(parents=True, exist_ok=True)
    src.save(staged)

    started = time.time()
    try:
        # only a model with LoRAs takes them; loras None = its defaults
        used = lora_keys(model, loras)
        args = {"loras": used} if has_loras(model) else {}
        img = generate(m["workflow"](prompt, f"morpheus/{name}.png", width, height, seed, steps or m["steps"], **args), ghost, progress)
    finally:
        if ghost:
            staged.unlink(missing_ok=True)

    # The gallery links back to the source when it lives under output/.
    if source_ref is None and isinstance(source, Path):
        try:
            source_ref = source.resolve().relative_to(out.resolve()).as_posix()
        except ValueError:
            source_ref = str(source)
    meta = {"image": f"{name}_raw.png", "raw": f"{name}_raw.png", "prompt": prompt, "style": "none", "model": model,
            "width": width, "height": height, "seed": seed, "seconds": round(time.time() - started, 1),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "source": source_ref}
    if args:
        meta["loras"] = used
    return keep(meta, img, img, out, name, ghost)


# ---------- Wake your dream: a picture and a motion prompt become a clip. All video renders in WSL (2026-11-14) ----------
# LTX 2.3 dev Q6 + distill LoRA runs in the WanGP engine inside WSL2 (engine/wsl_runner.py, warm worker). ComfyUI makes
# pictures only; its video graphs (Wan 2.2, LTX nodes) are gone. Glia seed: seed_2026_11_14_video_all_in_wsl.
VIDEO_DIR = OUTPUT / "videos"
VIDEO_MODELS = {
    "ltx": {"label": "LTX 2.3 dev Q6", "ltx": True},
}
VIDEO_DEFAULT = "ltx"
VIDEO_PRESETS = {"low": {"label": "Low", "note": "512 × 896 (480p) · about 2 min", "size": (512, 896)},
                 "high": {"label": "High", "note": "704 × 1280 (720p) · about 5 min", "size": (704, 1280)}}
VIDEO_FPS = 24  # LTX's own rate; at 16 props dissolve (engine/jobs.py)
VIDEO_STEPS = 11  # progress per clip: 8 main steps + WanGP's ~3 refine steps after the x2 latent upscale
VIDEO_FRAMES = VIDEO_FPS * 2 + 1  # 2 s
VIDEO_KEEP_WARM = 600  # seconds the WSL worker keeps LTX loaded after a clip (a board renders clip after clip)
VIDEO_NEGATIVE = ("vivid colors, overexposed, static, blurry details, subtitles, style, artwork, painting, still image, grey overall, "
                  "worst quality, low quality, JPEG artifacts, ugly, deformed, extra fingers, bad hands, bad face, malformed limbs, "
                  "fused fingers, motionless frame, cluttered background")  # kept for the page; LTX at CFG 1 has no negative
LTX_CKPT = "/root/wangp/ckpts/ltx-2.3-22b-dev-Q6_K.gguf"  # on the WSL ext4 disk


def ltx_frames(n: int) -> int:
    """LTX takes 8k + 1 frames: the next one at or above n."""
    return max(9, -(-(n - 1) // 8) * 8 + 1)


def engine_size(width: int, height: int) -> tuple[int, int]:
    """LTX renders multiples of 64; the frames are scaled to the asked size afterwards."""
    return max(64, round(width / 64) * 64), max(64, round(height / 64) * 64)


def video_clip(start, prompt: str, width: int, height: int, seed: int, frames: int = VIDEO_FRAMES,
               end=None, stopped=lambda: False, progress=None, stats: dict | None = None) -> list:
    """One LTX clip in the WSL engine: start picture (+ end picture for a bridge) -> its frames as PIL images at width x height.
    Call with the server's GPU lock held. ComfyUI is stopped first: the engine and ComfyUI never share the card."""
    from engine import jobs as ejobs, wsl_runner
    stop_comfy()
    w, h = engine_size(width, height)
    tmp = wsl_runner.JOBS_DIR / f"in_{time.time_ns():x}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        start.convert("RGB").resize((w, h), Image.LANCZOS).save(tmp / "start.png")
        s = {"model_type": ejobs.MODELS["ltx23"], "seed": int(seed) % 2**31, "resolution": f"{w}x{h}",
             "video_length": ltx_frames(frames), "force_fps": VIDEO_FPS, "multi_prompts_gen_type": "FG",
             "prompt": prompt, "image_prompt_type": "SE" if end is not None else "S", "image_start": str(tmp / "start.png"),
             **ejobs.ltx_steps(8)}
        if end is not None:
            end.convert("RGB").resize((w, h), Image.LANCZOS).save(tmp / "end.png")
            s["image_end"] = str(tmp / "end.png")
        seen = {"offset": 0, "key": None}

        def on_event(ev):
            if not progress or ev.get("phase") != "inference" or not ev.get("total"):
                return
            if ev["total"] != seen["key"] and seen["key"] is not None:
                seen["offset"] += seen["key"]  # the refine pass counts on from the main pass
            seen["key"] = ev["total"]
            progress(min(seen["offset"] + (ev.get("step") or 0), VIDEO_STEPS), VIDEO_STEPS)

        wsl_runner.wait_for_ram(stopped)
        out = wsl_runner.run({"kind": "render", "settings": s}, on_event, stopped, keep_warm=VIDEO_KEEP_WARM)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    made = Path(out["file"])
    try:
        clip = [f.resize((width, height), Image.LANCZOS) if f.size != (width, height) else f for f in video_frames(made)]
    finally:
        made.unlink(missing_ok=True)
    if stats is not None:
        stats.update({k: out[k] for k in ("vram_peak_gb", "ram_peak_gb") if k in out}, engine="wsl")
    return clip


# Motion: from the Wan 2.2 days; boards still store and show it, LTX does not use it.
VIDEO_MOTION = {"normal": {"label": "Normal", "high": 2, "shift": 5.0},
                "more": {"label": "More", "high": 3, "shift": 5.0},
                "wild": {"label": "Wild", "high": 3, "shift": 4.0}}


def shot_motion(board: dict, s: dict) -> str:
    return s.get("motion") if s.get("motion") in VIDEO_MOTION else board.get("motion", "normal")


def shot_model(board: dict, s: dict) -> str:
    """Per shot; boards from before keep their board-wide model as the fallback."""
    return s.get("model") if s.get("model") in VIDEO_MODELS else video_model(board.get("model"))


def shot_character(board: dict, s: dict) -> str:
    return s["character"] if isinstance(s.get("character"), str) else board.get("character", "")


def bridge_model(board: dict, s: dict) -> str:
    m = (s.get("transition") or {}).get("model")
    return m if m in VIDEO_MODELS else shot_model(board, s)


def video_model(m) -> str:
    return m if m in VIDEO_MODELS else VIDEO_DEFAULT


_INSTALLED = {"at": 0.0, "ok": False}


def video_installed(model: str = VIDEO_DEFAULT) -> bool:
    """LTX's checkpoint on the WSL disk (checked at most once a minute: wsl.exe takes a moment)."""
    import subprocess
    if time.time() - _INSTALLED["at"] > 60:
        try:
            r = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "-u", "root", "--", "test", "-f", LTX_CKPT], timeout=60,
                               capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            _INSTALLED["ok"] = r.returncode == 0
        except (OSError, subprocess.SubprocessError):
            _INSTALLED["ok"] = False
        _INSTALLED["at"] = time.time()
    return _INSTALLED["ok"]


def video_models() -> list[dict]:
    return [{"id": k, "label": m["label"]} for k, m in VIDEO_MODELS.items() if video_installed(k)]


def stop_comfy() -> None:
    """Kill whatever listens on ComfyUI's port and wait until it is gone. /interrupt only acts between steps."""
    import psutil
    for c in psutil.net_connections(kind="tcp"):
        if c.laddr and c.laddr.port == 8188 and c.status == psutil.CONN_LISTEN and c.pid:
            psutil.Process(c.pid).kill()
    for _ in range(40):
        if not comfy_up():
            return
        time.sleep(0.25)


def video_frames(path: Path) -> list:
    import av
    with av.open(str(path)) as c:
        return [f.to_image() for f in c.decode(video=0)]


def video_write(frames: list, dest: Path) -> None:
    """Encode to a temp name first, so a clip on disk is always a whole one."""
    import av
    tmp = dest.with_suffix(".tmp.mp4")
    with av.open(str(tmp), "w") as c:
        v = c.add_stream("libx264", rate=VIDEO_FPS)
        v.width, v.height, v.pix_fmt = frames[0].width, frames[0].height, "yuv420p"
        v.options = {"crf": "17"}
        for f in frames:
            c.mux(v.encode(av.VideoFrame.from_image(f).reformat(format="yuv420p")))
        c.mux(v.encode())
    tmp.replace(dest)


def run_video(source: Path, source_name: str, sets: list[dict], preset: str, length: int = 2, stopped=lambda: False,
              progress=None, on_part=None) -> dict:
    """Render a clip into output/videos in 2 s parts: each part starts on the last frame of the one before and takes its
    own prompt set from sets (the last set repeats if the clip is longer). The clip and
    its sidecar are rewritten after every part, so stopping keeps what is done. Each part renders in the WSL engine (video_clip)."""
    width, height = VIDEO_PRESETS[preset]["size"]
    with Image.open(source) as im:
        if im.width > im.height:
            width, height = height, width  # a landscape picture stays landscape
        start = im.convert("RGB")
    seed = random.randrange(2**31 - 16)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}_wake_{preset}_{slug(sets[0]['prompt'])}_{seed}"
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    began, parts, frames, meta = time.time(), length // 2, [], {}
    for n in range(parts):
        if on_part:
            on_part(n + 1, parts)
        use = sets[min(n, len(sets) - 1)]
        part = video_clip(start, use["prompt"], width, height, seed + n, stopped=stopped, progress=progress)
        frames += part if n == 0 else part[1:]  # the first frame repeats the last one of the part before
        start = part[-1]
        video_write(frames, VIDEO_DIR / f"{name}.mp4")
        frames[len(frames) // 2].resize((160, round(160 * height / width))).save(VIDEO_DIR / f"{name}_thumb.jpg", quality=85)
        meta = {"video": f"videos/{name}.mp4", "thumb": f"videos/{name}_thumb.jpg", "source": source_name, "prompt": sets[0]["prompt"], "sets": sets, "preset": preset,
                "width": width, "height": height, "seed": seed, "length": (n + 1) * 2, "seconds": round(time.time() - began),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (VIDEO_DIR / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


# ---------- Storyboard: a scene of 2 s shots, kept as JSON in output/boards/<id>/board.json next to its renders ----------
BOARD_DIR = OUTPUT / "boards"
BOARD_LOCK = threading.RLock()
SHOT_FIELDS = ("id", "label", "start", "source", "prompt", "negative", "seed", "transition", "length", "motion", "tempo", "model", "character")
SHOT_TEMPOS = (1, 1.25, 1.5)  # up-casting: render more frames, play them in the shot's length (against slow motion)
RENDER_FIELDS = ("clip", "last", "thumb", "seconds", "done", "bridge", "bridge_seconds", "bridge_done", "stats")  # what only a render writes
SHOT_LENGTHS = (2, 3, 4, 5)  # seconds; LTX takes 8n + 1 frames
TRANSITIONS = ("cut", "fade", "flash", "bridge")
FLASH_COLORS = {"violet": (150, 70, 255), "pink": (255, 60, 200), "cyan": (60, 230, 255), "white": (255, 255, 255), "black": (0, 0, 0)}


def transition_of(raw) -> dict:
    """How a shot that starts on a new picture is reached from the shot before."""
    t = raw if isinstance(raw, dict) else {}
    try:
        seconds = min(2.0, max(0.1, float(t.get("seconds", 0.5))))
    except (TypeError, ValueError):
        seconds = 0.5
    return {"type": t.get("type") if t.get("type") in TRANSITIONS else "cut", "seconds": seconds,
            "color": t.get("color") if t.get("color") in FLASH_COLORS else "violet", "prompt": str(t.get("prompt") or ""),
            "seed": t["seed"] if isinstance(t.get("seed"), int) else random.randrange(2**32),
            "model": t.get("model") if t.get("model") in VIDEO_MODELS else ""}  # "" = same as the shot it leads into


def has_bridge(board: dict, i: int) -> bool:
    s = board["shots"][i]
    return i > 0 and s["start"] == "picture" and s.get("transition", {}).get("type") == "bridge"


def bridge_sigs(board: dict) -> dict:
    """Fingerprint per bridge: both shots it joins, its prompt and seed. Keyed by the index of the shot it leads into."""
    sigs, out = board_sigs(board), {}
    for i, s in enumerate(board["shots"]):
        if has_bridge(board, i):
            t = s["transition"]
            # an own bridge model joins the fingerprint only when set, so older bridges keep theirs
            out[i] = hashlib.sha1(json.dumps([sigs[i - 1], sigs[i], t["prompt"], t["seed"]] + ([t["model"]] if t.get("model") else []))
                                  .encode("utf-8")).hexdigest()[:12]
    return out


def scene_sig(board: dict) -> str:
    """All that shapes the joined scene: a changed fade only needs a new join, no render."""
    cuts = [s.get("transition") if s["start"] == "picture" else None for s in board["shots"]]
    return hashlib.sha1(json.dumps([board_sigs(board), bridge_sigs(board), cuts]).encode("utf-8")).hexdigest()[:12]


def board_file(board_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-z]{1,40}", str(board_id)):
        raise ValueError("This board does not exist.")
    return BOARD_DIR / board_id / "board.json"


def board_load(board_id: str) -> dict:
    try:
        return json.loads(board_file(board_id).read_text(encoding="utf-8"))
    except OSError:
        raise ValueError("This board does not exist.")


def board_write(board: dict) -> None:
    """Through a temp name, so a board on disk is always a whole one."""
    path = board_file(board["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    board["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def shot_current(s: dict, sig: str) -> bool:
    """Rendered for these settings, or locked: a locked shot keeps its clip whatever changes around it."""
    return s.get("done") == sig or bool(s.get("locked") and s.get("clip"))


def board_sigs(board: dict) -> list[str]:
    """A fingerprint per shot of all that shapes its render; a continuing shot inherits the one before, so a change runs down the chain."""
    sigs, before = [], ""
    for i, s in enumerate(board["shots"]):
        chained = i > 0 and s["start"] == "continue"
        length = s.get("length", 2)  # a 2 s shot keeps the fingerprint it had before lengths existed
        model = shot_model(board, s)
        text = json.dumps([shot_character(board, s), s["prompt"], s["negative"], s["seed"], before if chained else s["source"]]
                          + ([length] if length != 2 else []) + ([model] if model != "q4" else [])  # fingerprint as before: old q4 shots now show as changed (true: new model)
                          + ([shot_motion(board, s)] if shot_motion(board, s) != "normal" else [])
                          + ([s["tempo"]] if s.get("tempo", 1) != 1 else []))
        before = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        sigs.append(before)
    return sigs


def board_view(board: dict) -> dict:
    """The board plus what is true right now: each shot's status, and whether the joined scene still matches."""
    view, sigs = json.loads(json.dumps(board)), board_sigs(board)
    for s, sig in zip(view["shots"], sigs):
        s["status"] = "waiting" if not s.get("clip") else "rendered" if shot_current(s, sig) else "outdated"
    for i, sig in bridge_sigs(board).items():
        s = view["shots"][i]
        s["bridge_status"] = "waiting" if not s.get("bridge") else "rendered" if s.get("bridge_done") == sig else "outdated"
    if view.get("scene") and view["scene"].get("sig") != scene_sig(board):
        view["scene"]["outdated"] = True
    return view


def board_list() -> list[dict]:
    boards = []
    BOARD_DIR.mkdir(parents=True, exist_ok=True)
    for f in BOARD_DIR.glob("*/board.json"):
        try:
            b = json.loads(f.read_text(encoding="utf-8"))
            first = b["shots"][0] if b["shots"] else {}
            scene = b.get("scene")
            if scene:
                scene = {**scene, "outdated": scene.get("sig") != scene_sig(b)}
            boards.append({"id": b["id"], "title": b["title"], "shots": len(b["shots"]), "updated": b.get("updated", ""),
                           "created": b.get("created", ""), "starred": bool(b.get("starred")), "scene": scene,
                           "thumb": (scene or {}).get("thumb") or first.get("thumb") or first.get("source")})
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    return sorted(boards, key=lambda b: b["updated"], reverse=True)


def board_store(req: dict) -> dict:
    """Save the page's edits. Render results stay the server's; a shot that left the board takes its files along."""
    with BOARD_LOCK:
        board_id = str(req.get("id") or f"{time.time_ns():x}")
        old = board_load(board_id) if board_file(board_id).is_file() else {"id": board_id, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "shots": []}
        before = {s["id"]: s for s in old["shots"]}
        shots = []
        for raw in req.get("shots") or []:
            s = {k: raw.get(k) for k in SHOT_FIELDS}
            if not re.fullmatch(r"[0-9a-z]{1,40}", str(s["id"])):
                raise ValueError("A shot has a broken id.")
            s.update(label=str(s["label"] or "").strip(), prompt=str(s["prompt"] or ""), negative=str(s["negative"] or ""),
                     source=s["source"] or None, start="continue" if s["start"] == "continue" else "picture")
            s["seed"] = s["seed"] if isinstance(s["seed"], int) else random.randrange(2**32)
            s["transition"] = transition_of(s["transition"])
            s["length"] = s["length"] if s["length"] in SHOT_LENGTHS else 2
            s["motion"] = s["motion"] if s["motion"] in VIDEO_MOTION else old.get("motion", "normal")
            s["tempo"] = s["tempo"] if s["tempo"] in SHOT_TEMPOS else 1
            s["model"] = s["model"] if s["model"] in VIDEO_MODELS else video_model(old.get("model"))
            s["character"] = str(s["character"]) if s["character"] is not None else old.get("character", "")
            s["locked"] = bool(raw["locked"]) if "locked" in raw else bool(before.get(s["id"], {}).get("locked"))
            s.update({k: before[s["id"]][k] for k in RENDER_FIELDS if k in before.get(s["id"], {})})
            shots.append(s)
        for sid in set(before) - {s["id"] for s in shots}:
            for f in (before[sid].get("clip"), before[sid].get("last"), before[sid].get("thumb"), before[sid].get("bridge")):
                if f:
                    (OUTPUT / f).unlink(missing_ok=True)
        # model and character live on the shots now; the board-wide ones stay only as the fallback for older boards
        old.update(title=str(req.get("title") or "Untitled board").strip()[:80], shots=shots)
        board_write(old)
        return board_view(old)


def board_star(board_id: str, starred: bool) -> None:
    """A favourite flag only: it does not count as an edit, so the board keeps its place in "newest first"."""
    with BOARD_LOCK:
        board = board_load(board_id)
        board["starred"] = bool(starred)
        path = board_file(board_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


def board_delete(board_id: str) -> None:
    with BOARD_LOCK:
        folder = board_file(board_id).parent
        if folder.is_dir():
            shutil.rmtree(folder)


def board_size(board: dict) -> tuple[int, int]:
    """Low only; a landscape first picture makes the whole board landscape."""
    width, height = VIDEO_PRESETS["low"]["size"]
    first = board["shots"][0] if board["shots"] else {}
    if first.get("source") and (OUTPUT / first["source"]).is_file():
        with Image.open(OUTPUT / first["source"]) as im:
            if im.width > im.height:
                return height, width
    return width, height


def board_check(board: dict, upto: int, first: int = 0) -> None:
    for i, s in list(enumerate(board["shots"]))[first:upto + 1]:
        name = f"Shot {i + 1}"
        if not s["prompt"].strip():
            raise ValueError(f"{name} needs a motion prompt.")
        if i == 0 or s["start"] == "picture":
            path = (OUTPUT / (s["source"] or "")).resolve()
            if OUTPUT.resolve() not in path.parents or not path.is_file():
                raise ValueError(f"{name} needs a picture.")


def board_bridge(board: dict, i: int, sig: str, stopped, progress) -> None:
    """A new clip from the last frame of shot i-1 to the first frame of shot i."""
    s, before, folder = board["shots"][i], board["shots"][i - 1], board_file(board["id"]).parent
    width, height = board_size(board)
    with Image.open(OUTPUT / before["last"]) as im:
        first = im.convert("RGB").resize((width, height))
    last = video_frames(OUTPUT / s["clip"])[0].resize((width, height))
    motion = s["transition"]["prompt"].strip() or "One smooth continuous camera move from the first picture to the second."
    prompt = ". ".join(x.strip().rstrip(".") for x in (shot_character(board, s), motion) if x.strip()) + "."
    began = time.time()
    frames = video_clip(first, prompt, width, height, s["transition"]["seed"], end=last, stopped=stopped, progress=progress)
    video_write(frames, folder / f"{s['id']}_bridge.mp4")
    with BOARD_LOCK:
        now = board_load(board["id"])
        for x in now["shots"]:
            if x["id"] == s["id"]:
                x.update(bridge=f"boards/{board['id']}/{s['id']}_bridge.mp4", bridge_seconds=round(time.time() - began), bridge_done=sig)
        board_write(now)


def board_shot(board: dict, i: int, sig: str, stopped, progress) -> None:
    """Render one shot into the board's folder and record it on the newest board, which may have been edited meanwhile."""
    from PIL import ImageOps
    s, folder = board["shots"][i], board_file(board["id"]).parent
    width, height = board_size(board)
    chained = i > 0 and s["start"] == "continue"
    with Image.open(OUTPUT / (board["shots"][i - 1]["last"] if chained else s["source"])) as im:
        start = ImageOps.fit(im.convert("RGB"), (width, height), Image.LANCZOS)
    prompt = ". ".join(x.strip().rstrip(".") for x in (shot_character(board, s), s["prompt"]) if x.strip()) + "."
    began = time.time()
    tempo = s.get("tempo", 1) if s.get("tempo", 1) in SHOT_TEMPOS else 1
    keep_n = s.get("length", 2) * VIDEO_FPS + 1  # what the shot plays
    frames_n = ltx_frames(round(s.get("length", 2) * VIDEO_FPS * tempo) + 1)  # what LTX renders: 8n + 1
    stats = {"model": shot_model(board, s), "tempo": tempo, "frames": frames_n, "size": f"{width}x{height}"}
    frames = video_clip(start, prompt, width, height, s["seed"], frames=frames_n, stopped=stopped, progress=progress, stats=stats)
    if len(frames) > keep_n:  # faster tempo: spread the shot's frames evenly over the render, first and last kept
        frames = [frames[round(i * (len(frames) - 1) / (keep_n - 1))] for i in range(keep_n)]
    video_write(frames, folder / f"{s['id']}.mp4")
    frames[-1].save(folder / f"{s['id']}_last.png")
    frames[len(frames) // 2].resize((160, round(160 * height / width))).save(folder / f"{s['id']}_thumb.jpg", quality=85)
    base = f"boards/{board['id']}/{s['id']}"
    with BOARD_LOCK:
        now = board_load(board["id"])
        for x in now["shots"]:
            if x["id"] == s["id"]:
                x.update(clip=f"{base}.mp4", last=f"{base}_last.png", thumb=f"{base}_thumb.jpg", seconds=round(time.time() - began), done=sig, locked=True,
                         stats=stats | {"seconds": round(time.time() - began)})
        board_write(now)
    with open(folder / "stats.jsonl", "a", encoding="utf-8") as f:  # every render, also the ones redone later
        f.write(json.dumps({"shot": s["id"], "label": s.get("label"), "length": s.get("length", 2), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                           | stats | {"seconds": round(time.time() - began)}) + "\n")


def board_join(board_id: str) -> None:
    """All shots and bridges current: join them with their transitions. A fade overlaps the shots; a flash passes through a colour."""
    board = board_load(board_id)
    sigs, bridges = board_sigs(board), bridge_sigs(board)
    if not board["shots"] or any(not shot_current(s, sig) for s, sig in zip(board["shots"], sigs)):
        return
    if any(board["shots"][i].get("bridge_done") != sig for i, sig in bridges.items()):
        return
    frames = []
    for i, s in enumerate(board["shots"]):
        part = video_frames(OUTPUT / s["clip"])
        t = s["transition"] if i and s["start"] == "picture" else None
        if i and not t:
            frames += part[1:]  # a continuing shot's first frame repeats the last one before it
        elif not t or t["type"] == "cut":
            frames += part
        elif t["type"] == "bridge":
            frames += video_frames(OUTPUT / s["bridge"])[1:-1] + part  # its ends are the two neighbouring frames
        elif t["type"] == "fade":
            n = min(round(t["seconds"] * VIDEO_FPS), len(frames) - 1, len(part) - 1)
            for k in range(n):
                frames[k - n] = Image.blend(frames[k - n], part[k], (k + 1) / (n + 1))
            frames += part[n:]
        else:
            n = min(max(1, round(t["seconds"] * VIDEO_FPS / 2)), len(frames) - 1, len(part) - 1)
            color = Image.new("RGB", part[0].size, FLASH_COLORS[t["color"]])
            for k in range(n):
                frames[k - n] = Image.blend(frames[k - n], color, (k + 1) / n)
                part[k] = Image.blend(color, part[k], (k + 1) / (n + 1))
            frames += part
    video_write(frames, board_file(board_id).parent / "scene.mp4")
    with BOARD_LOCK:
        now = board_load(board_id)
        now["scene"] = {"video": f"boards/{board_id}/scene.mp4", "thumb": now["shots"][0]["thumb"], "sig": scene_sig(board), "length": len(frames) / VIDEO_FPS,
                        "seconds": sum(s.get("seconds", 0) + s.get("bridge_seconds", 0) for s in now["shots"]), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        board_write(now)


def board_render(board_id: str, shot: str | None, stopped, progress, on_part) -> None:
    """Render what is not current, up to the given shot; with no shot given, all of it, then the bridges and the join.
    Finished work is kept, so stopping loses only the running clip."""
    def todo(board):
        sigs = board_sigs(board)
        shots = board["shots"]
        upto = next((i for i, s in enumerate(shots) if s["id"] == shot), len(shots) - 1)
        # one shot needs only its own chain: back to the nearest shot that starts on a picture, not the whole board
        first = upto if shot else 0
        while first > 0 and shots[first]["start"] == "continue":
            first -= 1
        work = [(board_shot, i, sigs[i]) for i in range(first, upto + 1) if not shot_current(shots[i], sigs[i])]
        if not shot:
            work += [(board_bridge, i, sig) for i, sig in bridge_sigs(board).items() if shots[i].get("bridge_done") != sig]
        return first, upto, work

    board = board_load(board_id)
    first, upto, work = todo(board)
    board_check(board, upto, first)
    parts, n = len(work), 0
    while True:
        board = board_load(board_id)
        first, upto, work = todo(board)
        if not work:
            break
        board_check(board, upto, first)
        n += 1
        on_part(min(n, parts), parts)
        render, i, sig = work[0]
        render(board, i, sig, stopped, progress)
    if not shot:
        board_join(board_id)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--style", choices=STYLES, default="none")
    ap.add_argument("--model", choices=[*MODELS, *EDIT_MODELS], default=None,
                    help="default: z-image-turbo, or flux2-klein-4b-edit with --edit")
    ap.add_argument("--edit", metavar="IMAGE", help="edit this image; the prompt is the instruction")
    ap.add_argument("--out", default=str(OUTPUT))
    ap.add_argument("--name", help="file name without extension (default: from prompt and seed)")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--grid", type=int, help="pixel styles: grid width in pixels")
    ap.add_argument("--colors", type=int, help="pixel styles: palette size")
    a = ap.parse_args()
    if a.edit:
        meta = run_edit(a.prompt, Path(a.edit), a.model or "flux2-klein-4b-edit", a.seed, a.steps, Path(a.out), a.name)
    else:
        meta = run(a.prompt, a.style, a.model or "z-image-turbo", a.width, a.height, a.seed, a.steps, Path(a.out),
                   a.name, a.grid, a.colors)
    print(json.dumps({**meta, "image": str(Path(a.out) / meta["image"]), "raw": str(Path(a.out) / meta["raw"])}))


if __name__ == "__main__":
    main()
