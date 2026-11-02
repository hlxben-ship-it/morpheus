"""morpheus — generate an image from a prompt via the local ComfyUI server.

Usage:
  python morpheus.py "sleeping Link in bed" --style pixel --out C:\\dev\\talk_zelda
  python morpheus.py "a cozy cabin at dusk" --style none --width 1344 --height 768
  python morpheus.py "make it night, warm light in the windows" --edit output\\cabin_raw.png

Styles add a prompt prefix and, for pixel art, a post-processing pass that snaps the
image onto a real pixel grid with a limited palette (diffusion models only fake it).
The web UI (server.py) uses the same run() as the command line.
"""
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


def sdxl_workflow(ckpt: str, sampler: str, scheduler: str, cfg: float, negative: str, prefix: str = "", clip_skip: int = 1):
    """Build a workflow function for an SDXL (or SD 1.5) checkpoint (model, text encoders and VAE in one file).

    negative is the default; a call can pass its own (Manual's Negative field). build.negative exposes the default.
    clip_skip 2 reads the text encoder's second-to-last layer, which Pony was trained with.
    """
    default, default_prefix = negative, prefix

    def build(prompt: str, width: int, height: int, seed: int, steps: int, negative: str | None = None,
              prefix: str | None = None) -> dict:
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
    """Qwen-Image-Edit-2511 (20B, Q4 GGUF) with the 4-step Lightning LoRA, after ComfyUI's image_qwen_image_edit_2511
    template. Its Qwen2.5-VL encoder reads the picture along with the instruction, which keeps who she is through
    an outfit change better than Klein, at the cost of loading 22 GB in turns.

    loras are LORAS keys (None = the defaults), chained after the model. Without a speed LoRA (Lightning) it samples
    the base model's way: 40 steps at cfg 4, about ten times slower.
    """
    loras = lora_keys("qwen-image-edit-2511", loras)
    graph, model = lora_chain({"1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_M.gguf"}}},
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
        "label": "Z-Image-Turbo",
        "note": "Fastest, Apache 2.0",
        "files": ["diffusion_models/z_image_turbo_bf16.safetensors", "text_encoders/qwen_3_4b.safetensors", "vae/ae.safetensors"],
        "steps": 8,
        "workflow": zimage_workflow,
    },
    "flux2-klein-4b": {
        "label": "FLUX.2 [klein] 4B",
        "note": "Balanced, Apache 2.0",
        "files": ["diffusion_models/flux-2-klein-4b.safetensors", "text_encoders/qwen_3_4b.safetensors", "vae/flux2-vae.safetensors"],
        "steps": 4,
        "workflow": flux2_klein_workflow,
    },
    "noobai-xl": {
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
        "label": "JANKU v5.0 (SDXL)",
        "note": "Polished modern anime, Danbooru tags, artist styles via \"by name\"",
        "files": ["checkpoints/JANKUTrainedChenkinNoobai_v50.safetensors"],
        "steps": 30,
        "manual_only": True,  # the chat doesn't offer it
        "workflow": sdxl_workflow("JANKUTrainedChenkinNoobai_v50.safetensors", "euler_ancestral", "normal", 5.0,
                                  "nsfw, worst quality, low quality, " + SDXL_NEGATIVE,
                                  prefix="masterpiece, best quality, absurdres, "),
    },
    # civitai.com/models/4384 DreamShaper 8, SD 1.5: trained at 512 px, so "pixels" renders the picked format at about
    # 0.4 MP (1024 px would double bodies and faces); DPM++ 2M Karras, cfg 7
    "dreamshaper-8": {
        "label": "DreamShaper 8 (SD 1.5)",
        "note": "All-rounder: fantasy, portraits, semi-real; small native size",
        "files": ["checkpoints/dreamshaper_8.safetensors"],
        "steps": 30,
        "pixels": 512 * 768,
        "manual_only": True,
        "workflow": sdxl_workflow("dreamshaper_8.safetensors", "dpmpp_2m", "karras", 7.0,
                                  "worst quality, low quality, " + SDXL_NEGATIVE),
    },
    # civitai.com/models/376130 Nova Anime XL IL v19.0 (Illustrious: NoobAI + Illustrious 2.0 + ChenkinNoob merge);
    # the author's Illustrious settings: Euler a, cfg 4-6, 20-30 steps, their quality tags and negative
    "nova-anime-xl": {
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
    # civitai.com/models/4468 Counterfeit V3.0, SD 1.5 anime; natural language or tags, CLIP skip 2 like the NovelAI-era
    # anime models, DPM++ 2M Karras, cfg 7; small native size like the other SD 1.5 models
    "counterfeit-v3": {
        "label": "Counterfeit V3.0 (SD 1.5)",
        "note": "Classic SD 1.5 anime, free composition; small native size",
        "files": ["checkpoints/counterfeitV30_v30_pruned_fp16.safetensors"],
        "steps": 30,
        "pixels": 512 * 768,
        "manual_only": True,
        "workflow": sdxl_workflow("counterfeitV30_v30_pruned_fp16.safetensors", "dpmpp_2m", "karras", 7.0,
                                  "(worst quality, low quality:1.4), " + SDXL_NEGATIVE,
                                  prefix="masterpiece, best quality, ", clip_skip=2),
    },
    # civitai.com/models/15003 CyberRealistic Final, SD 1.5 photoreal; author: DPM++ 2M Karras, 30 steps, cfg 5, 512x768
    "cyberrealistic": {
        "label": "CyberRealistic Final (SD 1.5)",
        "note": "Photoreal people and portraits; small native size",
        "files": ["checkpoints/cyberrealistic_final_pruned_fp16.safetensors"],
        "steps": 30,
        "pixels": 512 * 768,
        "manual_only": True,
        "workflow": sdxl_workflow("cyberrealistic_final_pruned_fp16.safetensors", "dpmpp_2m", "karras", 5.0,
                                  "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, "
                                  "cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, "
                                  "username, blurry"),
    },
    # civitai.com/models/372465 Pony Realism v2.2 (Pony finetune, photoreal); author: DPM++ SDE / Euler a, 30+ steps,
    # cfg 6-7, CLIP skip 2, score_9/8_up/7_up before, score_4/5/6 in the negative, "female"/"male" over "woman"/"man"
    "pony-realism": {
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


# LoRAs under ComfyUI/models/loras, each for one model. Manual shows a checkbox per installed LoRA of the picked
# model; "default" ones are on unless the request says otherwise (chat, Soulforge, CLI). "speed" marks a step
# distillation: the workflow drops to few steps only while one of those is on.
LORAS = {
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
    # fal/Qwen-Image-Edit-2511-SFW"
    "qwen-sfw": {
        "label": "SFW",
        "note": "Generate SWF Images",
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
    "flux2-klein-4b-edit": {
        "label": "FLUX.2 [klein] 4B",
        "note": "Fast, leaves the rest of the picture alone",
        "files": MODELS["flux2-klein-4b"]["files"],
        "steps": 4,
        "workflow": flux2_klein_edit_workflow,
    },
    "qwen-image-edit-2511": {
        "label": "Qwen-Image-Edit 2511",
        "note": "Best at outfit and detail changes, keeps the person and style; slower",
        "files": ["unet/qwen-image-edit-2511-Q4_K_M.gguf",
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
