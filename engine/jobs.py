"""Render jobs for the Morpheus video engine: a Playground snapshot or an upscale request -> the settings dict the
WanGP copy in engine/wangp/ takes. Pure Python (no torch), so server.py and the tests import it from the light venv;
engine/worker.py runs the result in the engine venv.

Every value here comes from the 2026-11 WanGP spike (Glia seed seed_2026_11_13_playground_wangp_native): what is
not measured there is not offered. Before adding anything: look how WanGP does it first (engine/PROVENANCE.md).
"""
import random
import re

FPS = 24  # both models; LTX broke at 16 (props dissolve)

# Native render sizes per Playground aspect. Multiples of 64 (LTX's VAE block) and within LTX 2.3's two-stage
# budget of 1920x1088 pixels; H3 gets the same request and WanGP fits it (1280x704 renders 1248x704).
SIZES = {
    "480p": {"16:9": (896, 512), "9:16": (512, 896), "1:1": (640, 640), "4:3": (768, 576), "3:4": (576, 768), "21:9": (1088, 448), "3:2": (768, 512), "2:3": (512, 768)},
    "720p": {"16:9": (1280, 704), "9:16": (704, 1280), "1:1": (960, 960), "4:3": (1088, 832), "3:4": (832, 1088), "21:9": (1472, 640), "3:2": (1152, 768), "2:3": (768, 1152)},
    "1080p": {"16:9": (1920, 1088), "9:16": (1088, 1920), "1:1": (1408, 1408), "4:3": (1600, 1216), "3:4": (1216, 1600), "21:9": (2176, 960), "3:2": (1728, 1152), "2:3": (1152, 1728)},
}
LTX_BUDGET = 1920 * 1088

MODELS = {"ltx23": "ltx2_22B_dev_q6",  # finetune in engine/wangp/finetunes/
          "h3_minimax": "minimax_h3_ref2va_hyperflow"}

UPSCALERS = {"relayvsr": "RelayVSR", "seedvr2": "SeedVR2 3B"}
FACTORS = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)  # RelayVSR refuses anything not x1.0-4.0 in 0.5 steps (x1.534 failed)


def frames(seconds: int) -> int:
    """LTX wants 8n+1 frames (3 s = 73, 5 s = 121, 10 s = 241)."""
    return FPS * int(seconds) + 1


H3_FRAME_OFFSET, H3_FRAME_STEP = 5, 17  # H3 frame counts are 5 + 17n (WanGP minimax_h3_handler: frames_offset / frames_steps)
H3_WANGP_MIN = 107  # WanGP's frames_minimum for H3 (4.5 s); we render below it (owner test 2026-11-13: 3 s looked right)


def h3_frames(seconds: int) -> int:
    """The H3 frame count for a duration: the next 5 + 17n at or above 24 fps · s + 1 (2 s = 56, 3 s = 73, 5 s = 124).
    Below WanGP's floor of 107 the worker lowers that floor for the job."""
    want = frames(seconds)
    floor = H3_FRAME_OFFSET + H3_FRAME_STEP
    return floor + H3_FRAME_STEP * max(0, -(-(want - floor) // H3_FRAME_STEP))


def size(resolution: str, aspect: str) -> tuple[int, int]:
    try:
        return SIZES[resolution][aspect]
    except KeyError:
        raise ValueError(f"No native size for {resolution} {aspect}. Native renders: {', '.join(SIZES)}.")


def h3_prompt(prompt: str, start: bool, end: bool) -> str:
    """H3 is trained on a structured prompt. A plain prompt gets the header and sections the spike used; a prompt that
    already is structured (it names its own sections) goes through untouched."""
    text = prompt.strip()
    if re.search(r"^\s*(integrated_multimodal_description|detailed_description|subject_definitions)\s*:", text, re.M):
        return text
    head = []
    if start:
        head.append("For the target video, at 0.00 seconds into the target video, Picture 1 (from Shot 1) is fully referenced.")
    if end:
        head.append("At the end of the target video, Picture 2 (from Shot 1) is fully referenced.")
    body = f"integrated_multimodal_description: [Shot 1] {text}\n\nnon_diegetic_music: N/A"
    return "\n\n".join(head + [body])


def ltx_steps(steps: int) -> dict:
    """G7: 8 = the proven fast path (dev Q6 + distill LoRA, WanGP's fixed 8-step schedule, then its automatic ~3 refine
    steps after the x2 latent upscale). Above 8 = WanGP's "Two-Stage Modality Guidance" dev preset with that many
    Euler steps (real CFG + STG at half size, the distill LoRA only in the refine phase): slower per step, the path
    for LoRAs that want more steps."""
    if steps == 8:
        return {"sample_solver": "distilled_8_steps", "num_inference_steps": 8, "guidance_phases": 2, "guidance_scale": 1.0}
    if not 8 < steps <= 30:
        raise ValueError("LTX steps: 8 to 30.")
    return {"sample_solver": "euler", "num_inference_steps": steps, "guidance_phases": 2, "guidance_scale": 3.0,
            "audio_guidance_scale": 7.0, "alt_guidance_scale": 3.0, "alt_scale": 0.7, "perturbation_switch": 2,
            "perturbation_layers": [28], "perturbation_start_perc": 0, "perturbation_end_perc": 100,
            "_distill_lora_phases": "0;1"}


DISTILL_LORA = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"


def render_settings(snap: dict, folder, seed: int | None = None) -> dict:
    """The WanGP settings for one Playground render. `snap` is playground.snapshot(), `folder` the playground folder
    (where its start/end pictures are). A LoRA whose file is gone stops the render here, before any model loads."""
    model = snap["model"]["id"]
    if model not in MODELS:
        raise ValueError(f"The engine does not render {snap['model'].get('label', model)}.")
    res = snap["resolution"]
    if res.get("render"):
        raise ValueError("Upscaled resolutions are gone from the render: render native, then use Upscale on the clip.")
    w, h = size(res["target"], snap["aspect"])
    gone = [l["name"] for l in snap.get("loras", []) if l.get("missing")]
    if gone:
        raise ValueError(f"LoRA file missing: {', '.join(gone)}. Pick it again or remove it.")
    start = str(folder / snap["start_frame"]) if snap.get("start_frame") else None
    end = str(folder / snap["end_frame"]) if snap.get("end_frame") else None
    s = {"model_type": MODELS[model], "seed": seed if seed is not None else random.randrange(1, 2**31),
         "resolution": f"{w}x{h}", "video_length": frames(snap["duration"]), "force_fps": FPS,
         "multi_prompts_gen_type": "FG",  # a multi-line prompt is one prompt, not one clip per line
         "image_prompt_type": ("S" if start else "") + ("E" if end else "")}
    if start:
        s["image_start"] = start
    if end:
        s["image_end"] = end
    loras = [l["path"] for l in snap.get("loras", [])]
    mults = [f"{float(l['strength']):g}" for l in snap.get("loras", [])]
    steps = snap["params"]["main.steps"]
    if model == "ltx23":
        if w * h > LTX_BUDGET:
            raise ValueError("LTX 2.3 renders at most 1920x1088 pixels.")
        s["prompt"] = snap["prompt"]
        st = ltx_steps(steps)
        phases = st.pop("_distill_lora_phases", None)
        s.update(st)
        if phases:  # the dev preset names the distill LoRA itself; at 8 steps WanGP loads it on its own
            loras, mults = [DISTILL_LORA] + loras, [phases] + mults
    else:
        if steps not in (4, 6, 8):
            raise ValueError("H3 HyperFlow steps: 4, 6 or 8.")
        s["video_length"] = h3_frames(snap["duration"])
        if s["video_length"] < H3_WANGP_MIN:
            s["_frames_minimum"] = s["video_length"]  # the worker lowers WanGP's floor for this one job
        s.update({"prompt": h3_prompt(snap["prompt"], bool(start), bool(end)), "num_inference_steps": steps,
                  "sample_solver": "euler", "guidance_scale": 1.0, "guidance_phases": 1,
                  "config": "distilled_4b,lightvae_int8"})  # distilled Qwen3-VL-4B encoder + LightVAE, both int8
        # Ref2VA wants reference images ("I"); the spike passed the start picture as the reference too, and that is
        # the proven setup. Without a start picture: no reference flag, text (+ end) only.
        if start:
            s["image_refs"], s["video_prompt_type"] = [start], "I"
        else:
            s["video_prompt_type"] = ""
    if loras:
        s["activated_loras"], s["loras_multipliers"] = loras, " ".join(mults)
    return s


def upscale_request(method: str, factor) -> str:
    """WanGP's spatial_upsampling string, e.g. relayvsr*1.5 or seedvr2*2."""
    if method not in UPSCALERS:
        raise ValueError(f"Upscaler: one of {', '.join(UPSCALERS)}.")
    try:
        factor = float(factor)
    except (TypeError, ValueError):
        raise ValueError("The factor must be a number.")
    if factor not in FACTORS:
        raise ValueError(f"Factor: one of {', '.join(f'x{f:g}' for f in FACTORS)}.")
    return f"{method}*{factor:g}"
