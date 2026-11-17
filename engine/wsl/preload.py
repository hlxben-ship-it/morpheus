"""VRAM preload per job (Glia seed h3_image_edit_speed). Pure, so it is testable on Windows.

WanGP reads its --preload at model load. The worker runs with 0 (LTX needs the VRAM; H3 video is compute-bound, preload bought
nothing there). H3-Image is different: its int8 transformer (~20 GB) is streamed from RAM every step and uses only ~2.5 GB of VRAM,
so keeping 14 GB of it on the card cut a 1-ref edit at 1664x2496 from 131 s to 92 s cold, same picture (SSIM 1.0, bench img_pre1).
"""
import os

H3_IMAGE_PRELOAD_MB = int(os.environ.get("WARM_PRELOAD_H3_IMAGE", "14000"))


def preload_for(job: dict, base: int, h3_image: int = H3_IMAGE_PRELOAD_MB) -> int:
    """The --preload (MB) the model of this job is loaded with: ``h3_image`` for an H3-Image job, ``base`` for everything else."""
    return h3_image if job.get("kind") == "image" and job.get("family") == "h3_image" else base
