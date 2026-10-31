"""Morpheus: what ComfyUI holds in memory, for the web UI's header (GET /morpheus/loaded).

Copied into ComfyUI/custom_nodes by server.py; ComfyUI has no route of its own for this.
"""
from pathlib import Path

from aiohttp import web

import comfy.model_management as mm
from server import PromptServer

NODE_CLASS_MAPPINGS = {}  # a route only, no nodes


def describe(patcher) -> dict:
    init = getattr(patcher, "cached_patcher_init", None)
    path = None
    if init and len(init) > 1 and init[1]:
        first = init[1][0]
        path = first[0] if isinstance(first, (list, tuple)) and first else first  # text encoders come as a list
    kind = type(getattr(patcher, "model", None)).__name__
    return {
        "name": Path(str(path)).name if path else kind,
        "kind": kind,
        "loaded": int(patcher.loaded_size()),
        "size": int(patcher.model_size()),
    }


@PromptServer.instance.routes.get("/morpheus/loaded")
async def loaded(request):
    models = []
    for lm in list(mm.current_loaded_models):
        if lm.real_model() is None:  # being freed: holding its patcher now could outlive ComfyUI's cleanup
            continue
        patcher = lm.model
        if patcher is None:
            continue
        try:
            models.append({**describe(patcher), "device": str(lm.device)})
        except Exception as e:  # one odd model shouldn't hide the others
            models.append({"name": type(patcher).__name__, "kind": "?", "loaded": 0, "size": 0, "error": str(e)})
    return web.json_response(models)


@PromptServer.instance.routes.post("/morpheus/cleanup")
async def cleanup(request):
    """Drop entries whose model is already gone. ComfyUI's finalizer sometimes misses one, and every later load
    then fails in free_memory with "'NoneType' object has no attribute 'model_size'" until a restart."""
    before = len(mm.current_loaded_models)
    mm.cleanup_models()
    return web.json_response({"removed": before - len(mm.current_loaded_models)})
