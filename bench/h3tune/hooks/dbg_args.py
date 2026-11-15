# prints what the session really runs with
a = wgp.args
print("## ARGS preload", a.preload, "profile", a.profile, "attention", a.attention, "perc", a.perc_reserved_mem_max,
      "ram_alloc", getattr(a, "ram_allocator", None), "vram_alloc", getattr(a, "vram_allocator", None), flush=True)
print("## CFG", {k: wgp.server_config.get(k) for k in ("video_preload_mode", "video_preload_in_VRAM", "video_profile", "profile", "int8_kernels", "kernel_precision", "boost", "attention_head_split", "vae_config", "compile")}, flush=True)
