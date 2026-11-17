# Let H3-Image take attention_mode='sol' (its model def lists no custom attention; the transformer is the H3 one, Sol plumbing is there).
for _mt in ("minimax_h3_image_pruned", "minimax_h3_image"):
    try: wgp.get_model_def(_mt).setdefault("custom_attention_modes", {})["sol"] = {"label": "Sol sparse attention", "supports_sparsity": True}
    except Exception as e: print("## h3img_sol", _mt, e)
