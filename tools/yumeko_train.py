"""Yumeko's training worker: sd-scripts' SDXL LoRA trainer with the settings proven on the RX 7900 XT.

Run with .venv-train from tools/sd-scripts. The ROCm Windows torch has no torch.distributed backend: accelerate only needs
its dtensor check answered no, and its cleanup after the last save skipped. Extra arguments override nothing; they add to these.
"""
import runpy
import sys
from pathlib import Path

import accelerate.accelerator
import accelerate.state
import accelerate.utils.other

accelerate.accelerator.model_has_dtensor = accelerate.utils.other.model_has_dtensor = lambda model: False
accelerate.state.PartialState.destroy_process_group = lambda self, group=None: None

SCRIPTS = Path(__file__).parent / "sd-scripts"
sys.path.insert(0, str(SCRIPTS))
# --text-encoders: the recipe proven on Nova2 v5. Short captions, the text encoders learn the trigger words, captions shuffle
# behind the first two tags (the trigger and, on signature pictures, the outfit trigger).
args = sys.argv[1:]
te = "--text-encoders" in args
args = [a for a in args if a != "--text-encoders"]
# --lr UNET,TE overrides the v2 recipe's learning rates (A/B tests); the default stays the Nova2 v5 recipe.
unet_lr, te_lr = "0.0001", "0.00002"
if "--lr" in args:
    i = args.index("--lr")
    unet_lr, te_lr = args[i + 1].split(",")
    del args[i:i + 2]
# --save_state keeps the newest full training state (weights, optimizer, step, RNG) beside the stages, so Pause can continue it
# with --resume; skip_until_initial_step makes the resumed run continue at the saved step.
state = "--save_state --save_last_n_steps_state 1" + (" --skip_until_initial_step" if "--resume" in args else "")
learn = (f"--unet_lr {unet_lr} --text_encoder_lr {te_lr} --shuffle_caption --keep_tokens 2" if te
         else "--network_train_unet_only --learning_rate 0.0001 --cache_text_encoder_outputs")
sys.argv = ["sdxl_train_network.py"] + args + f"""--caption_extension .txt --resolution 832,1216 --network_module networks.lora
 --network_dim 32 --network_alpha 16 {learn} --optimizer_type AdamW --lr_scheduler constant
 --train_batch_size 1 --save_model_as safetensors --mixed_precision fp16 --save_precision fp16 --sdpa --gradient_checkpointing
 --cache_latents --max_data_loader_n_workers 0 --seed 42 {state}""".split()
runpy.run_path(str(SCRIPTS / "sdxl_train_network.py"), run_name="__main__")
