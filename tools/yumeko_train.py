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
sys.argv = ["sdxl_train_network.py"] + sys.argv[1:] + """--caption_extension .txt --resolution 832,1216 --network_module networks.lora
 --network_dim 32 --network_alpha 16 --network_train_unet_only --learning_rate 0.0001 --optimizer_type AdamW --lr_scheduler constant
 --train_batch_size 1 --save_model_as safetensors --mixed_precision fp16 --save_precision fp16 --sdpa --gradient_checkpointing
 --cache_latents --cache_text_encoder_outputs --max_data_loader_n_workers 0 --seed 42""".split()
runpy.run_path(str(SCRIPTS / "sdxl_train_network.py"), run_name="__main__")
