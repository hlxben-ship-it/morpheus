# The production module (engine/wsl/sol_kernel.py), as the warm worker loads it.
import sys
sys.path.insert(0, "/mnt/c/dev/morpheus/engine/wsl")
import sol_kernel  # noqa: F401
print("## hook sol_kernel installed", flush=True)
