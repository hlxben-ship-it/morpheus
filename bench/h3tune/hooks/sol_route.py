# The production module (engine/wsl/sol_route.py), as the warm worker loads it.
import sys
sys.path.insert(0, "/mnt/c/dev/morpheus/engine/wsl")
import sol_route  # noqa: F401
print("## hook sol_route: ROUTE_FROM", sol_route.ROUTE_FROM, flush=True)
