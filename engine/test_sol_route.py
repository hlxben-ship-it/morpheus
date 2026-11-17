"""engine/wsl/sol_route.py: when the conditioning rows of Sol attention are routed (WanGP is stubbed, it only exists inside WSL):
python -m unittest engine.test_sol_route (Morpheus root)."""
import importlib, os, sys, types, unittest
from pathlib import Path
from unittest import mock

TAU_END = 0.8
WSL = str(Path(__file__).parent / "wsl")


class FakeSol:
    def __init__(self): self.enabled, self.sink_tokens, self.tau = True, 0, 0.0
    def begin_forward(self, layout, device, dtype, tau, grouped_video=False):
        self.sink_tokens, self.tau = 8610, float(tau)


def taus(steps, start=1.3):
    return [start + (TAU_END - start) * i / max(1, steps - 1) for i in range(steps)]


class SolRoute(unittest.TestCase):
    def load(self, route_from=None):
        sa = types.ModuleType("models.minimax_h3.sol_attention"); sa.MiniMaxH3SolAttention = type("Sol", (FakeSol,), {}); sa._graph_state = None
        pl = types.ModuleType("models.minimax_h3.pipeline"); pl.SOL_ATTN_TAU_END = TAU_END
        pkg = types.ModuleType("models.minimax_h3"); pkg.sol_attention, pkg.pipeline = sa, pl
        mods = {"models": types.ModuleType("models"), "models.minimax_h3": pkg, "models.minimax_h3.sol_attention": sa, "models.minimax_h3.pipeline": pl}
        env = {} if route_from is None else {"MORPHEUS_SOL_ROUTE_FROM": str(route_from)}
        for p in (mock.patch.dict(sys.modules, mods), mock.patch.dict(os.environ, env), mock.patch.object(sys, "path", [WSL] + sys.path)):
            p.start(); self.addCleanup(p.stop)
        if route_from is None: os.environ.pop("MORPHEUS_SOL_ROUTE_FROM", None)
        sys.modules.pop("sol_route", None); self.addCleanup(sys.modules.pop, "sol_route", None)
        importlib.import_module("sol_route")
        return sa

    def sinks(self, sa, tau_list, forwards_per_step=1):
        sol = sa.MiniMaxH3SolAttention(); out = []
        for tau in tau_list:
            for _ in range(forwards_per_step):
                sol.begin_forward(None, None, None, tau); out.append(sol.sink_tokens)
        return out

    def test_four_steps_route_the_second_half(self):
        self.assertEqual(self.sinks(self.load(), taus(4)), [8610, 8610, 0, 0])

    def test_more_steps_keep_the_first_half_exact(self):
        self.assertEqual(self.sinks(self.load(), taus(8)), [8610] * 4 + [0] * 4)

    def test_a_new_job_starts_exact_again(self):
        self.assertEqual(self.sinks(self.load(), taus(4) + taus(4)), [8610, 8610, 0, 0] * 2)

    def test_several_forwards_of_one_step_do_not_restart_the_schedule(self):
        self.assertEqual(self.sinks(self.load(), taus(4), forwards_per_step=2), [8610] * 4 + [0] * 4)

    def test_switch_off_keeps_every_step_exact(self):
        self.assertEqual(self.sinks(self.load(1.1), taus(4)), [8610] * 4)

    def test_graph_key_follows_the_routing(self):
        sa = self.load(); self.sinks(sa, taus(4))
        self.assertEqual(sa._graph_state[1], 0); self.assertAlmostEqual(sa._graph_state[0], TAU_END)

    def test_disabled_sol_is_left_alone(self):
        sa = self.load(); sol = sa.MiniMaxH3SolAttention(); sol.enabled = False
        for tau in taus(4): sol.begin_forward(None, None, None, tau)
        self.assertEqual(sol.sink_tokens, 8610)


if __name__ == "__main__":
    unittest.main()
