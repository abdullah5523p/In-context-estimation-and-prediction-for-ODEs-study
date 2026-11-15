"""Check the shared run list and routing without launching production runs."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from ode_paper import cli
from ode_paper.config import grid as original_grid
from ode_paper.runs import SEEDS, configurations


class RunTests(unittest.TestCase):
    def test_grid_and_seed_symmetry(self):
        first, second = [configurations(seed) for seed in SEEDS]
        self.assertEqual(len(first), 52)
        self.assertEqual(len({(c.family, c.task, c.M, c.n) for c in first}), 52)
        self.assertEqual(first[:46], original_grid())
        self.assertEqual(sum(c.family == "forced_linear" for c in first), 40)
        for a, b in zip(first, second):
            self.assertEqual(b.to_dict(), dict(a.to_dict(), seed=SEEDS[1], data_seed=SEEDS[1]))
        for family in ("logistic", "birth_death", "stuart_landau"):
            self.assertEqual({(c.task, c.n, c.M) for c in first if c.family == family},
                             {(t, n, 2**19) for t in ("parameter", "prediction") for n in (8, 128)})

    def invoke(self, *args):
        with patch.object(sys, "argv", ["ode-paper", *args]), contextlib.redirect_stdout(io.StringIO()):
            cli.main()

    def test_both_seeds_train_and_evaluate_same_runs(self):
        expected = [(Path("first"), c) for c in configurations(SEEDS[0])]
        expected += [(Path("second"), c) for c in configurations(SEEDS[1])]
        for command, target in (("train", "ode_paper.train.train"), ("evaluate", "ode_paper.evaluate.evaluate")):
            with patch(target) as run:
                self.invoke(command, "--all", "--seed", "20260907", "--root", "second", "--previous-root", "first")
                self.assertEqual([call.args[:2] for call in run.call_args_list], expected)

    def test_each_seed_and_plots_use_full_grid(self):
        for seed in SEEDS:
            with patch("ode_paper.train.train") as run:
                self.invoke("train", "--all", "--seed", str(seed), "--root", "results")
                self.assertEqual([call.args[1] for call in run.call_args_list], configurations(seed))
            with patch("ode_paper.plots.plot") as plot:
                self.invoke("plot", "--seed", str(seed), "--root", "results")
                self.assertEqual(plot.call_args.kwargs["sources"], [(Path("results"), configurations(seed))])


if __name__ == "__main__":
    unittest.main()
