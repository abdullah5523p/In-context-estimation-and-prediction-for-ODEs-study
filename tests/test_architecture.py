import json
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from ode_paper.architecture import Config, Transformer, baseline, configurations, loss_fn
from ode_paper.model import Transformer as OriginalTransformer
from ode_paper.data import generate
from ode_paper import architecture_train as training, architecture_evaluate as evaluation
from ode_paper import train as original_training
from ode_paper.utils import file_hash

class ArchitectureTests(unittest.TestCase):
    def test_cli_and_plots(self):
        from ode_paper import architecture_cli as cli
        from ode_paper.config import fingerprint
        with patch("sys.argv", ["screen", "train"]), patch.object(training, "train") as run:
            cli.main()
            self.assertEqual(run.call_count, 18)
            self.assertEqual({call.args[1].seed for call in run.call_args_list}, {20260906, 20260907})
        # Synthetic metric fixtures exercise plot routing/output without production runs.
        with tempfile.TemporaryDirectory() as tmp:
            roots = {s: Path(tmp)/str(s) for s in (20260906, 20260907)}
            for seed, root in roots.items():
                for name, cfg, trainer, evaluator in cli.entries(seed):
                    folder = trainer.run_path(root, cfg)
                    folder.mkdir(parents=True)
                    (folder/"best.pt").write_bytes(b"plot-test-fixture")
                    identity = evaluator.evaluation_identity(cfg, file_hash(folder/"best.pt"), 128)
                    cache = folder/"inference"/f"n128_{fingerprint(identity)}"
                    cache.mkdir(parents=True)
                    (cache/"metrics.json").write_text(json.dumps(dict(identity=identity,
                        best_epoch=1, metrics=dict(all_token_mse=.02, mean_output_mse=.01))))
                    np.savez(cache/"per_example.npz",
                             all_token_coordinate_mse=np.full((cfg.test_size, 3), .02),
                             mean_prediction=np.full((cfg.test_size, 3), .1),
                             targets=np.zeros((cfg.test_size, 3)))
                    (folder/"manifest.json").write_text(json.dumps(dict(parameters=123)))
                    keys = ("train_all_token_mse", "validation_all_token_mse") if name == "baseline" else (
                        "train_objective_mse", "validation_objective_mse")
                    (folder/"history.json").write_text(json.dumps([dict(epoch=1, **{k:.02 for k in keys})]))
                    for n in (2, 512):
                        identity_n = evaluator.evaluation_identity(cfg, file_hash(folder/"best.pt"), n)
                        cache_n = folder/"inference"/f"n{n}_{fingerprint(identity_n)}"
                        cache_n.mkdir(parents=True)
                        record = json.loads((cache/"metrics.json").read_text())
                        record["identity"] = identity_n
                        (cache_n/"metrics.json").write_text(json.dumps(record))
                        (cache_n/"per_example.npz").write_bytes((cache/"per_example.npz").read_bytes())
            cli.plot(roots, list(roots), ns=(2, 128, 512))
            output = roots[20260907]/"architecture_screen"/"figures"
            self.assertEqual(len(list(output.glob("*.pdf"))), 10)
            self.assertEqual(len(list(output.glob("*learning.png"))), 20)
            self.assertEqual(len((output/"summary.csv").read_text().splitlines()), 61)

    def test_evaluation_default_context_grid(self):
        from ode_paper import architecture_cli as cli
        from ode_paper.config import INFERENCE_NS
        with patch("sys.argv", ["screen", "evaluate"]), patch.object(Path, "exists", return_value=True), \
             patch.object(cli.base_eval, "evaluate") as base_eval, \
             patch.object(cli.evaluation, "evaluate") as screen_eval:
            cli.main()
            self.assertEqual(base_eval.call_count, 2)
            self.assertEqual(screen_eval.call_count, 18)
            for call in base_eval.call_args_list + screen_eval.call_args_list:
                self.assertEqual(tuple(call.kwargs["ns"]), INFERENCE_NS)

    def test_grid(self):
        for seed in (20260906, 20260907):
            configs = configurations(seed)
            self.assertEqual(len(configs), 9)
            self.assertEqual(len({c.variant for c in configs}), 9)
            for cfg in configs:
                self.assertEqual((cfg.seed, cfg.data_seed, cfg.n, cfg.M, cfg.epochs),
                                 (seed, seed, 128, 2**19, 30))
                changed = {k for k, v in baseline(seed).to_dict().items() if cfg.to_dict()[k] != v}
                expected = {"depth"} if cfg.variant.startswith("depth") else (
                    {"width", "ff_width"} if cfg.variant.startswith("width") else (
                    {"heads"} if cfg.variant.startswith("heads") else set()))
                self.assertEqual(changed, expected)
                self.assertEqual(cfg.loss, "mean_output" if cfg.variant == "mean_output" else "all_token")
                self.assertEqual(cfg.activation, "relu" if cfg.variant == "relu" else "squared_relu")

    def test_model_and_loss(self):
        cfg = replace(configurations(20260906)[0], depth=1, width=4, ff_width=8)
        torch.manual_seed(12); original = OriginalTransformer(cfg)
        torch.manual_seed(12); model = Transformer(cfg)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        original.eval(); model.eval()
        x = torch.randn(2, 8, 2)
        torch.testing.assert_close(original(x), model(x), rtol=0, atol=0)
        relu = Transformer(replace(cfg, activation="relu"))
        relu.load_state_dict(model.state_dict()); relu.eval()
        self.assertFalse(torch.allclose(model(x), relu(x)))
        outputs = torch.tensor([[[1., 3.], [-1., -3.]]], requires_grad=True)
        targets = torch.zeros(1, 2)
        self.assertEqual(loss_fn(outputs, targets, cfg).item(), 0.)
        all_cfg = replace(cfg, loss="all_token")
        self.assertEqual(loss_fn(outputs, targets, all_cfg).item(), 5.)
        loss_fn(outputs, targets, all_cfg).backward()
        self.assertIsNotNone(outputs.grad)
        a = generate(cfg, "train", 0, 3)
        b = generate(baseline(cfg.seed), "train", 0, 3)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    def test_resume_and_cache(self):
        for loss in ("all_token", "mean_output"):
            cfg = Config("forced_linear", "parameter", M=8, n=4, epochs=2,
                         batch_size=4, validation_size=4, test_size=4, depth=1,
                         width=4, ff_width=8, dropout=0., loss=loss, variant=loss)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.assertNotEqual(training.run_path(root, cfg), original_training.run_path(root, cfg))
                training.train(root, cfg, "cpu", stop_after_epoch=1)
                folder = training.train(root, cfg, "cpu")
                best_hash = file_hash(folder/"best.pt")
                training.train(root, cfg, "cpu")
                self.assertEqual(file_hash(folder/"best.pt"), best_hash)
                checkpoint = torch.load(folder/"last.pt", weights_only=False)
                self.assertEqual(checkpoint["epoch"], 2)
                self.assertEqual(checkpoint["best_checkpoint"]["validation_objective_mse"],
                                 min(r["validation_objective_mse"] for r in checkpoint["history"]))
                evaluation.evaluate(root, cfg, "cpu", ns=(4,), max_batch=2)
                marker = next((folder/"inference").glob("*/metrics.json"))
                timestamp = marker.stat().st_mtime_ns
                evaluation.evaluate(root, cfg, "cpu", ns=(4,), max_batch=2)
                self.assertEqual(marker.stat().st_mtime_ns, timestamp)
                metrics = json.loads(marker.read_text())["metrics"]
                self.assertLessEqual(metrics["mean_output_mse"], metrics["all_token_mse"]+1e-7)

if __name__ == "__main__":
    unittest.main()

