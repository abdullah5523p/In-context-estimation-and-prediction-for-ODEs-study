from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from ode_paper.config import Config, FAMILIES, grid
from ode_paper.data import generate, bank
from ode_paper.families import SPECS, solution, encode, decode
from ode_paper.model import Transformer, FullWidthAttention, all_token_loss
from ode_paper.train import train
from ode_paper.evaluate import evaluate
from ode_paper.utils import file_hash

torch.set_num_threads(1)


def tiny(family="forced_linear", task="parameter"):
    return Config(family, task, M=8, n=2, epochs=2, batch_size=4,
                  validation_size=8, test_size=12, depth=1, width=8, ff_width=16)


class ProtocolTests(unittest.TestCase):
    def test_grid(self):
        self.assertEqual(len(grid()), 46)
        self.assertEqual(len({(c.family, c.task, c.M, c.n) for c in grid()}), 46)
        self.assertEqual(sum(c.family == "forced_linear" for c in grid()), 40)
        self.assertTrue(all(c.n == 8 and c.M == 2**19 for c in grid() if c.family != "forced_linear"))

    def test_solutions_and_derivatives(self):
        for family in FAMILIES:
            cfg = tiny(family)
            data = generate(cfg, "test", 0, 7)
            theta = data["theta"]
            s = np.full((7, 1), .35)
            x = solution(theta, s, family)[:, 0]
            dx = (solution(theta, s+1e-6, family)-solution(theta, s-1e-6, family))[:, 0]/(2e-6)
            a, b, c = theta[:, :3].T
            if family == "forced_linear":
                rhs = (-x[:, 0]+b+c*4*s[:, 0])[:, None]
                initial = a[:, None]
            elif family == "logistic":
                rhs = (b*x[:, 0]*(1-x[:, 0]/c))[:, None]
                initial = a[:, None]
            elif family == "birth_death":
                rhs = ((a-b)*x[:, 0]-c*x[:, 0]**2)[:, None]
                initial = np.full((7, 1), .2)
            else:
                xx, yy = x.T; omega = theta[:, 3]; r2 = xx*xx+yy*yy
                rhs = np.stack((c*xx-omega*yy-r2*xx, omega*xx+c*yy-r2*yy), axis=1)
                initial = theta[:, :2]
            np.testing.assert_allclose(dx, 4*rhs, rtol=1e-7, atol=1e-8)
            np.testing.assert_allclose(solution(theta, np.zeros((7, 1)), family)[:, 0], initial)
            np.testing.assert_allclose(decode(encode(theta, family), family), theta)

    def test_nested_unsorted_and_chunk_independence(self):
        for family in FAMILIES:
            cfg = tiny(family)
            small = generate(cfg, "test", 0, 12, 2)
            big = generate(replace(cfg, M=1024), "test", 0, 20, 128)
            np.testing.assert_array_equal(small["points"], big["points"][:12, :2])
            chunk = generate(cfg, "test", 4, 3, 128)
            np.testing.assert_array_equal(chunk["points"], big["points"][4:7])
            pred = generate(replace(cfg, task="prediction"), "test", 0, 12, 2)
            np.testing.assert_array_equal(pred["points"][..., :-1], small["points"])
            self.assertTrue(np.any(np.diff(big["points"][..., 0], axis=1) < 0))
            self.assertTrue((pred["tau"] >= 0).all() and (pred["tau"] <= 1).all())

    def test_noise_scale_and_bank(self):
        cfg = tiny("stuart_landau")
        d = generate(cfg, "train", 0, 10000, 2)
        clean = solution(d["theta"], d["points"][..., 0], cfg.family)/SPECS[cfg.family]["scale"]
        noise = d["points"][..., 1:]-clean
        np.testing.assert_allclose(noise.std(axis=(0, 1)), .02, atol=.0005)
        with tempfile.TemporaryDirectory() as root:
            b = bank(root, cfg, "train", chunk_size=3)
            direct = generate(cfg, "train", 0, cfg.M)
            np.testing.assert_array_equal(b["points"], direct["points"])

    def test_attention_and_symmetry(self):
        torch.manual_seed(1)
        layer = FullWidthAttention(8, 2)
        x = torch.randn(3, 5, 8)
        manual = torch.zeros_like(x)
        for h in range(2):
            q, k, v = [torch.nn.functional.linear(x, a.weight[h*8:(h+1)*8]) for a in (layer.q, layer.k, layer.v)]
            manual += (q@k.transpose(-1,-2)/np.sqrt(8)).softmax(-1)@v
        torch.testing.assert_close(layer(x), manual)
        for task in ("parameter", "prediction"):
            cfg = tiny("stuart_landau", task)
            model = Transformer(cfg).eval()
            x = torch.from_numpy(generate(cfg, "test", 0, 3, 8)["points"])
            perm = torch.randperm(8)
            torch.testing.assert_close(model(x[:, perm]), model(x)[:, perm], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(model(x.repeat_interleave(2, dim=1)), model(x).repeat_interleave(2, dim=1), atol=1e-6, rtol=1e-5)
            y = torch.randn(3, model(x).shape[-1])
            self.assertAlmostEqual(all_token_loss(model(x), y).item(), all_token_loss(model(x).repeat_interleave(2, 1), y).item(), places=6)
            self.assertFalse(any("bias" in name for name, _ in model.named_parameters()))

    def test_resume_and_cache(self):
        cfg = tiny()
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            complete = train(a, cfg, "cpu")
            train(b, cfg, "cpu", stop_after_epoch=1)
            resumed = train(b, cfg, "cpu")
            first = torch.load(complete/"last.pt", weights_only=False)
            second = torch.load(resumed/"last.pt", weights_only=False)
            for key in first["model"]:
                torch.testing.assert_close(first["model"][key], second["model"][key], rtol=0, atol=0)
            self.assertEqual(first["scheduler"], second["scheduler"])
            before = file_hash(resumed/"best.pt")
            train(b, cfg, "cpu")
            self.assertEqual(before, file_hash(resumed/"best.pt"))
            evaluate(b, cfg, "cpu", ns=(2, 4), max_batch=3)
            paths = list((resumed/"inference").glob("*/metrics.json"))
            times = {p: p.stat().st_mtime_ns for p in paths}
            evaluate(b, cfg, "cpu", ns=(2, 4), max_batch=3)
            self.assertEqual(times, {p: p.stat().st_mtime_ns for p in paths})
            for p in paths:
                self.assertTrue(np.isfinite(json.loads(p.read_text())["metrics"]["all_token_mse"]))

    def test_all_families_both_tasks_train_and_evaluate(self):
        with tempfile.TemporaryDirectory() as root:
            for family in FAMILIES:
                for task in ("parameter", "prediction"):
                    cfg = replace(tiny(family, task), epochs=1)
                    train(root, cfg, "cpu")
                    evaluate(root, cfg, "cpu", ns=(2, 8), max_batch=4)

    def test_full_architecture_at_largest_inference_context(self):
        cfg = Config("stuart_landau", "prediction", 2**19, 128)
        torch.manual_seed(cfg.seed)
        model = Transformer(cfg).eval()
        b = generate(cfg, "test", 0, 1, 512)
        with torch.inference_mode():
            output = model(torch.from_numpy(b["points"]))
        self.assertEqual(tuple(output.shape), (1, 512, 2))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(len(model.blocks), 16)
        self.assertEqual(tuple(model.blocks[0].attn.q.weight.shape), (128, 64))
        self.assertEqual(tuple(model.blocks[0].fc1.weight.shape), (256, 64))

    def test_plot_pipeline(self):
        from ode_paper.plots import plot
        from ode_paper.train import run_path
        from ode_paper.evaluate import evaluation_identity
        from ode_paper.utils import write_json, save_torch
        from ode_paper.config import fingerprint
        with tempfile.TemporaryDirectory() as root:
            # Synthetic plotting fixture only, confined to a temporary directory.
            for cfg in [grid()[i] for i in (0, 4, 15, 19, 20, 24, 35, 39)]:
                folder = run_path(root, cfg)
                save_torch(folder/"best.pt", {"fixture": True})
                write_json(folder/"history.json", [dict(epoch=1, train_all_token_mse=.2, validation_all_token_mse=.1)])
                for n in (2, 128):
                    identity = evaluation_identity(cfg, file_hash(folder/"best.pt"), n)
                    dest = folder/"inference"/f"n{n}_{fingerprint(identity)}"/"metrics.json"
                    write_json(dest, dict(identity=identity, best_epoch=1, metrics=dict(all_token_mse=.1,
                        physical_all_token_mse=.2, interpolation=dict(all_token_mse=.08), extrapolation=dict(all_token_mse=.18))))
                    np.savez(dest.parent/"per_example.npz",
                             all_token_coordinate_mse=np.full((cfg.test_size, 3), .1),
                             mean_prediction=np.full((cfg.test_size, 3), .1),
                             targets=np.zeros((cfg.test_size, 3)))
            plot(root)
            self.assertTrue((Path(root)/"figures"/"inference_summary.csv").exists())
            self.assertEqual(len(list((Path(root)/"figures").glob("*.pdf"))), 6)
            self.assertEqual(len(list((Path(root)/"figures").glob("*.png"))), 6)

    def test_comparison_curve_membership(self):
        from ode_paper.plots import comparison_specs
        from ode_paper.config import SIZES, TRAIN_NS, INFERENCE_NS
        rows = [dict(M=M, train_n=nt, inference_n=ni, all_token_mse=.1)
                for M in SIZES for nt in TRAIN_NS for ni in INFERENCE_NS]
        a, b, d = comparison_specs(rows)
        self.assertEqual([len(p) for _, p in a["curves"]], [5]*4)
        self.assertTrue(all(r["train_n"] == r["inference_n"] for _, p in a["curves"] for r in p))
        self.assertEqual([len(p) for _, p in b["curves"]], [9]*5)
        self.assertTrue(all(r["train_n"] == 128 for _, p in b["curves"] for r in p))
        self.assertEqual([len(p) for _, p in d["curves"]], [9]*4)
        self.assertTrue(all(r["M"] == 2**19 for _, p in d["curves"] for r in p))


if __name__ == "__main__":
    unittest.main()
