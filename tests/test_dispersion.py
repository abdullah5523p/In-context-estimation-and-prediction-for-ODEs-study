import tempfile
import unittest
from pathlib import Path
import numpy as np
from ode_paper.test_dispersion import test_dispersion, shade_test_ci
test_dispersion.__test__ = False  # library function, not a pytest test

class DispersionTests(unittest.TestCase):
    def test_task_average_before_std_and_mean_readout(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Correlated coordinate errors: flattening coordinates gives the wrong SD.
            errors = np.array([[0., 0.], [2., 6.], [4., 12.]])
            prediction = np.array([[0., 0.], [1., 3.], [2., 6.]])
            np.savez(Path(tmp)/"per_example.npz", all_token_coordinate_mse=errors,
                     mean_prediction=prediction, targets=np.zeros_like(prediction))
            mean_losses = (prediction**2).mean(1)
            record = dict(identity=dict(count=3), metrics=dict(
                all_token_mse=4., mean_output_mse=float(mean_losses.mean())))
            got = test_dispersion(tmp, record)
            self.assertEqual(got["all_token_mse_std"], 4.)
            self.assertAlmostEqual(got["mean_output_mse_std"], mean_losses.std(ddof=1))
            self.assertEqual(got["all_token_mse_minus_std"], 0.)
            self.assertEqual(got["test_count"], 3)
            se = 4./np.sqrt(3)
            self.assertAlmostEqual(got["all_token_mse_se"], se)
            self.assertAlmostEqual(got["all_token_mse_ci95_half_width"], 1.959963984540054*se)
            self.assertAlmostEqual(got["all_token_mse_ci95_lower"], 4.-1.959963984540054*se)
            self.assertAlmostEqual(got["all_token_mse_ci95_upper"], 4.+1.959963984540054*se)
            self.assertAlmostEqual(got["mean_output_mse_ci95_half_width"],
                                   1.959963984540054*mean_losses.std(ddof=1)/np.sqrt(3))
            record["metrics"]["all_token_mse"] = 5.
            with self.assertRaises(ValueError):
                test_dispersion(tmp, record)

    def test_log_band_with_nonpositive_lower_bound(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        ax.plot([1, 2], [1., 2.])
        shade_test_ci(ax, [([1, 2], [1., 2.], [2., .5], "C0")])
        self.assertGreater(ax.get_ylim()[0], 0)
        self.assertGreaterEqual(ax.get_ylim()[1], 3.)
        self.assertIn("clipped", ax.texts[-1].get_text())
        self.assertTrue(np.isfinite(ax.collections[0].get_paths()[0].vertices).all())
        plt.close(fig)
