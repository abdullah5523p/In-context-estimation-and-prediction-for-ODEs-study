"""Run the 4-model concentrated-prior study reported in the paper:
one-mode truncated Gaussian prior, S in {0.2, 0.1}, q in {0.1, 0.2}, seed 20260906.

prior_config.py defines a larger grid; it is left unchanged because its bytes
identify the cached runs. The filter below selects the reported settings."""
import argparse
from pathlib import Path
from .config import INFERENCE_NS
from .prior_config import SEEDS, configurations
from .prior_train import train
from .prior_evaluate import evaluate

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "train", "evaluate"))
    parser.add_argument("--root", type=Path, default=Path("/export/abdullah/ode-results/prior-screen"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260906])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ns", type=int, nargs="+", default=list(INFERENCE_NS))
    args = parser.parse_args()
    configs = [cfg for cfg in configurations(args.seeds)
               if cfg.prior == "one_mode" and cfg.s in (.1, .2) and cfg.q in (.1, .2)]
    for i, cfg in enumerate(configs, 1):
        root = args.root / f"seed_{cfg.seed}"
        print(f"Prior sweep {i}/{len(configs)}: seed={cfg.seed} prior={cfg.prior} "
              f"S={cfg.s:g} q={cfg.q:g} n={cfg.n} M={cfg.M:,}", flush=True)
        if args.command == "train":
            train(root, cfg, args.device)
        elif args.command == "evaluate":
            evaluate(root, cfg, args.device, ns=args.ns)

if __name__ == "__main__":
    main()

