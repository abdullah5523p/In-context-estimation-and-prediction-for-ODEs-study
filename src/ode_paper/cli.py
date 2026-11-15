import argparse
import json
from pathlib import Path
from .config import INFERENCE_NS
from .runs import SEEDS, configurations


def main():
    parser = argparse.ArgumentParser(description="Reproducible ODE experiments (52 configurations per seed)")
    parser.add_argument("command", choices=("list", "train", "evaluate", "plot"))
    parser.add_argument("--root", type=Path, default=Path("results"), help="All generated data/artifacts go here")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0], help="Set both model and data seed")
    parser.add_argument("--previous-root", type=Path, help="With seed 20260907, also process seed 20260906 in this folder")
    parser.add_argument("--index", type=int, help="Zero-based configuration index; see list")
    parser.add_argument("--all", action="store_true", help="Explicitly run the entire configured grid serially")
    parser.add_argument("--device", default="cuda", choices=("cuda", "mps", "cpu"))
    parser.add_argument("--ns", nargs="+", type=int, default=list(INFERENCE_NS))
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--token-budget", type=int, default=16384)
    args = parser.parse_args()
    configs = configurations(args.seed)
    sources = [(args.root, configs)]
    if args.previous_root is not None:
        if args.seed != SEEDS[1]:
            parser.error("--previous-root requires --seed 20260907")
        if args.previous_root.resolve() == args.root.resolve():
            parser.error("Use separate original-seed and new-seed results folders")
        sources.insert(0, (args.previous_root, configurations(SEEDS[0])))
    if args.command == "list":
        for root, seed_configs in sources:
            for index, cfg in enumerate(seed_configs):
                print(json.dumps(dict(index=index, root=str(root), **cfg.to_dict())))
        return
    if args.command == "plot":
        from .plots import plot
        plot(args.root, sources=sources); return
    if args.all == (args.index is not None):
        parser.error("Choose exactly one of --index or --all")
    if args.index is not None and not 0 <= args.index < len(configs):
        parser.error(f"Index must be 0..{len(configs)-1}")
    if args.eval_batch_size <= 0 or args.token_budget <= 0 or any(n <= 0 for n in args.ns):
        parser.error("Evaluation sizes must be positive")
    for root, seed_configs in sources:
        selected = list(enumerate(seed_configs)) if args.all else [(args.index, seed_configs[args.index])]
        print(f"Results folder: {root}", flush=True)
        for index, cfg in selected:
            print(f"Seed {cfg.seed}, configuration {index+1}/{len(seed_configs)}: {cfg}", flush=True)
            if args.command == "train":
                from .train import train
                train(root, cfg, args.device)
            else:
                from .evaluate import evaluate
                evaluate(root, cfg, args.device, args.ns, args.token_budget, args.eval_batch_size)


if __name__ == "__main__":
    main()
