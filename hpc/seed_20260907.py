"""Compatibility entry point for the unified 52-run, two-seed CLI."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ode_paper.cli import main

if __name__ == "__main__":
    sys.argv.extend(["--seed", "20260907", "--all"])
    main()
