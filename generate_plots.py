#!/usr/bin/env python3
"""
generate_plots.py  —  Post-hoc analysis plots for EvoRob evolutionary runs.
============================================================================

Usage examples
--------------
# All plots for a single run (saves to results/.../plots/):
  python generate_plots.py --run-dir results/body_brain_023/seed_0

# Explicit output directory:
  python generate_plots.py --run-dir results/body_brain_023/seed_0 --out-dir my_plots/

# Average across replicates (shaded std band):
  python generate_plots.py --run-dir results/body_brain_023/seed_0 \
                                      results/body_brain_024/seed_0

# NN vs Hebbian comparison (Plots 2 & 6):
  python generate_plots.py --nn-dir   results/body_brain_023/seed_0 \
                           --hebb-dir results/body_brain_025/seed_0

# One specific plot:
  python generate_plots.py --run-dir results/body_brain_023/seed_0 --plot fitness
  python generate_plots.py --run-dir results/body_brain_025/seed_0 --plot weight-traj --terrain hill
  python generate_plots.py --run-dir results/body_brain_025/seed_0 --plot adaptation  --terrain flat
  python generate_plots.py --run-dir results/body_brain_025/seed_0 --plot abcd
  python generate_plots.py --run-dir results/body_brain_023/seed_0 --plot pareto

Available --plot choices
------------------------
  all           (default) every applicable plot for each --run-dir
  fitness       Plot 1: per-terrain fitness curves
  comparison    Plot 2 + 6: NN vs Hebbian (needs --nn-dir and --hebb-dir)
  weight-traj   Plot 3: Hebbian weight trajectory (Hebbian only)
  pareto        Plot 4: per-terrain Pareto fronts
  abcd          Plot 5: ABCD coefficient heatmap (Hebbian only)
  distributions Plot 6: fitness violin/box distributions
  adaptation    Plot 7: Hebbian adaptation curve (Hebbian only)
"""

import argparse
import os
import sys

# Ensure the project root (this file's directory) is importable as 'evorob'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evorob.utils.plots import (
    generate_all_plots,
    generate_comparison_plots,
    get_last_checkpoint_dir,
    load_run_metadata,
    plot_abcd_heatmap,
    plot_adaptation_curve,
    plot_fitness_distributions,
    plot_nn_vs_hebbian,
    plot_per_terrain_fitness,
    plot_terrain_pareto_fronts,
    plot_weight_trajectory,
)


def _out(run_dir: str, explicit: str | None) -> str:
    """Resolve output directory: explicit if given, else {run_dir}/plots."""
    return explicit or os.path.join(run_dir, "plots")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate analysis plots for EvoRob evolutionary runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    parser.add_argument(
        "--run-dir", nargs="+", metavar="DIR", default=[],
        help="Seed directory (one or more for replicate averaging). "
             "E.g. results/body_brain_023/seed_0",
    )
    parser.add_argument(
        "--nn-dir", nargs="+", metavar="DIR", default=[],
        help="NN run directories — used for NN vs Hebbian comparison.",
    )
    parser.add_argument(
        "--hebb-dir", nargs="+", metavar="DIR", default=[],
        help="Hebbian run directories — used for NN vs Hebbian comparison.",
    )

    # Output
    parser.add_argument(
        "--out-dir", metavar="DIR", default=None,
        help="Output directory.  Default: {run_dir}/plots/ for single-run modes.",
    )

    # What to produce
    parser.add_argument(
        "--plot",
        choices=[
            "all", "fitness", "comparison",
            "weight-traj", "pareto", "abcd",
            "distributions", "adaptation",
        ],
        default="all",
        help="Which plot(s) to generate.  Default: all.",
    )
    parser.add_argument(
        "--terrain",
        choices=["flat", "ice", "hill"],
        default="flat",
        help="Terrain for --plot weight-traj and --plot adaptation.  Default: flat.",
    )
    parser.add_argument(
        "--style",
        choices=["violin", "box"],
        default="violin",
        help="Style for --plot distributions.  Default: violin.",
    )

    args = parser.parse_args()

    run_dirs  = args.run_dir
    nn_dirs   = args.nn_dir
    hebb_dirs = args.hebb_dir

    # ------------------------------------------------------------------ #
    if not run_dirs and not nn_dirs and not hebb_dirs:
        parser.print_help()
        sys.exit(1)

    # ------------------------------------------------------------------ #
    #  Dispatch
    # ------------------------------------------------------------------ #

    if args.plot == "comparison" or (nn_dirs or hebb_dirs):
        out = args.out_dir or "plots_comparison"
        generate_comparison_plots(nn_dirs, hebb_dirs, out)

    elif args.plot == "all":
        for rd in run_dirs:
            out = args.out_dir or os.path.join(rd, "plots")
            generate_all_plots(rd, out)

    elif args.plot == "fitness":
        if not run_dirs:
            print("ERROR: --run-dir required for --plot fitness")
            sys.exit(1)
        out = _out(run_dirs[0], args.out_dir)
        plot_per_terrain_fitness(run_dirs, out)

    elif args.plot == "weight-traj":
        if not run_dirs:
            print("ERROR: --run-dir required for --plot weight-traj")
            sys.exit(1)
        out = _out(run_dirs[0], args.out_dir)
        for rd in run_dirs:
            ckpt = get_last_checkpoint_dir(rd)
            if ckpt:
                plot_weight_trajectory(ckpt, out, args.terrain)
            else:
                print(f"  No checkpoint found in {rd!r} — skipping")

    elif args.plot == "pareto":
        if not run_dirs:
            print("ERROR: --run-dir required for --plot pareto")
            sys.exit(1)
        for rd in run_dirs:
            out = _out(rd, args.out_dir)
            plot_terrain_pareto_fronts(rd, out)

    elif args.plot == "abcd":
        if not run_dirs:
            print("ERROR: --run-dir required for --plot abcd")
            sys.exit(1)
        for rd in run_dirs:
            out = _out(rd, args.out_dir)
            ckpt = get_last_checkpoint_dir(rd)
            if ckpt:
                plot_abcd_heatmap(ckpt, out)
            else:
                print(f"  No checkpoint found in {rd!r} — skipping")

    elif args.plot == "distributions":
        if not run_dirs:
            print("ERROR: --run-dir required for --plot distributions")
            sys.exit(1)
        out = _out(run_dirs[0], args.out_dir)
        # Group by detected controller
        dirs_by_ctrl: dict[str, list[str]] = {}
        for rd in run_dirs:
            ctrl = load_run_metadata(rd).get("controller", "nn")
            dirs_by_ctrl.setdefault(ctrl, []).append(rd)
        plot_fitness_distributions(dirs_by_ctrl, out, style=args.style)

    elif args.plot == "adaptation":
        if not run_dirs:
            print("ERROR: --run-dir required for --plot adaptation")
            sys.exit(1)
        for rd in run_dirs:
            out = _out(rd, args.out_dir)
            ckpt = get_last_checkpoint_dir(rd)
            if ckpt:
                plot_adaptation_curve(ckpt, out, args.terrain)
            else:
                print(f"  No checkpoint found in {rd!r} — skipping")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
