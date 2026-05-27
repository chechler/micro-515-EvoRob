"""
evorob/utils/plots.py
=====================
Post-hoc analysis and visualisation for EvoRob evolutionary runs.

All functions load their data from disk and save plots to an output directory.
Every function is callable independently — missing data prints a warning and
returns early instead of crashing.

New data logged during training (added 2026-05-27)
---------------------------------------------------
  run_metadata.json              controller type, n_params, hyperparams
  fitness_log.json               per-generation best/mean per terrain
  population_fitnesses_final.npy full 200-individual fitness matrix at last gen
  population_ranks_final.npy     Pareto ranks of those 200 individuals

Entry points
------------
generate_all_plots(run_dir, out_dir=None)
    Every applicable plot for one run.  Hebbian-only plots are silently skipped
    for NN runs.

generate_comparison_plots(nn_dirs, hebb_dirs, out_dir)
    Cross-controller comparison: overlay (Plot 2) and distributions (Plot 6).

Individual plot functions
-------------------------
1. plot_per_terrain_fitness(run_dirs, out_dir)
2. plot_nn_vs_hebbian(nn_dirs, hebb_dirs, out_dir)
3. plot_weight_trajectory(checkpoint_dir, out_dir, terrain)
4. plot_terrain_pareto_fronts(run_dir, out_dir)
5. plot_abcd_heatmap(checkpoint_dir, out_dir)
6. plot_fitness_distributions(run_dirs_by_ctrl, out_dir)
7. plot_adaptation_curve(checkpoint_dir, out_dir, terrain)
"""

from __future__ import annotations

import json
import os
import platform
import sys
import warnings
from os.path import join
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

TERRAIN_LABELS = ["Flat", "Ice", "Hill"]
TERRAIN_COLORS = {"Flat": "#2196F3", "Ice": "#FF9800", "Hill": "#4CAF50"}
CTRL_COLORS    = {"nn": "#E91E63", "hebbian": "#9C27B0"}
CTRL_LABELS    = {"nn": "Neural Network", "hebbian": "Hebbian"}

_STYLE: dict = {
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.size":         10,
}


def _apply_style() -> None:
    plt.rcParams.update(_STYLE)


# ---------------------------------------------------------------------------
# Data-loading helpers
# ---------------------------------------------------------------------------

def _sorted_ckpt_dirs(run_dir: str) -> List[str]:
    """Sorted list of numeric checkpoint sub-directory names (e.g. '0','25','50',...)."""
    try:
        return sorted(
            [d for d in os.listdir(run_dir) if d.isdigit()],
            key=int,
        )
    except FileNotFoundError:
        return []


def load_fitness_log(run_dir: str) -> Optional[List[dict]]:
    """Load per-generation fitness log.

    Prefers ``fitness_log.json`` (saved during training).  Falls back to
    reconstructing a sparse log from checkpoint ``f_best.npy`` files (every
    25 gens).  The fallback only has "best" values; "mean" mirrors "best".

    Returns:
        List of dicts with keys: "gen", "best" ([flat,ice,hill]), "mean".
        None if no data could be found.
    """
    log_path = join(run_dir, "fitness_log.json")
    if os.path.isfile(log_path):
        with open(log_path) as fh:
            return json.load(fh)

    # Fallback — sparse reconstruction from checkpoints
    ckpts = _sorted_ckpt_dirs(run_dir)
    if not ckpts:
        warnings.warn(
            f"load_fitness_log: no fitness_log.json and no checkpoint dirs in {run_dir!r}",
            stacklevel=2,
        )
        return None

    log: List[dict] = []
    for ckpt in ckpts:
        fpath = join(run_dir, ckpt, "f_best.npy")
        if os.path.isfile(fpath):
            f = np.load(fpath).tolist()
            log.append({"gen": int(ckpt), "best": f, "mean": f})

    if not log:
        warnings.warn(
            f"load_fitness_log: could not reconstruct log from checkpoints in {run_dir!r}",
            stacklevel=2,
        )
        return None

    warnings.warn(
        f"load_fitness_log: using sparse fallback from checkpoints ({len(log)} points) in {run_dir!r}",
        stacklevel=2,
    )
    return log


def load_run_metadata(run_dir: str) -> dict:
    """Load run metadata dict.

    Prefers ``run_metadata.json``.  Falls back to parsing ``training_score.txt``
    for the controller type.

    Returns:
        Dict with at minimum: "controller" ("nn" or "hebbian"), "n_weights".
    """
    meta_path = join(run_dir, "run_metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as fh:
            return json.load(fh)

    # Fallback
    meta = {"controller": "nn", "n_weights": 280, "n_body_params": 2, "n_params": 282}
    score_path = join(run_dir, "training_score.txt")
    if os.path.isfile(score_path):
        with open(score_path) as fh:
            for line in fh:
                if "HebbianController" in line:
                    meta.update({"controller": "hebbian", "n_weights": 1120, "n_params": 1122})
                    break
    return meta


def load_population_fitnesses(run_dir: str) -> Optional[np.ndarray]:
    """Load final-generation population fitness matrix.

    Requires ``population_fitnesses_final.npy`` (saved at end of training for
    runs after 2026-05-27).  Returns None for older runs.

    Shape: (population_size, 3)  —  columns = [flat, ice, hill]
    """
    pop_path = join(run_dir, "population_fitnesses_final.npy")
    if os.path.isfile(pop_path):
        return np.load(pop_path)
    return None


def load_population_ranks(run_dir: str) -> Optional[np.ndarray]:
    """Load Pareto ranks for the final population.  Shape: (population_size,)."""
    path = join(run_dir, "population_ranks_final.npy")
    return np.load(path) if os.path.isfile(path) else None


def get_last_checkpoint_dir(run_dir: str) -> Optional[str]:
    """Return absolute path to the highest-numbered checkpoint sub-directory."""
    ckpts = _sorted_ckpt_dirs(run_dir)
    return join(run_dir, ckpts[-1]) if ckpts else None


def detect_controller(run_dir: str) -> str:
    """Return 'nn' or 'hebbian' inferred from run metadata."""
    return load_run_metadata(run_dir).get("controller", "nn")


# ---------------------------------------------------------------------------
# Multi-run aggregation
# ---------------------------------------------------------------------------

def _aggregate_logs(
    run_dirs: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align fitness logs from multiple runs and compute mean ± std per terrain.

    Runs are trimmed to the length of the shortest log before aggregation.

    Returns:
        gens   shape (G,)    — generation numbers
        means  shape (G, 3)  — mean best-per-terrain across runs
        stds   shape (G, 3)  — std of best-per-terrain across runs
    """
    logs = [load_fitness_log(d) for d in run_dirs]
    logs = [lg for lg in logs if lg is not None]
    if not logs:
        return np.array([]), np.array([]), np.array([])

    min_len = min(len(lg) for lg in logs)
    trimmed = [lg[:min_len] for lg in logs]

    gens = np.array([e["gen"] for e in trimmed[0]], dtype=float)
    # best = per-terrain maximum across the population this generation
    best_curves = np.array([[e["best"] for e in lg] for lg in trimmed])  # (n_runs, G, 3)
    return gens, best_curves.mean(axis=0), best_curves.std(axis=0)


# ---------------------------------------------------------------------------
# Plot 1 — Per-terrain fitness curves
# ---------------------------------------------------------------------------

def plot_per_terrain_fitness(
    run_dirs: Sequence[str],
    out_dir: str,
    label: str = "",
) -> None:
    """Per-terrain best & population-mean fitness over generations.

    Single run: best (solid) and mean (dashed) on the same axes.
    Multiple replicates: mean ± 1-std shaded band across runs.

    Args:
        run_dirs: One or more seed directories.
        out_dir:  Where to save the PNG.
        label:    Optional suffix for the filename.
    """
    os.makedirs(out_dir, exist_ok=True)
    _apply_style()

    if len(run_dirs) == 1:
        log = load_fitness_log(run_dirs[0])
        if log is None:
            warnings.warn(f"plot_per_terrain_fitness: no data in {run_dirs[0]!r}")
            return

        gens  = np.array([e["gen"]  for e in log], dtype=float)
        bests = np.array([e["best"] for e in log])   # (G, 3)
        means = np.array([e["mean"] for e in log])   # (G, 3)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
        for j, (ax, terrain) in enumerate(zip(axes, TERRAIN_LABELS)):
            c = TERRAIN_COLORS[terrain]
            ax.plot(gens, bests[:, j], color=c, lw=2, label="best (this gen)")
            ax.plot(gens, means[:, j], color=c, lw=1.2, ls="--", alpha=0.65,
                    label="population mean")
            ax.set_title(terrain, fontweight="bold")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Fitness")
            ax.legend(fontsize=8)

        run_name = os.path.basename(os.path.normpath(run_dirs[0]))
        fig.suptitle(f"Per-terrain Fitness — {run_name}", fontsize=13)

    else:
        gens, rep_means, rep_stds = _aggregate_logs(run_dirs)
        if len(gens) == 0:
            warnings.warn("plot_per_terrain_fitness: no data in any run_dir")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
        for j, (ax, terrain) in enumerate(zip(axes, TERRAIN_LABELS)):
            c = TERRAIN_COLORS[terrain]
            ax.plot(gens, rep_means[:, j], color=c, lw=2,
                    label=f"mean  (n={len(run_dirs)})")
            ax.fill_between(
                gens,
                rep_means[:, j] - rep_stds[:, j],
                rep_means[:, j] + rep_stds[:, j],
                color=c, alpha=0.2, label="±1 std",
            )
            ax.set_title(terrain, fontweight="bold")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Fitness")
            ax.legend(fontsize=8)

        fig.suptitle(f"Per-terrain Fitness  ({len(run_dirs)} replicates)", fontsize=13)

    plt.tight_layout()
    suffix = f"_{label}" if label else ""
    out_path = join(out_dir, f"fitness_per_terrain{suffix}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — NN vs Hebbian overlay
# ---------------------------------------------------------------------------

def plot_nn_vs_hebbian(
    nn_dirs: Sequence[str],
    hebb_dirs: Sequence[str],
    out_dir: str,
) -> None:
    """Overlay NN and Hebbian best-fitness curves, one subplot per terrain.

    Shaded ±1-std band is shown when multiple replicates exist; a bare line
    otherwise.  Either controller group may be empty (just plots the other one).

    Args:
        nn_dirs:   Run directories using NeuralNetworkController.
        hebb_dirs: Run directories using HebbianController.
        out_dir:   Where to save the PNG.
    """
    os.makedirs(out_dir, exist_ok=True)
    _apply_style()

    groups = {
        "nn":      (nn_dirs,   CTRL_COLORS["nn"],      CTRL_LABELS["nn"]),
        "hebbian": (hebb_dirs, CTRL_COLORS["hebbian"],  CTRL_LABELS["hebbian"]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    for ctrl, (dirs, color, ctrl_label) in groups.items():
        if not dirs:
            continue
        gens, rep_means, rep_stds = _aggregate_logs(dirs)
        if len(gens) == 0:
            continue
        for j, ax in enumerate(axes):
            ax.plot(gens, rep_means[:, j], color=color, lw=2, label=ctrl_label)
            if len(dirs) > 1:
                ax.fill_between(
                    gens,
                    rep_means[:, j] - rep_stds[:, j],
                    rep_means[:, j] + rep_stds[:, j],
                    color=color, alpha=0.18,
                )

    for ax, terrain in zip(axes, TERRAIN_LABELS):
        ax.set_title(terrain, fontweight="bold")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        # Deduplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        seen: dict = {}
        for h, lbl in zip(handles, labels):
            seen.setdefault(lbl, h)
        ax.legend(seen.values(), seen.keys(), fontsize=8)

    fig.suptitle("NN vs Hebbian — Per-terrain Fitness", fontsize=13)
    plt.tight_layout()
    out_path = join(out_dir, "nn_vs_hebbian_fitness.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3 — Hebbian weight trajectory (within one episode)
# ---------------------------------------------------------------------------

def plot_weight_trajectory(
    checkpoint_dir: str,
    out_dir: str,
    terrain: str = "flat",
) -> None:
    """Track mean |weight| of lin1 and output matrices over timesteps in one episode.

    Confirms that Hebbian adaptation is actually occurring during rollout.
    The horizontal dashed line marks the initialisation scale (0.1).

    Requires HebbianController checkpoint.  Skips gracefully for NN.

    Args:
        checkpoint_dir: Directory containing x_best.npy (and sibling Robot.xml).
        out_dir:        Where to save the PNG.
        terrain:        "flat", "ice", or "hill".
    """
    os.makedirs(out_dir, exist_ok=True)

    run_dir = _find_run_dir(checkpoint_dir)
    meta = load_run_metadata(run_dir) if run_dir else {}
    if meta.get("controller", "nn") != "hebbian":
        warnings.warn("plot_weight_trajectory: not a Hebbian checkpoint — skipping")
        return

    x_best_path = join(checkpoint_dir, "x_best.npy")
    if not os.path.isfile(x_best_path):
        warnings.warn(f"plot_weight_trajectory: x_best.npy not found in {checkpoint_dir!r}")
        return

    x_best = np.load(x_best_path)
    _apply_style()

    try:
        FinalWorld = _import_final_world()
    except Exception as exc:
        warnings.warn(f"plot_weight_trajectory: could not import FinalWorld — {exc}")
        return

    import gymnasium as gym  # noqa: PLC0415

    world = FinalWorld(co_evolve_body=True, controller="hebbian")
    world.update_robot_xml(x_best)

    env_id, world_file = _terrain_env(world, terrain)
    env = gym.make(env_id, robot_path=world_file, max_episode_steps=1000)
    world.controller.reset_controller(batch_size=1)

    obs, _ = env.reset(seed=0)
    lin1_trace: List[float] = []
    output_trace: List[float] = []
    done = False

    while not done:
        # Snapshot BEFORE the forward (and Hebbian update) of this step
        lin1_trace.append(float(np.mean(np.abs(world.controller.model.lin1[0]))))
        output_trace.append(float(np.mean(np.abs(world.controller.model.output[0]))))

        action = world.controller.get_action(obs)
        if action.ndim > 1:
            action = action.squeeze(0)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()

    steps = np.arange(len(lin1_trace))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, lin1_trace,   color="#1565C0", lw=2, label="lin1  (input → hidden, 8×27)")
    ax.plot(steps, output_trace, color="#B71C1C", lw=2, label="output (hidden → output, 8×8)")
    ax.axhline(0.1, color="gray", ls="--", lw=1, alpha=0.7, label="init scale (0.1)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean |weight|")
    ax.set_title(
        f"Hebbian Weight Magnitude During Episode — {terrain.capitalize()}",
        fontweight="bold",
    )
    ax.legend()
    plt.tight_layout()
    out_path = join(out_dir, f"hebbian_weight_trajectory_{terrain}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4 — Per-terrain Pareto fronts (individual files)
# ---------------------------------------------------------------------------

def plot_terrain_pareto_fronts(run_dir: str, out_dir: str) -> None:
    """Save one high-quality Pareto scatter per terrain-pair, with the rank-0 front highlighted.

    Produces three files:
      pareto_flat_vs_ice.png
      pareto_flat_vs_hill.png
      pareto_ice_vs_hill.png

    Args:
        run_dir: Seed directory containing population_fitnesses_final.npy.
        out_dir: Where to save the PNGs.
    """
    os.makedirs(out_dir, exist_ok=True)

    fitnesses = load_population_fitnesses(run_dir)
    if fitnesses is None:
        warnings.warn(
            f"plot_terrain_pareto_fronts: population_fitnesses_final.npy not found in "
            f"{run_dir!r}.  Run again after 2026-05-27 training to generate this file.",
        )
        return

    ranks = load_population_ranks(run_dir)
    if ranks is None:
        # Recompute on the fly
        try:
            from evorob.algorithms.nsga import NSGAII  # noqa: PLC0415
            ea = NSGAII(population_size=len(fitnesses), n_opt_params=1)
            _, ranks_list = ea.fast_nondominated_sort(fitnesses)
            ranks = np.array(ranks_list)
        except Exception as exc:
            warnings.warn(f"plot_terrain_pareto_fronts: could not compute ranks — {exc}")
            ranks = np.zeros(len(fitnesses), dtype=int)

    _apply_style()
    rank_arr = np.asarray(ranks)
    vmax = max(int(rank_arr.max()), 1)

    pairs = [
        (0, 1, "Flat",  "Ice"),
        (0, 2, "Flat",  "Hill"),
        (1, 2, "Ice",   "Hill"),
    ]

    for (i, j, xl, yl) in pairs:
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(
            fitnesses[:, i], fitnesses[:, j],
            c=rank_arr, cmap="plasma_r", vmin=0, vmax=vmax,
            alpha=0.65, s=20, edgecolors="none",
        )
        # Highlight rank-0 Pareto front
        mask0 = rank_arr == 0
        if mask0.any():
            # Sort by the x-axis objective so the connecting line is clean
            order = np.argsort(fitnesses[mask0, i])
            ax.plot(
                fitnesses[mask0][order, i],
                fitnesses[mask0][order, j],
                color="#FF6F00", lw=1.5, alpha=0.85, zorder=5,
            )
            ax.scatter(
                fitnesses[mask0, i], fitnesses[mask0, j],
                color="#FF6F00", s=45, zorder=6,
                label=f"Pareto front  (n={mask0.sum()})",
            )

        cbar = fig.colorbar(sc, ax=ax, shrink=0.9)
        cbar.set_label("Pareto rank  (0 = front)")
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel(yl, fontsize=11)
        ax.set_title(f"Pareto Front — {xl} vs {yl}", fontweight="bold")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fname = f"pareto_{xl.lower()}_vs_{yl.lower()}.png"
        out_path = join(out_dir, fname)
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 5 — ABCD coefficient heatmap (Hebbian only)
# ---------------------------------------------------------------------------

def plot_abcd_heatmap(checkpoint_dir: str, out_dir: str) -> None:
    """Heatmap of the evolved Hebbian A, B, C, D matrices for the best individual.

    Layout: 2 rows × 4 columns.
      Row 0 — layer 1  (input→hidden, shape 8×27 per coefficient)
      Row 1 — layer 2  (hidden→output, shape 8×8 per coefficient)
      Columns — A, B, C, D

    Colour scale is symmetric (RdBu_r), shared between the two layers of each
    coefficient so relative magnitudes are comparable.

    Args:
        checkpoint_dir: Directory containing x_best.npy.
        out_dir:        Where to save the PNG.
    """
    os.makedirs(out_dir, exist_ok=True)

    run_dir = _find_run_dir(checkpoint_dir)
    meta = load_run_metadata(run_dir) if run_dir else {}
    if meta.get("controller", "nn") != "hebbian":
        warnings.warn("plot_abcd_heatmap: not a Hebbian checkpoint — skipping")
        return

    x_best_path = join(checkpoint_dir, "x_best.npy")
    if not os.path.isfile(x_best_path):
        warnings.warn(f"plot_abcd_heatmap: x_best.npy not found in {checkpoint_dir!r}")
        return

    x_best    = np.load(x_best_path)
    n_weights = meta.get("n_weights", 1120)

    from evorob.world.robot.controllers.mlp_hebbian import HebbianController  # noqa: PLC0415

    ctrl = HebbianController(input_size=27, output_size=8, hidden_size=8)
    ctrl.geno2pheno(x_best[:n_weights])
    net = ctrl.model

    coeff_labels = ["A", "B", "C", "D"]
    layer1_mats  = [net.A1, net.B1, net.C1, net.D1]   # each (8, 27)
    layer2_mats  = [net.A2, net.B2, net.C2, net.D2]   # each (8, 8)

    _apply_style()
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    fig.suptitle(
        "Evolved Hebbian ABCD Coefficients — Best Individual",
        fontsize=14, fontweight="bold",
    )

    for col, (lbl, m1, m2) in enumerate(zip(coeff_labels, layer1_mats, layer2_mats)):
        vmax = max(float(np.abs(m1).max()), float(np.abs(m2).max()), 1e-6)

        # Layer 1
        ax1 = axes[0, col]
        im1 = ax1.imshow(m1, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax1.set_title(f"{lbl}₁  (input → hidden)", fontsize=10)
        ax1.set_xlabel("Input dim (27)", fontsize=8)
        ax1.set_ylabel("Hidden (8)", fontsize=8)
        fig.colorbar(im1, ax=ax1, shrink=0.85, pad=0.02)

        # Layer 2
        ax2 = axes[1, col]
        im2 = ax2.imshow(m2, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax2.set_title(f"{lbl}₂  (hidden → output)", fontsize=10)
        ax2.set_xlabel("Hidden (8)", fontsize=8)
        ax2.set_ylabel("Output (8)", fontsize=8)
        fig.colorbar(im2, ax=ax2, shrink=0.85, pad=0.02)

    plt.tight_layout()
    out_path = join(out_dir, "hebbian_abcd_heatmap.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 6 — Fitness distribution violin / box plots
# ---------------------------------------------------------------------------

def plot_fitness_distributions(
    run_dirs_by_ctrl: Dict[str, Sequence[str]],
    out_dir: str,
    style: str = "violin",
) -> None:
    """Violin or box plots of the final-generation population fitness.

    Requires ``population_fitnesses_final.npy`` (saved for runs after 2026-05-27).
    Skips silently if no data is available for a controller group.

    Args:
        run_dirs_by_ctrl: Maps controller key ("nn" or "hebbian") to run dirs.
                          E.g. {"nn": ["results/.../seed_0"], "hebbian": [...]}.
        out_dir:          Where to save the PNG.
        style:            "violin" (default) or "box".
    """
    os.makedirs(out_dir, exist_ok=True)

    # Gather per-terrain pools per controller
    all_data: Dict[str, List[np.ndarray]] = {}
    for ctrl, dirs in run_dirs_by_ctrl.items():
        pools: List[List[float]] = [[], [], []]
        for d in dirs:
            f = load_population_fitnesses(d)
            if f is not None:
                for t in range(3):
                    pools[t].extend(f[:, t].tolist())
        if any(len(p) > 0 for p in pools):
            all_data[ctrl] = [np.asarray(p) if p else np.array([0.0]) for p in pools]

    if not all_data:
        warnings.warn(
            "plot_fitness_distributions: population_fitnesses_final.npy not found in any "
            "run_dir.  This file is only saved for training runs after 2026-05-27."
        )
        return

    _apply_style()
    width = 0.35
    gap   = 0.1

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for t_idx, (ax, terrain) in enumerate(zip(axes, TERRAIN_LABELS)):
        positions: List[float] = []
        data_sets: List[np.ndarray] = []
        tick_labels: List[str] = []
        face_colors: List[str] = []

        for c_idx, (ctrl, terrain_data) in enumerate(all_data.items()):
            pos = c_idx * (width + gap)
            positions.append(pos)
            data_sets.append(terrain_data[t_idx])
            tick_labels.append(CTRL_LABELS.get(ctrl, ctrl))
            face_colors.append(CTRL_COLORS.get(ctrl, f"C{c_idx}"))

        if not data_sets:
            continue

        if style == "violin" and all(len(d) > 1 for d in data_sets):
            parts = ax.violinplot(
                data_sets, positions=positions, widths=width,
                showmedians=True, showextrema=True,
            )
            for body, color in zip(parts["bodies"], face_colors):
                body.set_facecolor(color)
                body.set_alpha(0.72)
            for key in ("cbars", "cmins", "cmaxes", "cmedians"):
                if key in parts:
                    parts[key].set_color("black")
                    parts[key].set_linewidth(1.2)
        else:
            bp = ax.boxplot(
                data_sets, positions=positions, widths=width,
                patch_artist=True,
                medianprops=dict(color="black", lw=2),
                whiskerprops=dict(lw=1.2),
                capprops=dict(lw=1.2),
                flierprops=dict(marker=".", markersize=3, alpha=0.5),
            )
            for patch, color in zip(bp["boxes"], face_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.72)

        ax.set_title(terrain, fontweight="bold")
        ax.set_ylabel("Fitness")
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.grid(axis="x", alpha=0.0)

    fig.suptitle("Final-generation Fitness Distributions", fontsize=13)
    plt.tight_layout()
    out_path = join(out_dir, f"fitness_distributions_{style}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 7 — Adaptation curve (Hebbian only)
# ---------------------------------------------------------------------------

def plot_adaptation_curve(
    checkpoint_dir: str,
    out_dir: str,
    terrain: str = "flat",
) -> None:
    """Cumulative reward over timesteps: adaptive Hebbian vs frozen weights.

    Runs the best Hebbian individual twice on the same terrain:
      • Adaptive  — normal Hebbian weight updates each step.
      • Frozen    — ABCD coefficients zeroed; no plasticity.

    The gap between the two curves isolates the contribution of within-episode
    adaptation.  The right panel shows per-step reward (moving average).

    Args:
        checkpoint_dir: Directory containing x_best.npy.
        out_dir:        Where to save the PNG.
        terrain:        "flat", "ice", or "hill".
    """
    os.makedirs(out_dir, exist_ok=True)

    run_dir = _find_run_dir(checkpoint_dir)
    meta = load_run_metadata(run_dir) if run_dir else {}
    if meta.get("controller", "nn") != "hebbian":
        warnings.warn("plot_adaptation_curve: not a Hebbian checkpoint — skipping")
        return

    x_best_path = join(checkpoint_dir, "x_best.npy")
    if not os.path.isfile(x_best_path):
        warnings.warn(f"plot_adaptation_curve: x_best.npy not found in {checkpoint_dir!r}")
        return

    x_best = np.load(x_best_path)
    _apply_style()

    try:
        FinalWorld = _import_final_world()
    except Exception as exc:
        warnings.warn(f"plot_adaptation_curve: could not import FinalWorld — {exc}")
        return

    import gymnasium as gym  # noqa: PLC0415

    world = FinalWorld(co_evolve_body=True, controller="hebbian")
    world.update_robot_xml(x_best)
    env_id, world_file = _terrain_env(world, terrain)

    def _run(freeze: bool) -> np.ndarray:
        env = gym.make(env_id, robot_path=world_file, max_episode_steps=1000)
        world.controller.reset_controller(batch_size=1)
        net = world.controller.model

        saved_abcd: Optional[np.ndarray] = None
        if freeze:
            saved_abcd = np.concatenate([net.A, net.B, net.C, net.D])
            n_abcd = (net.size_l1 + net.size_l2) * 4
            net.set_hebbian_rules(np.zeros(n_abcd, dtype=np.float32))

        obs, _ = env.reset(seed=0)
        step_rewards: List[float] = []
        done = False
        while not done:
            action = world.controller.get_action(obs)
            if action.ndim > 1:
                action = action.squeeze(0)
            obs, r, terminated, truncated, _ = env.step(action)
            step_rewards.append(float(r))
            done = terminated or truncated

        env.close()
        if freeze and saved_abcd is not None:
            net.set_hebbian_rules(saved_abcd)

        return np.array(step_rewards)

    r_adaptive = _run(freeze=False)
    r_frozen   = _run(freeze=True)

    cum_adapt = np.cumsum(r_adaptive)
    cum_frz   = np.cumsum(r_frozen)

    window = min(20, len(r_adaptive) // 5 or 1)

    def _smooth(arr: np.ndarray, w: int) -> np.ndarray:
        return np.convolve(arr, np.ones(w) / w, mode="valid")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # Left: cumulative reward
    ax1.plot(np.arange(len(cum_adapt)), cum_adapt,
             color="#1565C0", lw=2, label="Hebbian (adaptive)")
    ax1.plot(np.arange(len(cum_frz)), cum_frz,
             color="#B71C1C", lw=2, ls="--", label="Frozen weights")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Cumulative reward")
    ax1.set_title(f"Adaptation Curve — {terrain.capitalize()}", fontweight="bold")
    ax1.legend()

    # Right: smoothed per-step reward
    s_adapt = _smooth(r_adaptive, window)
    s_frz   = _smooth(r_frozen,   window)
    ax2.plot(np.arange(len(s_adapt)), s_adapt,
             color="#1565C0", lw=2, label=f"Hebbian (adaptive)")
    ax2.plot(np.arange(len(s_frz)),   s_frz,
             color="#B71C1C", lw=2, ls="--", label="Frozen weights")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel(f"Reward  ({window}-step MA)")
    ax2.set_title("Per-step Reward (smoothed)", fontweight="bold")
    ax2.legend()

    plt.tight_layout()
    out_path = join(out_dir, f"hebbian_adaptation_curve_{terrain}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _find_run_dir(checkpoint_dir: str) -> Optional[str]:
    """Walk up from checkpoint_dir to find the seed-level run root.

    Recognises a run root by the presence of training_score.txt or
    run_metadata.json.  Searches up to 4 levels.
    """
    candidate = os.path.abspath(checkpoint_dir)
    for _ in range(4):
        if (os.path.isfile(join(candidate, "training_score.txt"))
                or os.path.isfile(join(candidate, "run_metadata.json"))):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    return checkpoint_dir  # fallback: treat the given dir as the run root


def _terrain_env(world, terrain: str):
    """Return (env_id, world_file) tuple for the given terrain name."""
    mapping = {
        "flat": ("FlatEnv-v0", world.flat_world_file),
        "ice":  ("IceEnv-v0",  world.ice_world_file),
        "hill": ("HillEnv-v0", world.hill_world_file),
    }
    return mapping.get(terrain.lower(), mapping["flat"])


def _import_final_world():
    """Lazy import of FinalWorld to avoid pulling in MuJoCo at module import time."""
    _setup_gl()
    from evorob.utils.filesys import get_project_root  # noqa: PLC0415
    root = str(get_project_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from final_project_train import FinalWorld  # noqa: PLC0415
    return FinalWorld


def _setup_gl() -> None:
    """Set a sensible MuJoCo GL backend if not already configured."""
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "glfw" if platform.system() == "Darwin" else "egl"
    if "PYOPENGL_PLATFORM" not in os.environ and platform.system() != "Darwin":
        os.environ["PYOPENGL_PLATFORM"] = "egl"


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def generate_all_plots(run_dir: str, out_dir: Optional[str] = None) -> None:
    """Generate every applicable plot for one run.

    Hebbian-specific plots (3, 5, 7) are silently skipped for NN runs.
    Plots that require ``population_fitnesses_final.npy`` (4, 6) are skipped
    for runs predating 2026-05-27 and a warning is printed.

    Args:
        run_dir: Seed directory, e.g. ``results/body_brain_023/seed_0``.
        out_dir: Where to save plots.  Defaults to ``{run_dir}/plots/``.
    """
    if out_dir is None:
        out_dir = join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    run_name = os.path.basename(os.path.normpath(run_dir))
    print(f"\n{'='*60}")
    print(f"  Generating all plots for: {run_name}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    meta     = load_run_metadata(run_dir)
    ctrl     = meta.get("controller", "nn")
    last_ckpt = get_last_checkpoint_dir(run_dir)

    # -- 1. Per-terrain fitness curves -----------------------------------
    print("\n[1/7] Per-terrain fitness curves")
    plot_per_terrain_fitness([run_dir], out_dir)

    # -- 2. NN vs Hebbian (requires both — handled by generate_comparison_plots) --
    print("\n[2/7] NN vs Hebbian  ← use generate_comparison_plots() for cross-controller overlay")

    # -- 3. Weight trajectory (Hebbian only) -----------------------------
    if ctrl == "hebbian" and last_ckpt:
        for t in ["flat", "ice", "hill"]:
            print(f"\n[3/7] Hebbian weight trajectory ({t})")
            plot_weight_trajectory(last_ckpt, out_dir, t)
    else:
        print("\n[3/7] Weight trajectory — skipped (NN controller)")

    # -- 4. Per-terrain Pareto fronts ------------------------------------
    print("\n[4/7] Per-terrain Pareto fronts")
    plot_terrain_pareto_fronts(run_dir, out_dir)

    # -- 5. ABCD heatmap (Hebbian only) ----------------------------------
    if ctrl == "hebbian" and last_ckpt:
        print("\n[5/7] ABCD coefficient heatmap")
        plot_abcd_heatmap(last_ckpt, out_dir)
    else:
        print("\n[5/7] ABCD heatmap — skipped (NN controller)")

    # -- 6. Fitness distributions ----------------------------------------
    print("\n[6/7] Fitness distributions")
    plot_fitness_distributions({ctrl: [run_dir]}, out_dir)

    # -- 7. Adaptation curve (Hebbian only) ------------------------------
    if ctrl == "hebbian" and last_ckpt:
        for t in ["flat", "hill"]:
            print(f"\n[7/7] Adaptation curve ({t})")
            plot_adaptation_curve(last_ckpt, out_dir, t)
    else:
        print("\n[7/7] Adaptation curve — skipped (NN controller)")

    print(f"\n✓  Done.  All plots saved to: {out_dir}\n")


def generate_comparison_plots(
    nn_dirs: Sequence[str],
    hebb_dirs: Sequence[str],
    out_dir: str,
) -> None:
    """Plots 2 and 6 with both controller types on the same figure.

    Args:
        nn_dirs:   NN run directories (may be empty).
        hebb_dirs: Hebbian run directories (may be empty).
        out_dir:   Where to save plots.
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nGenerating comparison plots → {out_dir}")

    print("\n[2] NN vs Hebbian fitness overlay")
    plot_nn_vs_hebbian(nn_dirs, hebb_dirs, out_dir)

    dirs_by_ctrl: Dict[str, List[str]] = {}
    if nn_dirs:
        dirs_by_ctrl["nn"] = list(nn_dirs)
    if hebb_dirs:
        dirs_by_ctrl["hebbian"] = list(hebb_dirs)

    print("\n[6] Fitness distributions")
    plot_fitness_distributions(dirs_by_ctrl, out_dir)

    print(f"\n✓  Done.  Comparison plots saved to: {out_dir}\n")
