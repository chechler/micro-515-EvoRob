"""
MICRO-515 Final Project — Multi-task Robot Evolution
=====================================================
Evolve a legged robot (body + controller) to walk in the +x direction across
three training environments simultaneously.  The genotype encodes both the
neural controller weights and the body morphology (leg lengths).

Training environments (3 objectives)
-------------------------------------
1  Flat  — standard ground, good friction  (FlatEnv-v0  / flat_world.xml)
2  Ice   — slippery ground, low friction   (IceEnv-v0   / ice_world.xml)
3  Hill  — procedural hilly terrain        (HillEnv-v0  / hill_world.xml)

The evaluation terrain is separate and fixed.  Students test their best
evolved robot on it using final_project_test.py — it is not trained on.
"""

import os
import platform

if "MUJOCO_GL" not in os.environ:
    if platform.system() == "Darwin":
        os.environ["MUJOCO_GL"] = "glfw"
    else:
        os.environ["MUJOCO_GL"] = "osmesa"

import copy
import multiprocessing
import shutil
import time
import xml.etree.ElementTree as xml
from concurrent.futures import ProcessPoolExecutor
from os.path import join
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
import scipy.ndimage
from PIL import Image
from gymnasium.vector import SyncVectorEnv

import evorob.world                         # registers EvalEnv-v0
from evorob.algorithms.nsga import NSGAII
from evorob.utils.filesys import get_last_checkpoint_dir, get_project_root
from evorob.world.base import World
from evorob.world.robot.controllers.mlp_hebbian import HebbianController
from evorob.world.robot.morphology.ant_custom_robot import AntRobot

ROOT_DIR = get_project_root()
_ASSETS  = join(ROOT_DIR, "evorob", "world", "robot", "assets")
MAX_EPISODE_STEPS = 1000  # fixed for leaderboard — do not change


# ---------------------------------------------------------------------------
# FinalWorld — body + brain co-evolution across multiple terrains
# ---------------------------------------------------------------------------

class FinalWorld(World):
    """Translates a genotype into a robot phenotype and evaluates it.

    The genotype is a 1-D array: [controller_params | body_params].
    Each call to evaluate_individual generates the robot body XML, injects it
    into every terrain template, then runs the controller in parallel episodes.
    """

    def __init__(self):
        # Choose your controller — swap for your own MLP, SO2Controller, Hebbian, or custom.
        # Whatever you choose determines self.n_weights (controller parameter count).
        #
        # from evorob.world.robot.controllers.mlp import NeuralNetworkController  # your impl
        # from evorob.world.robot.controllers.so2 import SO2Controller
        # self.controller = SO2Controller(input_size=27, output_size=8, hidden_size=8)
        self.controller = HebbianController(
            input_size=27, output_size=8, hidden_size=8
        )

        self.n_weights     = self.controller.n_params
        self.n_body_params = 2          # 1 upper + 1 lower length, shared across all 4 legs
        self.n_params      = self.n_weights + self.n_body_params

        # Temporary directory holds AntRobot.xml + one combined world XML per terrain
        self.temp_dir        = TemporaryDirectory()
        self.flat_world_file = join(self.temp_dir.name, "WorldFlat.xml")
        self.ice_world_file  = join(self.temp_dir.name, "WorldIce.xml")
        self.hill_world_file = join(self.temp_dir.name, "WorldHill.xml")
        self.world_file      = self.hill_world_file  # default for visualisation

        # Joint geometry — matches the AntRobot topology
        self.joint_limits = [
            [-30, 30], [30, 70],
            [-30, 30], [30, 70],
            [-30, 30], [30, 70],
            [-30, 30], [30, 70],
        ]
        self.joint_axis = [
            [0, 0, 1], [-1,  1, 0],   # front_left  hip, knee
            [0, 0, 1], [-1, -1, 0],   # front_right hip, knee
            [0, 0, 1], [ 1, -1, 0],   # back_left   hip, knee
            [0, 0, 1], [ 1,  1, 0],   # back_right  hip, knee
        ]

        # Custom sensor function — intercepts the raw env observation before it
        # reaches the controller.  Set to any callable obs -> obs' to filter,
        # augment, or reshape observations.  The controller input_size must match
        # the output of this function.
        #
        # Example — use only joint angles and velocities (14 values):
        #   self.sensor_fn = lambda obs: obs[:14]
        #   self.controller = NeuralNetworkController(input_size=14, ...)
        self.sensor_fn = None

        self._create_terrain_file("terrain.png")

        # Parse terrain templates once — deepcopy them per individual instead of re-reading disk
        self._flat_tmpl = xml.parse(join(_ASSETS, "flat_world.xml")).getroot()
        self._ice_tmpl  = xml.parse(join(_ASSETS, "ice_world.xml")).getroot()
        self._hill_tmpl = xml.parse(join(_ASSETS, "hill_world.xml")).getroot()

    # ------------------------------------------------------------------
    # Genotype → phenotype
    # ------------------------------------------------------------------

    def geno2pheno(self, genotype: np.ndarray):
        """Decode genotype into controller weights and body parameters.

        Splits genotype into:
          genotype[:n_weights]  → controller (scaled by 0.1 before loading)
          genotype[n_weights:]  → 2 shared leg-segment lengths (upper, lower) via (g+1)/4 + 0.1

        Returns (points, connectivity_mat) for AntRobot construction.
        """
        control_params = genotype[:self.n_weights] * 0.1
        body_params    = (genotype[self.n_weights:] + 1) / 4 + 0.1
        self.controller.geno2pheno(control_params)

        upper_leg, lower_leg = body_params  # shared across all 4 legs
        u = np.sqrt(0.5) * upper_leg        # diagonal component for upper segment
        l = np.sqrt(0.5) * lower_leg        # diagonal component for lower segment

        # Define the 3D coordinates of the relative tree structure.
        # All four legs use the same upper_leg and lower_leg lengths.
        front_left_hip_xyz   = np.array([ 0.2,  0.2, 0])
        front_left_knee_xyz  = front_left_hip_xyz  + np.array([ u,  u, 0])
        front_left_toe_xyz   = front_left_knee_xyz + np.array([ l,  l, 0])

        front_right_hip_xyz  = np.array([-0.2,  0.2, 0])
        front_right_knee_xyz = front_right_hip_xyz + np.array([-u,  u, 0])
        front_right_toe_xyz  = front_right_knee_xyz + np.array([-l,  l, 0])

        back_left_hip_xyz    = np.array([-0.2, -0.2, 0])
        back_left_knee_xyz   = back_left_hip_xyz   + np.array([-u, -u, 0])
        back_left_toe_xyz    = back_left_knee_xyz  + np.array([-l, -l, 0])

        back_right_hip_xyz   = np.array([ 0.2, -0.2, 0])
        back_right_knee_xyz  = back_right_hip_xyz  + np.array([ u, -u, 0])
        back_right_toe_xyz   = back_right_knee_xyz + np.array([ l, -l, 0])

        points = np.vstack([front_left_hip_xyz,
                            front_left_knee_xyz,
                            front_left_toe_xyz,
                            front_right_hip_xyz,
                            front_right_knee_xyz,
                            front_right_toe_xyz,
                            back_left_hip_xyz,
                            back_left_knee_xyz,
                            back_left_toe_xyz,
                            back_right_hip_xyz,
                            back_right_knee_xyz,
                            back_right_toe_xyz,
                            ])

        # define the type of connections [FIXED ARCHITECTURE]
        connectivity_mat = np.array(
            [[150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ]
        )
        return points, connectivity_mat

    # ------------------------------------------------------------------
    # Robot XML generation
    # ------------------------------------------------------------------

    def update_robot_xml(self, genotype: np.ndarray) -> None:
        """Build robot body XML from genotype and inject into every terrain template.

        Writes AntRobot.xml to temp_dir, then creates one combined world XML per
        terrain (flat, ice, hill) by appending an <include> to the template.
        """
        points, connectivity_mat = self.geno2pheno(genotype)
        robot = AntRobot(
            points, connectivity_mat, self.joint_limits, self.joint_axis,
            name="Robot", verbose=False,
        )
        robot.xml = robot.define_robot()
        robot.write_xml(self.temp_dir.name)          # → Robot.xml

        for tmpl_root, world_file in [
            (self._flat_tmpl, self.flat_world_file),
            (self._ice_tmpl,  self.ice_world_file),
            (self._hill_tmpl, self.hill_world_file),
        ]:
            root = copy.deepcopy(tmpl_root)
            root.append(xml.Element("include", attrib={"file": "Robot.xml"}))
            with open(world_file, "w") as f:
                f.write(xml.tostring(root, encoding="unicode"))

    def _create_terrain_file(self, filename: str, width: int = 200, depth: int = 400):
        """Hill terrain PNG: smooth start, bumpy middle, smooth end."""
        slope_deg = 5.0
        bump_scale = 0.08
        sigma = 4.0

        rise = np.tan(np.deg2rad(slope_deg))
        x = np.linspace(0, 1, depth)
        y = np.linspace(0, 1, width)
        X, Y = np.meshgrid(x, y)

        # Smooth slope: starts flat, gradually rises
        slope_map = np.clip(X * rise, 0, 1)

        # Bell-shaped envelope: smooth at both ends, bumpy in the middle
        rng = np.random.default_rng(42)
        noise = rng.uniform(0, 1, (width, depth))
        bump_envelope = np.sin(np.pi * X)  # 0 at start, peaks at mid, 0 at end
        noise = scipy.ndimage.gaussian_filter(noise, sigma=sigma)
        noise = (noise - noise.min()) / (noise.max() - noise.min()) * bump_envelope
        noise_map = noise * bump_scale

        terrain = np.clip(slope_map + noise_map, 0, 1)
        terrain[-1, -1] = 1  # ensure max value for normalization

        img = Image.fromarray((terrain * 255).astype(np.uint8), mode="L")
        img.save(join(self.temp_dir.name, filename))

    def create_env(self, render_mode: str = "rgb_array", **kwargs):
        """Return a HillEnv-v0 instance (used for visualisation)."""
        return gym.make("HillEnv-v0", robot_path=self.hill_world_file,
                        render_mode=render_mode, **kwargs)

    # ------------------------------------------------------------------
    # Combined fitness for NSGA-II
    # ------------------------------------------------------------------

    def evaluate_individual(self, genotype: np.ndarray,
                            n_repeats: int = 4, n_steps: int = 500) -> np.ndarray:
        """Evaluate one genotype on all three terrains in a single SyncVectorEnv.

        Creates n_repeats*3 environments at once (flat×n_repeats, ice×n_repeats,
        hill×n_repeats), runs them simultaneously, then splits rewards by terrain.
        This eliminates two MuJoCo init/teardown cycles compared to calling
        separate SyncVectorEnvs per terrain.

        Returns a 1-D array of three objective values: [flat, ice, hill].
        """
        self.update_robot_xml(genotype)

        terrain_configs = [
            ("FlatEnv-v0", self.flat_world_file),
            ("IceEnv-v0",  self.ice_world_file),
            ("HillEnv-v0", self.hill_world_file),
        ]
        n_envs = len(terrain_configs) * n_repeats

        env_fns = [
            (lambda eid, wf: lambda: gym.make(
                eid, robot_path=wf, max_episode_steps=n_steps
            ))(env_id, wf)
            for env_id, wf in terrain_configs
            for _ in range(n_repeats)
        ]

        envs = SyncVectorEnv(env_fns)
        # Reduce MuJoCo solver iterations 100→20: stable for ant, 5× faster physics.
        for env in envs.envs:
            env.unwrapped.model.opt.iterations = 20

        self.controller.reset_controller(batch_size=n_envs)

        rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        obs, _ = envs.reset()
        if self.sensor_fn is not None:
            obs = self.sensor_fn(obs)
        done = np.zeros(n_envs, dtype=bool)

        for t in range(n_steps):
            actions = np.where(done[:, None], 0, self.controller.get_action(obs))
            obs, r, terminated, truncated, _ = envs.step(actions)
            if self.sensor_fn is not None:
                obs = self.sensor_fn(obs)
            rewards[t, ~done] = r[~done]
            done |= terminated | truncated
            if done.all():
                break

        envs.close()
        total = rewards.sum(axis=0)
        return np.array([
            total[0           :   n_repeats].mean(),
            total[  n_repeats : 2*n_repeats].mean(),
            total[2*n_repeats : 3*n_repeats].mean(),
        ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Neutral leaderboard evaluation  (TA-graded — do not modify)
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_dir: str,
    output_dir: str = "evaluation_output",
    n_episodes: int = 256,          # set to 256 for submission; lower for testing
) -> dict | None:
    """Evaluate the best genotype from a checkpoint on all three training terrains.

    Loads x_best.npy, evaluates it on flat, ice, and hill for n_episodes each,
    prints per-episode scores, records one video per terrain, and writes a score file.

    Args:
        checkpoint_dir: Path to your NSGA-II checkpoint folder.
        output_dir:     Where to save the score file and videos.
        n_episodes:     Episodes per terrain (256 for submission).
    """
    MAX_STEPS = MAX_EPISODE_STEPS   # DO NOT CHANGE
    SEED      = 0                   # DO NOT CHANGE

    # --- Locate checkpoint ---
    last_gen = get_last_checkpoint_dir(checkpoint_dir)

    def _load(fname):
        for d in ([last_gen] if last_gen else []) + [checkpoint_dir]:
            p = join(d, fname)
            if os.path.isfile(p):
                return np.load(p, allow_pickle=True)
        return None

    x_best = _load("x_best.npy")
    if x_best is None:
        print(f"ERROR: x_best.npy not found in '{checkpoint_dir}'.")
        return None
    print(f"Loaded x_best  (shape: {x_best.shape})")

    world = FinalWorld()
    world.update_robot_xml(x_best)
    ctrl_name = type(world.controller).__name__
    print(f"Controller: {ctrl_name}  |  n_weights={world.n_weights}"
          f"  |  genotype size={world.n_params}\n")

    terrains = {
        "flat": ("FlatEnv-v0", world.flat_world_file),
        "ice":  ("IceEnv-v0",  world.ice_world_file),
        "hill": ("HillEnv-v0", world.hill_world_file),
    }

    def _neutral(info: dict) -> float:
        return (float(info.get("healthy_reward", 1.0))
                + float(info.get("x_position",   0.0))
                - float(info.get("ctrl_cost",     0.0))
                - float(info.get("cfrc_cost",     0.0)))

    def _stats(values: list) -> dict:
        arr = np.asarray(values)
        return dict(mean=float(arr.mean()), std=float(arr.std()),
                    best=float(arr.max()), worst=float(arr.min()), values=values)

    def _run(env_id: str, world_file: str) -> list:
        rng = np.random.default_rng(SEED)
        env = gym.make(env_id, robot_path=world_file, max_episode_steps=MAX_STEPS)
        rewards = []
        for ep in range(n_episodes):
            world.controller.reset_controller(batch_size=1)
            obs, _ = env.reset(seed=int(rng.integers(0, 2 ** 31)))
            total, done = 0.0, False
            while not done:
                action = world.controller.get_action(obs)
                if action.ndim > 1:
                    action = action.squeeze(0)
                obs, _, terminated, truncated, info = env.step(action)
                total += _neutral(info)
                done = terminated or truncated
            rewards.append(total)
        env.close()
        return rewards

    def _record(env_id: str, world_file: str, out_path: str) -> None:
        try:
            import imageio
            env = gym.make(env_id, robot_path=world_file,
                           render_mode="rgb_array", max_episode_steps=MAX_STEPS)
            world.controller.reset_controller(batch_size=1)
            obs, _ = env.reset(seed=SEED)
            frames = []
            for _ in range(MAX_STEPS):
                frames.append(env.render())
                action = world.controller.get_action(obs)
                if action.ndim > 1:
                    action = action.squeeze(0)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            env.close()
            imageio.mimwrite(out_path, frames, fps=20)
            print(f"  Video: {out_path}")
        except Exception as exc:
            print(f"  Video skipped: {exc}")

    # --- Evaluate on each terrain ---
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for terrain_name, (env_id, world_file) in terrains.items():
        print(f"  Running {terrain_name}  ({n_episodes} episodes)...", flush=True)
        results[terrain_name] = _stats(_run(env_id, world_file))

    # Per-episode 3-column table
    t_names = list(results.keys())
    col_w = 12
    hdr = f"  {'Ep':>4}   " + "   ".join(f"{n.capitalize():>{col_w}}" for n in t_names)
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)
    for ep in range(n_episodes):
        row = f"  {ep + 1:>4}   " + "   ".join(
            f"{results[n]['values'][ep]:>{col_w}.2f}" for n in t_names
        )
        print(row)
    print(sep)
    print(f"  {'mean':>4}   " + "   ".join(
        f"{results[n]['mean']:>{col_w}.2f}" for n in t_names
    ))
    print(f"  {'std':>4}   " + "   ".join(
        f"{results[n]['std']:>{col_w}.2f}" for n in t_names
    ))
    print()

    # --- Record one video per terrain ---
    print("Recording videos...")
    for terrain_name, (env_id, world_file) in terrains.items():
        _record(env_id, world_file, join(output_dir, f"evaluation_{terrain_name}.mp4"))

    # --- Score file ---
    score_path = join(output_dir, "evaluation_score.txt")
    col = 60
    with open(score_path, "w") as f:
        f.write("=" * col + "\n")
        f.write("MICRO-515 Final Project — Evaluation Results\n")
        f.write("=" * col + "\n\n")
        f.write(f"Controller      : {ctrl_name} ({world.n_weights} params)\n")
        f.write(f"Genotype size   : {world.n_params}"
                f"  (controller={world.n_weights}, body={world.n_body_params})\n")
        f.write(f"Checkpoint      : {checkpoint_dir}\n")
        f.write(f"Episodes/terrain: {n_episodes}\n")
        f.write(f"Reward          : healthy_reward + x_position - ctrl_cost - cfrc_cost\n\n")

        f.write("=" * col + "\n")
        f.write("SUMMARY\n")
        f.write("=" * col + "\n")
        f.write(f"{'Terrain':<8} {'Mean':>9} {'Std':>8} {'Best':>9} {'Worst':>9}\n")
        f.write("-" * col + "\n")
        for terrain_name, r in results.items():
            f.write(f"{terrain_name:<8} {r['mean']:9.2f} {r['std']:8.2f}"
                    f" {r['best']:9.2f} {r['worst']:9.2f}\n")
        f.write("\n")

        for terrain_name, r in results.items():
            f.write("-" * 50 + "\n")
            f.write(f"{terrain_name.upper()} — Per-episode rewards\n")
            f.write("-" * 50 + "\n")
            for i, v in enumerate(r["values"]):
                f.write(f"  Episode {i + 1:3d}: {v:10.2f}\n")
            f.write("\n")

    print(f"\nScore saved to: {score_path}")
    print("=" * col)
    for terrain_name, r in results.items():
        print(f"  {terrain_name:<6}: {r['mean']:8.2f} ± {r['std']:7.2f}"
              f"  best={r['best']:.2f}  worst={r['worst']:.2f}")
    print("=" * col)
    return results


# ---------------------------------------------------------------------------
# Parallel evaluation helpers  (module-level so multiprocessing can pickle them)
# ---------------------------------------------------------------------------

# One FinalWorld per worker process — created by the pool initializer, reused
# across all tasks that land on the same worker (avoids repeating temp-dir +
# terrain-PNG setup for every individual).
_worker_world: "FinalWorld | None" = None


def _init_worker() -> None:
    # Set GL env vars before any MuJoCo context is created in this process.
    # Critical for spawn mode; harmless (and explicit) for fork mode.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    global _worker_world
    _worker_world = FinalWorld()


def _eval_individual_parallel(args: tuple) -> tuple:
    """Evaluate one genotype and return (fitness_array, robot_xml_str)."""
    genotype, n_repeats, n_steps = args
    fitness = _worker_world.evaluate_individual(genotype, n_repeats, n_steps)
    robot_xml_path = join(_worker_world.temp_dir.name, "Robot.xml")
    xml_str = None
    if os.path.isfile(robot_xml_path):
        with open(robot_xml_path) as fh:
            xml_str = fh.read()
    return fitness, xml_str


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_multi_task_evolution(
    num_generations: int = 200,
    population_size: int = 100,
    n_parents:       int = 50,
    n_repeats:       int = 3,
    n_steps:         int = 500,
    mutation_prob:   float = 0.3,
    crossover_prob:  float = 0.5,
    bounds:          tuple = (-1, 1),
    ckpt_interval:   int = 25,
    results_dir:     str = None,
    random_seed:     int = 42,
) -> None:
    np.random.seed(random_seed)

    world = FinalWorld()
    print(f"Genotype : {world.n_params} params"
          f"  (controller={world.n_weights}, body={world.n_body_params})")

    if results_dir is None:
        scratch = os.environ.get("SCRATCH")
        if scratch:
            results_dir = os.path.join(scratch, "micro-515-EvoRob", "results", "final_project")
        else:
            results_dir = join(ROOT_DIR, "results", "final_project")

    ea = NSGAII(
        population_size=population_size,
        n_opt_params=world.n_params,
        n_parents=n_parents,
        num_generations=num_generations,
        bounds=bounds,
        mutation_prob=mutation_prob,
        crossover_prob=crossover_prob,
        output_dir=results_dir,
    )

    n_obj = 3
    print(f"\nRunning {num_generations} generations  pop={population_size}")
    print(f"Objectives : [flat, ice, hill]")
    print(f"Checkpoints: {results_dir}\n")

    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)
    _best_xml_stage = join(results_dir, "_best_robot.xml")  # staging copy of best robot
    _best_scalar = -np.inf

    # One worker process per allocated core. On SLURM, SLURM_CPUS_PER_TASK is
    # the correct limit; os.cpu_count() returns the full node count and causes
    # oversubscription. fork is used on Linux (much faster — no reimport of
    # Python/MuJoCo per worker); spawn is required on macOS.
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    n_workers = min(population_size, slurm_cpus)
    print(f"Workers : n_workers={n_workers}  population_size={population_size}"
          f"  SLURM_CPUS_PER_TASK={slurm_cpus}", flush=True)
    mp_ctx = multiprocessing.get_context(
        "spawn" if platform.system() == "Darwin" else "fork"
    )

    t_run_start = time.perf_counter()
    gen_times: list[float] = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=mp_ctx,
        initializer=_init_worker,
    ) as executor:
        for gen in range(num_generations):
            t_gen0 = time.perf_counter()
            pop = ea.ask()
            fitnesses = np.empty((len(pop), n_obj))

            results = list(executor.map(
                _eval_individual_parallel,
                [(g, n_repeats, n_steps) for g in pop],
            ))

            for idx, (fitness, xml_str) in enumerate(results):
                fitnesses[idx] = fitness
                scalar = float(fitness.sum())
                if scalar > _best_scalar:
                    _best_scalar = scalar
                    if xml_str is not None:
                        with open(_best_xml_stage, "w") as fh:
                            fh.write(xml_str)

            ea.tell(pop, fitnesses, save_checkpoint=False)

            gen_time = time.perf_counter() - t_gen0
            gen_times.append(gen_time)
            avg_gen = sum(gen_times) / len(gen_times)
            gens_left = num_generations - (gen + 1)
            eta_s = avg_gen * gens_left
            elapsed_s = time.perf_counter() - t_run_start

            # Per-generation log — flushed immediately so SLURM log stays current
            terrain_labels = ["flat", "ice ", "hill"]
            best_scalar_this_gen = float(fitnesses.sum(axis=1).max())
            print(f"\n=== Gen {gen+1}/{num_generations}  best_sum={best_scalar_this_gen:+.1f}"
                  f"  |  gen={gen_time:.0f}s  elapsed={elapsed_s/60:.1f}m"
                  f"  ETA={eta_s/60:.1f}m ===", flush=True)
            for label, col in zip(terrain_labels, fitnesses.T):
                mean_f    = float(col.mean())
                pct_alive = float((col > 0).mean()) * 100
                best_f    = float(col.max())
                print(f"  {label}: mean={mean_f:+8.1f}  ({pct_alive:3.0f}% alive)  best={best_f:+8.1f}", flush=True)
            if gen % ckpt_interval == 0:
                gen_dir = join(results_dir, str(gen))
                os.makedirs(gen_dir, exist_ok=True)
                np.save(join(gen_dir, "x_best"), ea.x_best_so_far)
                np.save(join(gen_dir, "f_best"), ea.f_best_so_far)
                if os.path.isfile(_best_xml_stage):
                    shutil.copy2(_best_xml_stage, join(gen_dir, "Robot.xml"))

    # --- Training summary ---
    best_f = ea.f_best_so_far  # shape (3,) for NSGA-II
    score_path = join(results_dir, "training_score.txt")
    with open(score_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MICRO-515 Final Project — Training Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generations     : {num_generations}\n")
        f.write(f"Population size : {population_size}\n")
        f.write(f"Controller      : {type(world.controller).__name__}"
                f"  ({world.n_weights} params)\n")
        f.write(f"Genotype size   : {world.n_params}"
                f"  (controller={world.n_weights}, body={world.n_body_params})\n\n")
        f.write("Best individual (highest sum of objectives):\n")
        labels = ["flat", "ice", "hill"]
        for label, val in zip(labels, best_f):
            f.write(f"  {label:<6}: {float(val):10.2f}\n")
        f.write(f"  {'sum':<6}: {float(best_f.sum()):10.2f}\n")
    print(f"\nTraining summary saved to: {score_path}")


if __name__ == "__main__":
    run_multi_task_evolution(
        num_generations=200,
        population_size=100,
        n_parents=50,
        n_repeats=3,
        n_steps=500,
        ckpt_interval=25,
        results_dir=join(ROOT_DIR, "results", "final_project"),
    )
