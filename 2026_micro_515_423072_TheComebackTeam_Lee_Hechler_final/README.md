# MICRO-515 Final Project — Submission README

## Submission contents

| File | Description |
|------|-------------|
| `final_project_train.py` | Main training script — defines FinalWorld, co-evolution loop, NSGA-II wiring |
| `x_best.npy` | Best evolved genotype (282 params: 280 controller + 2 leg lengths) |
| `Robot.xml` | MuJoCo robot body XML matching `x_best.npy` (same individual) |
| `mlp.py` | NeuralNetworkController source (input=27, output=8, hidden=8, 280 params) |
| `README.md` | This file |

## How to run the compatibility check

```bash
python final_project_test.py \
    --best_dir_path 2026_micro_515_423072_TheComebackTeam_Lee_Hechler_final/
```

`x_best.npy` and `Robot.xml` are in the submission folder root, which
`load_from_checkpoint` resolves correctly when no numbered sub-directories
are present.

## Environment setup
- Python: 3.12


```bash
python final_project_test.py --best_dir_path 2026_micro_515_423072_TheComebackTeam_Lee_Hechler_final/
```

## Approach

We co-evolved the ant's morphology (upper and lower leg lengths, shared across all
four legs) together with a neural network controller (NeuralNetworkController,
27 inputs, 8 outputs, 8 hidden units, tanh activations) using NSGA-II over three
simultaneous terrain objectives: flat, ice, and hill. The genotype is a 282-dimensional
vector whose first 280 entries are controller weights and last 2 are leg-length scalars.
Each generation, all 200 individuals are evaluated in parallel across all three terrains
(5 repeats × 500 steps each per terrain). Pareto-front ranking with crowding distance
maintains diversity across the three objectives. The best individual (selected by minimum
fitness across terrains, enforcing balanced locomotion on every surface) is checkpointed
every 25 generations. Run 018 (K_EXP=0.20 exponential position reward, 500 generations)
produced the submitted robot, achieving approximately flat=2715, ice=2770, hill=2594
on training rewards.
