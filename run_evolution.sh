#!/bin/bash
#SBATCH --job-name=evorob_body_brain
#SBATCH --output=logs/evorob_%j.out
#SBATCH --error=logs/evorob_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=academic
#SBATCH --account=micro-515

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate evorob

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export SLURM_CPUS_PER_TASK=64

mkdir -p logs
cd /home/hechler/ER_course/micro-515-EvoRob
python -u final_project_train.py