#!/bin/bash
# Run this on the LOGIN NODE: bash run_evolution_body_brain.sh
# Co-evolves body morphology (leg lengths) alongside the controller.
# Detects the next available run ID, then submits a SLURM array job (seeds 0-2).
# Results land in: results/body_brain_NNN/seed_{0,1,2}/

RUN_NAME="body_brain"
RESULTS_BASE="results"

# --- Auto-detect next available run ID ---
NEXT_ID=1
if [ -d "${RESULTS_BASE}" ]; then
    LAST=$(ls -d "${RESULTS_BASE}/${RUN_NAME}_"[0-9][0-9][0-9] 2>/dev/null \
           | grep -oE '[0-9]{3}$' | sort -n | tail -1)
    [ -n "$LAST" ] && NEXT_ID=$((10#${LAST} + 1))
fi
ID_STR=$(printf "%03d" "${NEXT_ID}")
FULL_NAME="${RUN_NAME}_${ID_STR}"

echo "Submitting ${FULL_NAME} → ${RESULTS_BASE}/${FULL_NAME}/seed_{0,1,2}/"
mkdir -p logs

sbatch \
    --job-name="${FULL_NAME}" \
    --output="logs/${FULL_NAME}_%a.out" \
    --error="logs/${FULL_NAME}_%a.err" \
    << SBATCH_SCRIPT
#!/bin/bash
#SBATCH --array=0-2%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=academic
#SBATCH --account=micro-515

source \$HOME/miniconda3/etc/profile.d/conda.sh
conda activate evorob

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export SLURM_CPUS_PER_TASK=64

cd /home/hechler/ER_course/micro-515-EvoRob

python -u final_project_train.py \
    --results-dir "${RESULTS_BASE}/${FULL_NAME}" \
    --seed "\${SLURM_ARRAY_TASK_ID}" \
    --co-evolve-body
SBATCH_SCRIPT
