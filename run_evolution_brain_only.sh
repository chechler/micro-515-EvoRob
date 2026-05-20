#!/bin/bash
# Run this on the LOGIN NODE: bash run_evolution_brain_only.sh
# Evolves the controller only; body uses fixed default leg lengths (0.35 m).
# Submits a single job (seed 0) using a temp script file — no CLI flags to sbatch.
# Results land in: results/brain_only_NNN/seed_0/

RUN_NAME="brain_only"
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

echo "Submitting ${FULL_NAME} → ${RESULTS_BASE}/${FULL_NAME}/seed_0/"
mkdir -p logs

TMPSCRIPT=$(mktemp /tmp/slurm_XXXXXX.sh)

# Static directives (single-quoted — no expansion)
cat > "${TMPSCRIPT}" << 'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=academic
#SBATCH --account=micro-515
EOF

# Dynamic directives (double-quoted — variables expand now)
cat >> "${TMPSCRIPT}" << EOF
#SBATCH --job-name=${FULL_NAME}
#SBATCH --output=logs/${FULL_NAME}.out
#SBATCH --error=logs/${FULL_NAME}.err

source \$HOME/miniconda3/etc/profile.d/conda.sh
conda activate evorob

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export SLURM_CPUS_PER_TASK=64

cd /home/hechler/ER_course/micro-515-EvoRob
python -u final_project_train.py \\
    --results-dir "${RESULTS_BASE}/${FULL_NAME}" \\
    --seed "0" \\
    --no-co-evolve-body
EOF

echo "--- Script to be submitted ---"
cat "${TMPSCRIPT}"
echo "------------------------------"

sbatch "${TMPSCRIPT}"
rm "${TMPSCRIPT}"
