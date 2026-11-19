#!/bin/bash
#SBATCH -t 00:40:00
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH --job-name=per_maint
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Pereira audit job: recover finished work from checkpoints -> completeness audit.
# Submitted as the final (CPU-only) stage by run_all_analyses.sh; can also be run
# directly with: sbatch other/pereira_maintenance.sh. Safe to re-run: it only
# converts existing checkpoints into CSVs and never recomputes anything.

ENV_SETUP="${BA_ENV_SETUP:-pipeline/env_setup.sh}"
[ -f "${SLURM_SUBMIT_DIR:-.}/$ENV_SETUP" ] && ENV_SETUP="${SLURM_SUBMIT_DIR:-.}/$ENV_SETUP"
[ -f "$ENV_SETUP" ] || { echo "ERROR: $ENV_SETUP not found (copy pipeline/env_setup.example.sh)"; exit 1; }
# shellcheck disable=SC1090
source "$ENV_SETUP"

export PYTHONUNBUFFERED=1
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
echo "host=$(hostname)  job=${SLURM_JOB_ID:-}  dir=$PWD"
echo

echo "############ 1. RECOVER FINISHED WORK FROM CHECKPOINTS ############"
echo "  Converts checkpoint -> CSV. No recomputation, no GPU."
python other/flush_pereira_checkpoints.py
echo
echo "############ 2. COMPLETENESS AUDIT ############"
python other/check_pereira_completeness.py
