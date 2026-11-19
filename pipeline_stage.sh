#!/bin/bash
# Unified SLURM array worker for the basis-alignment pipeline.
#
# One array task = one line of the manifest named by $MANIFEST. Do not sbatch
# this directly; run_all_analyses.sh submits it with the right --array range,
# concurrency cap and inter-stage dependencies.
#
#   MANIFEST=pipeline/stage_all.txt sbatch --array=0-21%6 pipeline_stage.sh
#
# Every underlying script has checkpoint/resume, so re-running a stage is safe
# and only recomputes what is missing.

#SBATCH --gres=gpu:1             # paper runs: 1x A100 80GB
#SBATCH -t 2-00:00:00
#SBATCH -c 8
#SBATCH --mem=36G
#SBATCH --job-name=ba_stage
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -uo pipefail

# Cluster-specific setup (modules, conda env, HF cache, partition-independent
# paths) lives in pipeline/env_setup.sh, which every user edits for their own
# cluster (template: pipeline/env_setup.example.sh). It must activate a Python
# environment with the packages in environment.yml and may set HF_HOME.
ENV_SETUP="${BA_ENV_SETUP:-pipeline/env_setup.sh}"
[ -f "${SLURM_SUBMIT_DIR:-.}/$ENV_SETUP" ] && ENV_SETUP="${SLURM_SUBMIT_DIR:-.}/$ENV_SETUP"
[ -f "$ENV_SETUP" ] || { echo "ERROR: $ENV_SETUP not found (copy pipeline/env_setup.example.sh)"; exit 1; }
# shellcheck disable=SC1090
source "$ENV_SETUP"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_OFFLOAD_DIR="${HF_OFFLOAD_DIR:-${HF_HOME}/offload}"
mkdir -p "$HF_OFFLOAD_DIR"
# HF token (gated Llama/Gemma weights) comes from the submitting environment
# (run_all_analyses.sh submits with --export=ALL) or from `huggingface-cli login`.
# Never hardcode it here.
if [ -z "${HF_TOKEN:-}" ] && [ ! -s "${HF_HOME}/token" ] && [ ! -s "$HOME/.cache/huggingface/token" ]; then
    echo "ERROR: no Hugging Face token: export HF_TOKEN=... before ./run_all_analyses.sh"; exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"   # repo root (run_all_analyses.sh submits from it)
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }

: "${MANIFEST:?MANIFEST env var not set - submit via run_all_analyses.sh}"
[ -f "$MANIFEST" ] || { echo "ERROR: manifest $MANIFEST not found"; exit 1; }

# Index the Nth REAL job line: manifests carry comments and blank lines, so a
# plain `sed -n Np` would silently run the wrong job.
LINE=$(grep -v '^[[:space:]]*#' "$MANIFEST" | grep -v '^[[:space:]]*$' \
       | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [ -z "$LINE" ]; then
    echo "[task ${SLURM_ARRAY_TASK_ID}] no job line at this index in $MANIFEST - nothing to do"
    exit 0
fi

echo "=========================================================================="
echo " manifest : $MANIFEST"
echo " task     : ${SLURM_ARRAY_TASK_ID}  (job ${SLURM_ARRAY_JOB_ID})"
echo " line     : $LINE"
echo " host     : $(hostname)   started $(date)"
echo "=========================================================================="

read -r TYPE REST <<< "$LINE"
START=$SECONDS

case "$TYPE" in
    MODEL)
        # Everything for ONE model, in order, then free its disk. Grouping per
        # model is what makes cleanup safe: LANG's activation cache is read by
        # SPARSITY and JOINT, so it can only be deleted once all of them are
        # done -- which is guaranteed here because they are the same task.
        # Format: MODEL <model> <nw_lang> <nw_pereira> <nw_sem> [pereira flags...]
        read -r MODEL NWL NWP NWS FLAGS <<< "$REST"
        MM_B="$MODEL"
        # The MM reference is pythia-160m; its own pair partner is the seed twin.
        [ "$MODEL" = "EleutherAI/pythia-160m" ] && MM_B="EleutherAI/pythia-160m-seed2"
        RC=0
        # PIPELINE_STEPS selects which steps to run (default: all). Set by
        # run_all_analyses.sh --steps, e.g. --steps LANG,JOINT to rerun a subset.
        STEPS="${PIPELINE_STEPS:-LANG,PEREIRA,MM,SEM,SPARSITY,JOINT}"
        want_step() { case ",$STEPS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
        run_step() {  # run_step <label> <step-key> <cmd...>
            local label="$1"; local key="$2"; shift 2
            if ! want_step "$key"; then
                echo; echo "-------- [$MODEL] $label  SKIPPED (not in PIPELINE_STEPS=$STEPS) --------"
                return 0
            fi
            echo; echo "-------- [$MODEL] $label  ($(date +%H:%M:%S)) --------"; echo
            "$@"
            local rc=$?
            [ $rc -ne 0 ] && { echo "[STEP-FAILED] $MODEL / $label rc=$rc"; RC=$rc; }
            return 0
        }
        echo "[STEPS] $MODEL will run: $STEPS"
        run_step LANG LANG python LANG_selectivity.py --modelname "$MODEL" --n_workers_per_gpu "$NWL"
        # Brain targets: repetition-averaged voxels, main-text set (top20rel) then
        # SI set (all). PEREIRA_VOXSETS overrides, e.g. PEREIRA_VOXSETS=all.
        for VS in ${PEREIRA_VOXSETS:-top20rel all}; do
            run_step "PEREIRA-expt2-$VS" PEREIRA python PEREIRA_alignment_gpu.py --modelname "$MODEL" \
                --n_workers_per_gpu "$NWP" --only_expt 2 --voxels "$VS" $FLAGS
            run_step "PEREIRA-expt3-$VS" PEREIRA python PEREIRA_alignment_gpu.py --modelname "$MODEL" \
                --n_workers_per_gpu "$NWP" --only_expt 3 --voxels "$VS" $FLAGS
        done
        run_step MM MM python other/model_model_alignment.py             --model_a EleutherAI/pythia-160m --model_b "$MM_B" --n_epochs 50
        if [ "${WITH_SEM:-0}" = "1" ]; then
            run_step SEM SEM python SEM_alignment.py --modelname "$MODEL" --n_workers_per_gpu "$NWS"
        fi
        run_step SPARSITY SPARSITY python other/sparsity_analysis.py --modelname "$MODEL" --n_seeds 10
        run_step JOINT JOINT python other/JOINT_optimal_features.py --modelname "$MODEL" --n_epochs 50
        # Only reclaim disk if every step succeeded: a failed step will be
        # resubmitted and would otherwise have to re-download and re-extract.
        if [ "${NO_CLEANUP:-0}" = "1" ]; then
            echo; echo "-------- [$MODEL] cleanup skipped (--no-cleanup) --------"
        elif [ $RC -eq 0 ]; then
            echo; echo "-------- [$MODEL] cleanup: freeing activations + weights --------"
            python other/cleanup_model.py --model "$MODEL" || echo "[WARN] cleanup failed (non-fatal)"
        else
            echo; echo "-------- [$MODEL] cleanup SKIPPED: a step failed (rc=$RC); caches kept for the resubmit --------"
        fi
        MODEL_RC=$RC   # not `exit`, so the common summary banner below still prints
        ;;
    PEREIRA)
        read -r MODEL NW FLAGS <<< "$REST"
        python PEREIRA_alignment_gpu.py --modelname "$MODEL" --n_workers_per_gpu "$NW" $FLAGS
        ;;
    LANG)
        read -r MODEL NW <<< "$REST"
        python LANG_selectivity.py --modelname "$MODEL" --n_workers_per_gpu "$NW"
        ;;
    SEM)
        read -r MODEL NW <<< "$REST"
        python SEM_alignment.py --modelname "$MODEL" --n_workers_per_gpu "$NW"
        ;;
    MM)
        read -r A B <<< "$REST"
        python other/model_model_alignment.py --model_a "$A" --model_b "$B" --n_epochs 50
        ;;
    CKPT)
        read -r A B <<< "$REST"
        python other/model_model_checkpoints.py --model_a "$A" --model_b "$B" \
            --n_checkpoints 50 --n_seeds 10 --n_epochs 50
        ;;
    RSA)
        read -r A B <<< "$REST"
        python other/model_model_checkpoints_rsa.py --model_a "$A" --model_b "$B" \
            --n_checkpoints 50
        ;;
    SPARSITY)
        read -r MODEL _ <<< "$REST"
        python other/sparsity_analysis.py --modelname "$MODEL" --n_seeds 10
        ;;
    JOINT)
        read -r MODEL _ <<< "$REST"
        python other/JOINT_optimal_features.py --modelname "$MODEL" --n_epochs 50
        ;;
    *)
        echo "ERROR: unknown job type '$TYPE' in $MANIFEST line: $LINE"
        exit 1
        ;;
esac
# MODEL sets MODEL_RC (aggregated over its steps); every other type reports $?
RC=${MODEL_RC:-$?}
echo "=========================================================================="
echo " [task ${SLURM_ARRAY_TASK_ID}] $TYPE finished rc=$RC after $((SECONDS-START))s ($(date))"
echo "=========================================================================="
exit $RC
